import time
import random
import numpy as np

N = 256
DEADLINE = 55.0

def _precompute():
    ap3_u = [None] * (N + 1)
    ap3_v = [None] * (N + 1)
    ap4_u = [None] * (N + 1)
    ap4_v = [None] * (N + 1)
    ap4_w = [None] * (N + 1)
    
    # Initialize empty arrays
    for i in range(1, N + 1):
        ap3_u[i] = np.array([], dtype=np.int32)
        ap3_v[i] = np.array([], dtype=np.int32)
        ap4_u[i] = np.array([], dtype=np.int32)
        ap4_v[i] = np.array([], dtype=np.int32)
        ap4_w[i] = np.array([], dtype=np.int32)

    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            
            # For a: partners are b, c
            ap3_u[a] = np.append(ap3_u[a], b)
            ap3_v[a] = np.append(ap3_v[a], c)
            
            # For b: partners are a, c
            ap3_u[b] = np.append(ap3_u[b], a)
            ap3_v[b] = np.append(ap3_v[b], c)
            
            # For c: partners are a, b
            ap3_u[c] = np.append(ap3_u[c], a)
            ap3_v[c] = np.append(ap3_v[c], b)

    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For a: partners b, c, e
            ap4_u[a] = np.append(ap4_u[a], b)
            ap4_v[a] = np.append(ap4_v[a], c)
            ap4_w[a] = np.append(ap4_w[a], e)
            
            # For b: partners a, c, e
            ap4_u[b] = np.append(ap4_u[b], a)
            ap4_v[b] = np.append(ap4_v[b], c)
            ap4_w[b] = np.append(ap4_w[b], e)
            
            # For c: partners a, b, e
            ap4_u[c] = np.append(ap4_u[c], a)
            ap4_v[c] = np.append(ap4_v[c], b)
            ap4_w[c] = np.append(ap4_w[c], e)
            
            # For e: partners a, b, c
            ap4_u[e] = np.append(ap4_u[e], a)
            ap4_v[e] = np.append(ap4_v[e], b)
            ap4_w[e] = np.append(ap4_w[e], c)
            
    return ap3_u, ap3_v, ap4_u, ap4_v, ap4_w

def _count_3aps(S, ap3_u, ap3_v):
    total = 0
    for i in range(1, N + 1):
        if S[i]:
            if len(ap3_u[i]) > 0:
                total += np.sum(S[ap3_u[i]] & S[ap3_v[i]])
    return total // 3

def _is_valid_addition(y, S, ap4_u, ap4_v, ap4_w):
    if len(ap4_u[y]) == 0:
        return True
    # Check if any triple is fully in S
    return not np.any(S[ap4_u[y]] & S[ap4_v[y]] & S[ap4_w[y]])

def _is_valid_swap(x, y, S, ap4_u, ap4_v, ap4_w):
    # Check if y can be added to S \ {x}
    # Temporarily remove x
    S[x] = False
    valid = _is_valid_addition(y, S, ap4_u, ap4_v, ap4_w)
    S[x] = True
    return valid

def _calculate_gain(y, S, ap3_u, ap3_v, exclude_x=None):
    if len(ap3_u[y]) == 0:
        return 0
    if exclude_x is None:
        return np.sum(S[ap3_u[y]] & S[ap3_v[y]])
    else:
        # Mask out exclude_x
        u = ap3_u[y]
        v = ap3_v[y]
        # We need to ensure neither u nor v is exclude_x
        # Since exclude_x is a single element, we can just mask it
        mask_u = u != exclude_x
        mask_v = v != exclude_x
        if not np.all(mask_u) or not np.all(mask_v):
             # This filtering is slightly expensive, but exclude_x is rare in gain calc?
             # Actually, for swap, we want gain of y in S\{x}.
             # The pairs (u,v) must be in S\{x}.
             # So we check S[u] & S[v] & (u != x) & (v != x).
             # Since u,v are indices, we can just do:
             valid_mask = (u != exclude_x) & (v != exclude_x)
             if not np.any(valid_mask):
                 return 0
             u_valid = u[valid_mask]
             v_valid = v[valid_mask]
             return np.sum(S[u_valid] & S[v_valid])
        else:
            return np.sum(S[u] & S[v])

def _calculate_loss(x, S, ap3_u, ap3_v):
    if len(ap3_u[x]) == 0:
        return 0
    return np.sum(S[ap3_u[x]] & S[ap3_v[x]])

def construct():
    start_time = time.time()
    ap3_u, ap3_v, ap4_u, ap4_v, ap4_w = _precompute()
    
    best_score = -1
    best_set = np.zeros(N + 1, dtype=bool)
    
    num_iterations = 0
    while time.time() - start_time < DEADLINE - 2.0:
        num_iterations += 1
        
        # --- Phase 1: GRASP Initialization ---
        S = np.zeros(N + 1, dtype=bool)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_candidates = candidates[:]
        
        while current_candidates:
            valid_candidates = []
            gains = {}
            
            for y in current_candidates:
                if _is_valid_addition(y, S, ap4_u, ap4_v, ap4_w):
                    gain = _calculate_gain(y, S, ap3_u, ap3_v)
                    valid_candidates.append(y)
                    gains[y] = gain
            
            if not valid_candidates:
                break
            
            # Select from top K
            K = min(5, len(valid_candidates))
            valid_candidates.sort(key=lambda x: gains[x], reverse=True)
            top_k = valid_candidates[:K]
            
            y = random.choice(top_k)
            
            S[y] = True
            current_candidates.remove(y)
            
        score = _count_3aps(S, ap3_u, ap3_v)
        
        if score > best_score:
            best_score = score
            best_set = S.copy()
            
        # --- Phase 2: Local Search ---
        tabu_list = []
        tabu_tenure = 10
        no_improve_steps = 0
        max_no_improve = 50
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and no_improve_steps < max_no_improve:
            if time.time() - start_time > DEADLINE - 1.0:
                break
                
            best_delta = -10**9
            best_move = None
            
            # 1. Add y
            for y in range(1, N + 1):
                if not S[y]:
                    if _is_valid_addition(y, S, ap4_u, ap4_v, ap4_w):
                        is_tabu = any(move[0] == 0 and move[1] == y for move in tabu_list)
                        if not is_tabu:
                            delta = _calculate_gain(y, S, ap3_u, ap3_v)
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (0, y)
            
            # 2. Remove x
            for x in range(1, N + 1):
                if S[x]:
                    is_tabu = any(move[0] == 1 and move[1] == x for move in tabu_list)
                    if not is_tabu:
                        delta = -_calculate_loss(x, S, ap3_u, ap3_v)
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (1, x)
            
            # 3. Swap x, y
            present = [i for i in range(1, N + 1) if S[i]]
            absent = [i for i in range(1, N + 1) if not S[i]]
            
            if present and absent:
                # Increase sample size for swaps
                sample_size = min(20, len(present), len(absent))
                sample_present = random.sample(present, sample_size)
                sample_absent = random.sample(absent, sample_size)
                
                for x in sample_present:
                    for y in sample_absent:
                        if _is_valid_swap(x, y, S, ap4_u, ap4_v, ap4_w):
                            is_tabu = any((move[0] == 2 and move[1] == x and move[2] == y) or 
                                          (move[0] == 2 and move[1] == y and move[2] == x) 
                                          for move in tabu_list)
                            if not is_tabu:
                                gain_y = _calculate_gain(y, S, ap3_u, ap3_v, exclude_x=x)
                                loss_x = _calculate_loss(x, S, ap3_u, ap3_v)
                                delta = gain_y - loss_x
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (2, x, y)
            
            if best_move is None:
                break
            
            # Apply best move
            if best_move[0] == 0: # Add
                y = best_move[1]
                S[y] = True
                score += best_delta
                tabu_list.append((0, y))
            elif best_move[0] == 1: # Remove
                x = best_move[1]
                S[x] = False
                score += best_delta
                tabu_list.append((1, x))
            else: # Swap
                x, y = best_move[1], best_move[2]
                S[x] = False
                S[y] = True
                score += best_delta
                tabu_list.append((2, x, y))
            
            # Update best
            if score > best_score:
                best_score = score
                best_set = S.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            # Update tabu list
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
                
            steps += 1
            
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
