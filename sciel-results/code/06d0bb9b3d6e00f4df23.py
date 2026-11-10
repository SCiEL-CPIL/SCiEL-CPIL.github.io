import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps(in_set, ap3_partners):
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _is_valid_addition(y, in_set, ap4_partners):
    for j, k, l in ap4_partners[y]:
        if in_set[j] and in_set[k] and in_set[l]:
            return False
    return True

def _is_valid_swap(x, y, in_set, ap4_partners):
    # Check if y can be added to S \ {x}
    for j, k, l in ap4_partners[y]:
        if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
            return False
    return True

def _calculate_gain(y, in_set, ap3_partners, exclude_x=None):
    count = 0
    for j, k in ap3_partners[y]:
        j_ok = in_set[j] and (j != exclude_x)
        k_ok = in_set[k] and (k != exclude_x)
        if j_ok and k_ok:
            count += 1
    return count

def _calculate_loss(x, in_set, ap3_partners):
    count = 0
    for j, k in ap3_partners[x]:
        if in_set[j] and in_set[k]:
            count += 1
    return count

def _greedy_init(ap3_partners, ap4_partners, randomness=0.3):
    in_set = [False] * (N + 1)
    current_candidates = list(range(1, N + 1))
    random.shuffle(current_candidates)
    
    while current_candidates:
        valid_candidates = []
        gains = {}
        
        for y in current_candidates:
            if _is_valid_addition(y, in_set, ap4_partners):
                gain = _calculate_gain(y, in_set, ap3_partners)
                valid_candidates.append(y)
                gains[y] = gain
        
        if not valid_candidates:
            break
        
        # Select from top K with some randomness
        K = min(8, len(valid_candidates))
        valid_candidates.sort(key=lambda x: gains[x], reverse=True)
        top_k = valid_candidates[:K]
        
        # Weighted random selection favoring higher gains
        weights = [gains[y] + 1 for y in top_k]  # +1 to avoid zero weights
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        chosen = top_k[0]
        for i, y in enumerate(top_k):
            cum += weights[i]
            if r <= cum:
                chosen = y
                break
        
        in_set[chosen] = True
        current_candidates.remove(chosen)
        
    return in_set

def _local_search(in_set, ap3_partners, ap4_partners, max_steps=3000, time_limit=5.0):
    start_time = time.time()
    score = _count_3aps(in_set, ap3_partners)
    best_score = score
    best_set = in_set[:]
    
    tabu_list = []
    tabu_tenure = 15
    no_improve_steps = 0
    max_no_improve = 80
    steps = 0
    
    present = [i for i in range(1, N + 1) if in_set[i]]
    absent = [i for i in range(1, N + 1) if not in_set[i]]
    
    while steps < max_steps and no_improve_steps < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        best_delta = -10**9
        best_move = None
        
        # 1. Add y
        for y in absent:
            if _is_valid_addition(y, in_set, ap4_partners):
                is_tabu = any(move[0] == 0 and move[1] == y for move in tabu_list)
                if not is_tabu:
                    delta = _calculate_gain(y, in_set, ap3_partners)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (0, y)
        
        # 2. Remove x
        for x in present:
            is_tabu = any(move[0] == 1 and move[1] == x for move in tabu_list)
            if not is_tabu:
                delta = -_calculate_loss(x, in_set, ap3_partners)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (1, x)
        
        # 3. Swap x, y - sample to keep it fast
        if present and absent:
            sample_size = min(15, len(present), len(absent))
            sample_present = random.sample(present, sample_size)
            sample_absent = random.sample(absent, sample_size)
            
            for x in sample_present:
                for y in sample_absent:
                    if _is_valid_swap(x, y, in_set, ap4_partners):
                        is_tabu = any((move[0] == 2 and move[1] == x and move[2] == y) or 
                                      (move[0] == 2 and move[1] == y and move[2] == x) 
                                      for move in tabu_list)
                        if not is_tabu:
                            gain_y = _calculate_gain(y, in_set, ap3_partners, exclude_x=x)
                            loss_x = _calculate_loss(x, in_set, ap3_partners)
                            delta = gain_y - loss_x
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (2, x, y)
        
        if best_move is None:
            break
        
        # Apply best move
        if best_move[0] == 0:  # Add
            y = best_move[1]
            in_set[y] = True
            absent.remove(y)
            present.append(y)
            score += best_delta
            tabu_list.append((0, y))
        elif best_move[0] == 1:  # Remove
            x = best_move[1]
            in_set[x] = False
            present.remove(x)
            absent.append(x)
            score += best_delta
            tabu_list.append((1, x))
        else:  # Swap
            x, y = best_move[1], best_move[2]
            in_set[x] = False
            in_set[y] = True
            present.remove(x)
            present.append(y)
            absent.remove(y)
            absent.append(x)
            score += best_delta
            tabu_list.append((2, x, y))
        
        # Update best
        if score > best_score:
            best_score = score
            best_set = in_set[:]
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        # Update tabu list
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        steps += 1
        
    return best_score, best_set

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    # Multiple restarts with local search
    while time.time() - start_time < DEADLINE - 3.0:
        # Greedy initialization
        in_set = _greedy_init(ap3_partners, ap4_partners)
        
        # Local search
        score, refined_set = _local_search(in_set, ap3_partners, ap4_partners, 
                                           max_steps=2000, time_limit=4.0)
        
        if score > best_score:
            best_score = score
            best_set = refined_set[:]
        
        # Check if we're running low on time
        if time.time() - start_time > DEADLINE - 10.0:
            # Do one more quick local search from best
            if best_score > 0:
                score, refined_set = _local_search(best_set, ap3_partners, ap4_partners,
                                                   max_steps=1000, time_limit=3.0)
                if score > best_score:
                    best_score = score
                    best_set = refined_set[:]
            break
    
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
