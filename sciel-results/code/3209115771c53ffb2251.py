"""Optimized 4-AP-free subset of [1,256] maximizing 3-APs using numpy-accelerated local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b := a + d, c))
    
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            e = a + 3 * d
            if e <= N:
                all_4aps.append((a, a + d, a + 2 * d, e))
    
    # Convert to numpy arrays for fast evaluation
    arr_3aps = np.array(all_3aps, dtype=np.int32)
    arr_4aps = np.array(all_4aps, dtype=np.int32)
    
    # For each element x, precompute which 3-APs and 4-APs it participates in
    # 3-AP membership: for each x, list of indices into arr_3aps
    three_membership = [[] for _ in range(N + 1)]
    for idx in range(len(arr_3aps)):
        a, b, c = arr_3aps[idx]
        three_membership[a].append(idx)
        three_membership[b].append(idx)
        three_membership[c].append(idx)
    
    # 4-AP membership
    four_membership = [[] for _ in range(N + 1)]
    for idx in range(len(arr_4aps)):
        a, b, c, e = arr_4aps[idx]
        four_membership[a].append(idx)
        four_membership[b].append(idx)
        four_membership[c].append(idx)
        four_membership[e].append(idx)
    
    def count_3aps(S_present):
        """Count 3-APs using numpy vectorization."""
        # S_present is a boolean numpy array of length N+1
        # For each 3-AP (a,b,c), check if all three are present
        mask = (S_present[arr_3aps[:, 0]] & S_present[arr_3aps[:, 1]] & S_present[arr_3aps[:, 2]])
        return int(np.sum(mask))
    
    def has_4ap(S_present):
        """Check if any 4-AP is fully present."""
        mask = (S_present[arr_4aps[:, 0]] & S_present[arr_4aps[:, 1]] & 
                S_present[arr_4aps[:, 2]] & S_present[arr_4aps[:, 3]])
        return bool(np.any(mask))
    
    def check_4ap_with_x(S_present, x):
        """Check if adding x to S_present would create a 4-AP."""
        for idx in four_membership[x]:
            a, b, c, e = arr_4aps[idx]
            others = [a, b, c, e]
            others.remove(x)
            if all(S_present[o] for o in others):
                return True
        return False
    
    def gain_3aps_with_x(S_present, x):
        """Count how many new 3-APs would be created by adding x to S_present."""
        gain = 0
        for idx in three_membership[x]:
            a, b, c = arr_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_present[o] for o in others):
                gain += 1
        return gain
    
    def loss_3aps_with_removal(S_present, x):
        """Count how many 3-APs would be lost by removing x from S_present."""
        loss = 0
        for idx in three_membership[x]:
            a, b, c = arr_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_present[o] for o in others):
                loss += 1
        return loss
    
    # Degree-based greedy construction
    def degree_greedy():
        S_present = np.zeros(N + 1, dtype=bool)
        S = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_x = None
            best_score = -1
            
            for x in remaining:
                if check_4ap_with_x(S_present, x):
                    continue
                gain = gain_3aps_with_x(S_present, x)
                # Score: gain + small bonus for higher degree (more future flexibility)
                degree = len(three_membership[x])
                score = gain * 10 + degree * 0.1
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            S_present[best_x] = True
            S.add(best_x)
            remaining.remove(best_x)
        
        return S
    
    # Local search with numpy-accelerated evaluation
    def local_search(S, max_iter=5000):
        S_present = np.zeros(N + 1, dtype=bool)
        for x in S:
            S_present[x] = True
        
        current_3ap = count_3aps(S_present)
        best_S = set(S)
        best_3ap = current_3ap
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding each element
            for x in range(1, N + 1):
                if S_present[x]:
                    continue
                if check_4ap_with_x(S_present, x):
                    continue
                gain = gain_3aps_with_x(S_present, x)
                if gain > 0:
                    S_present[x] = True
                    S.add(x)
                    current_3ap += gain
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = set(S)
                    improved = True
            
            # Try swaps: remove one, add another
            if not improved:
                for x_remove in list(S):
                    if time.time() > deadline:
                        break
                    loss = loss_3aps_with_removal(S_present, x_remove)
                    S_present[x_remove] = False
                    S.discard(x_remove)
                    current_3ap -= loss
                    
                    best_swap = None
                    best_swap_gain = -1
                    
                    for x_add in range(1, N + 1):
                        if S_present[x_add]:
                            continue
                        if check_4ap_with_x(S_present, x_add):
                            continue
                        gain = gain_3aps_with_x(S_present, x_add)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = x_add
                    
                    if best_swap is not None and best_swap_gain > loss:
                        S_present[best_swap] = True
                        S.add(best_swap)
                        current_3ap += best_swap_gain
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_S = set(S)
                        improved = True
                    else:
                        # Revert
                        S_present[x_remove] = True
                        S.add(x_remove)
                        current_3ap += loss
            
            if not improved:
                break
        
        return best_S, best_3ap
    
    # Simulated annealing
    def simulated_annealing(S, max_iter=10000, T_init=5.0, T_final=0.1, cooling=0.9995):
        S_present = np.zeros(N + 1, dtype=bool)
        for x in S:
            S_present[x] = True
        
        current_3ap = count_3aps(S_present)
        best_S = set(S)
        best_3ap = current_3ap
        
        T = T_init
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            if T < T_final:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                candidates = [x for x in range(1, N + 1) if not S_present[x]]
                if not candidates:
                    T *= cooling
                    continue
                x = random.choice(candidates)
                if check_4ap_with_x(S_present, x):
                    T *= cooling
                    continue
                gain = gain_3aps_with_x(S_present, x)
                S_present[x] = True
                S.add(x)
                current_3ap += gain
                delta = gain
                
            elif move_type == 'remove':
                if not S:
                    T *= cooling
                    continue
                x = random.choice(list(S))
                loss = loss_3aps_with_removal(S_present, x)
                S_present[x] = False
                S.discard(x)
                current_3ap -= loss
                delta = -loss
            
            else:  # swap
                if not S:
                    T *= cooling
                    continue
                x_remove = random.choice(list(S))
                candidates = [x for x in range(1, N + 1) if not S_present[x]]
                if not candidates:
                    T *= cooling
                    continue
                x_add = random.choice(candidates)
                if check_4ap_with_x(S_present, x_add):
                    T *= cooling
                    continue
                
                loss = loss_3aps_with_removal(S_present, x_remove)
                S_present[x_remove] = False
                S.discard(x_remove)
                current_3ap -= loss
                
                gain = gain_3aps_with_x(S_present, x_add)
                S_present[x_add] = True
                S.add(x_add)
                current_3ap += gain
                delta = gain - loss
            
            # Metropolis acceptance
            if delta >= 0 or random.random() < np.exp(delta / T):
                if current_3ap > best_3ap:
                    best_3ap = current_3ap
                    best_S = set(S)
            else:
                # Revert
                if move_type == 'add':
                    S_present[x] = False
                    S.discard(x)
                    current_3ap -= gain
                elif move_type == 'remove':
                    S_present[x] = True
                    S.add(x)
                    current_3ap += loss
                else:  # swap
                    S_present[x_add] = False
                    S.discard(x_add)
                    current_3ap -= gain
                    S_present[x_remove] = True
                    S.add(x_remove)
                    current_3ap += loss
            
            T *= cooling
        
        return best_S, best_3ap
    
    # Run degree-based greedy
    S = degree_greedy()
    if not S:
        S = {1}
    
    # Run local search
    best_S, best_3ap = local_search(S, max_iter=3000)
    
    # Run simulated annealing from the best found
    if time.time() < deadline - 5:
        sa_S, sa_3ap = simulated_annealing(best_S, max_iter=5000, T_init=3.0, T_final=0.05, cooling=0.9998)
        if sa_3ap > best_3ap:
            best_S = sa_S
            best_3ap = sa_3ap
    
    # Try a few random restarts with perturbation if time allows
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        
        perturbed = set(best_S)
        for _ in range(random.randint(3, 6)):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        
        result_S, result_3ap = local_search(perturbed, max_iter=1500)
        if result_3ap > best_3ap:
            best_S = result_S
            best_3ap = result_3ap
    
    return sorted(best_S)
