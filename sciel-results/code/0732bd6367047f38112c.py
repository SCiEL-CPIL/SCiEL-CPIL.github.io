import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            ap3s.append((a, a+d, a+2*d))
            
    ap4s = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap4s.append((a, a+d, a+2*d, a+3*d))
            
    # Map element to AP indices
    elem_to_ap3 = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)
    
    # Convert to numpy arrays for faster operations
    ap3_array = np.array(ap3s)  # shape: (num_3aps, 3)
    ap4_array = np.array(ap4s)  # shape: (num_4aps, 4)
    
    # Precompute bitmask representations for each element
    # Use Python ints as bitmasks since N=256 fits in 256 bits
    elem_mask = [0] * (N+1)
    for i in range(1, N+1):
        elem_mask[i] = 1 << (i-1)
    
    def count_3aps_fast(S_set):
        """Count 3-APs using numpy vectorization"""
        if not S_set:
            return 0
        # Create membership vector
        in_set = np.zeros(N+1, dtype=bool)
        for x in S_set:
            in_set[x] = True
        
        # For each 3-AP, check if all 3 elements are in set
        # ap3_array has shape (num_3aps, 3)
        # in_set[ap3_array] gives shape (num_3aps, 3)
        all_in = np.all(in_set[ap3_array], axis=1)
        return int(np.sum(all_in))
    
    def has_4ap_fast(S_set):
        """Check if S contains any 4-AP using numpy"""
        if not S_set:
            return False
        in_set = np.zeros(N+1, dtype=bool)
        for x in S_set:
            in_set[x] = True
        
        # ap4_array has shape (num_4aps, 4)
        all_in = np.all(in_set[ap4_array], axis=1)
        return bool(np.any(all_in))
    
    def compute_score_delta_add(S_set, y):
        """Compute delta in 3-AP count when adding y to S"""
        delta = 0
        for idx in elem_to_ap3[y]:
            a, b, c = ap3s[idx]
            others = [x for x in (a, b, c) if x != y]
            if all(o in S_set for o in others):
                delta += 1
        return delta
    
    def compute_score_delta_remove(S_set, x):
        """Compute delta in 3-AP count when removing x from S"""
        delta = 0
        for idx in elem_to_ap3[x]:
            a, b, c = ap3s[idx]
            others = [o for o in (a, b, c) if o != x]
            if all(o in S_set for o in others):
                delta += 1
        return delta
    
    def is_add_valid(S_set, y):
        """Check if adding y to S creates a 4-AP"""
        for idx in elem_to_ap4[y]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != y]
            if all(o in S_set for o in others):
                return False
        return True
    
    def is_swap_valid(S_set, x, y):
        """Check if swapping x for y in S is valid (no 4-AP)"""
        # After swap, S' = (S - {x}) ∪ {y}
        # A 4-AP is formed if there's a 4-AP containing y where other 3 are in S - {x}
        for idx in elem_to_ap4[y]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != y]
            # All others must be in S and none can be x
            all_in_S_minus_x = True
            for o in others:
                if o not in S_set or o == x:
                    all_in_S_minus_x = False
                    break
            if all_in_S_minus_x:
                return False
        return True
    
    def simulated_annealing(initial_S, max_iters=50000):
        """Simulated annealing with geometric cooling"""
        S = set(initial_S)
        current_score = count_3aps_fast(S)
        best_S = set(S)
        best_score = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        not_in_S = [x for x in range(1, N+1) if x not in S]
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            # Choose move type: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(S) < 200:
                # Add a random element
                if not not_in_S:
                    continue
                y = random.choice(not_in_S)
                if not is_add_valid(S, y):
                    continue
                
                delta = compute_score_delta_add(S, y)
                new_S = S | {y}
                new_score = current_score + delta
                
                # Metropolis acceptance
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    not_in_S.remove(y)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
            elif move_type < 0.5 and len(S) > 50:
                # Remove a random element
                x = random.choice(list(S))
                delta = -compute_score_delta_remove(S, x)
                new_S = S - {x}
                new_score = current_score + delta
                
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    not_in_S.append(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
            else:
                # Swap: remove x, add y
                if not S or not not_in_S:
                    continue
                x = random.choice(list(S))
                y = random.choice(not_in_S)
                
                if not is_swap_valid(S, x, y):
                    continue
                
                # Compute delta
                remove_delta = compute_score_delta_remove(S, x)
                add_delta = compute_score_delta_add(S - {x}, y)
                delta = add_delta - remove_delta
                
                new_S = (S - {x}) | {y}
                new_score = current_score + delta
                
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    not_in_S.remove(y)
                    not_in_S.append(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
            
            T *= cooling_rate
        
        return best_S, best_score
    
    # Run multiple restarts with different initial solutions
    best_S = set()
    best_score = -1
    
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Generate initial solution using greedy
        S = set()
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > deadline - 10:
                break
            if is_add_valid(S, x):
                S.add(x)
        
        # Run simulated annealing
        S, score = simulated_annealing(S)
        
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Final verification
    if time.time() - start_time > deadline - 2:
        return sorted(best_S)
    
    # Ensure no 4-APs
    for idx in range(len(ap4s)):
        a, b, c, d = ap4s[idx]
        if a in best_S and b in best_S and c in best_S and d in best_S:
            best_S.remove(d)
            break
    
    return sorted(best_S)
