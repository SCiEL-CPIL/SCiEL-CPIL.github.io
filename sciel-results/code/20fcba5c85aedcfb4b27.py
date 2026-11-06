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
    
    # Use numpy boolean array for fast membership checks
    S = np.zeros(N+1, dtype=bool)
    S_set = set()
    
    def count_3aps_fast():
        """Count 3-APs using numpy vectorization"""
        if not S_set:
            return 0
        # S is already a numpy boolean array indexed 1..N
        all_in = np.all(S[ap3_array], axis=1)
        return int(np.sum(all_in))
    
    def has_4ap_fast():
        """Check if S contains any 4-AP using numpy"""
        if not S_set:
            return False
        all_in = np.all(S[ap4_array], axis=1)
        return bool(np.any(all_in))
    
    def compute_delta_add(y):
        """Compute delta in 3-AP count when adding y to S"""
        delta = 0
        for idx in elem_to_ap3[y]:
            a, b, c = ap3s[idx]
            others = [x for x in (a, b, c) if x != y]
            if all(S[o] for o in others):
                delta += 1
        return delta
    
    def compute_delta_remove(x):
        """Compute delta in 3-AP count when removing x from S"""
        delta = 0
        for idx in elem_to_ap3[x]:
            a, b, c = ap3s[idx]
            others = [o for o in (a, b, c) if o != x]
            if all(S[o] for o in others):
                delta += 1
        return delta
    
    def is_add_valid(y):
        """Check if adding y to S creates a 4-AP"""
        for idx in elem_to_ap4[y]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != y]
            if all(S[o] for o in others):
                return False
        return True
    
    def is_swap_valid(x, y):
        """Check if swapping x for y in S is valid (no 4-AP)"""
        # After swap, S' = (S - {x}) ∪ {y}
        # A 4-AP is formed if there's a 4-AP containing y where other 3 are in S - {x}
        for idx in elem_to_ap4[y]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != y]
            all_in_S_minus_x = True
            for o in others:
                if not S[o] or o == x:
                    all_in_S_minus_x = False
                    break
            if all_in_S_minus_x:
                return False
        return True
    
    def greedy_initial_solution():
        """Generate initial solution using greedy with randomization"""
        S_init = np.zeros(N+1, dtype=bool)
        S_init_set = set()
        
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > deadline - 20:
                break
            if is_add_valid_with_array(x, S_init):
                S_init[x] = True
                S_init_set.add(x)
                
        return S_init, S_init_set
    
    def is_add_valid_with_array(y, S_arr):
        """Check if adding y to S (numpy array) creates a 4-AP"""
        for idx in elem_to_ap4[y]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != y]
            if all(S_arr[o] for o in others):
                return False
        return True
    
    def simulated_annealing(S_init, S_init_set, max_iters=100000):
        """Simulated annealing with geometric cooling"""
        S = S_init.copy()
        S_set = set(S_init_set)
        current_score = count_3aps_fast()
        best_S = S.copy()
        best_score = current_score
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        not_in_S = [x for x in range(1, N+1) if x not in S_set]
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            # Choose move type: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(S_set) < 200:
                # Add a random element
                if not not_in_S:
                    continue
                y = random.choice(not_in_S)
                if not is_add_valid(y):
                    continue
                
                delta = compute_delta_add(y)
                new_score = current_score + delta
                
                # Metropolis acceptance
                if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                    S[y] = True
                    S_set.add(y)
                    current_score = new_score
                    not_in_S.remove(y)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                        
            elif move_type < 0.5 and len(S_set) > 50:
                # Remove a random element
                x = random.choice(list(S_set))
                delta = -compute_delta_remove(x)
                new_score = current_score + delta
                
                if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                    S[x] = False
                    S_set.remove(x)
                    current_score = new_score
                    not_in_S.append(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                        
            else:
                # Swap: remove x, add y
                if not S_set or not not_in_S:
                    continue
                x = random.choice(list(S_set))
                y = random.choice(not_in_S)
                
                if not is_swap_valid(x, y):
                    continue
                
                # Compute delta
                remove_delta = compute_delta_remove(x)
                # Temporarily remove x to compute add delta correctly
                S[x] = False
                add_delta = compute_delta_add(y)
                S[x] = True
                delta = add_delta - remove_delta
                
                new_score = current_score + delta
                
                if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                    S[x] = False
                    S[y] = True
                    S_set.remove(x)
                    S_set.add(y)
                    current_score = new_score
                    not_in_S.remove(y)
                    not_in_S.append(x)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
            
            T *= cooling_rate
        
        return best_S, best_score
    
    # Run multiple restarts with different initial solutions
    best_S = None
    best_score = -1
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Generate initial solution using greedy
        S_init, S_init_set = greedy_initial_solution()
        
        # Run simulated annealing
        S_sa, score_sa = simulated_annealing(S_init, S_init_set)
        
        if score_sa > best_score:
            best_score = score_sa
            best_S = S_sa
    
    # Final local search: try to add more elements
    if best_S is not None:
        S_final = best_S.copy()
        S_final_set = set(x for x in range(1, N+1) if S_final[x])
        
        for x in range(1, N+1):
            if time.time() - start_time > deadline - 2:
                break
            if not S_final[x]:
                if is_add_valid_with_array(x, S_final):
                    S_final[x] = True
                    S_final_set.add(x)
        
        # Convert to sorted list
        result = sorted(S_final_set)
    else:
        # Fallback: return empty or a minimal valid set
        result = [1]
    
    return result

construct()
