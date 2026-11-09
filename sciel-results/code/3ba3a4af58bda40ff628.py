import numpy as np
import random
import time
from collections import deque

def construct():
    n = 192
    max_time = 55.0
    start_time = time.time()
    
    # 1. Initialization: Greedy with heuristic
    # Heuristic: Prefer numbers that are "hard to divide" (large, prime)
    # and "hard to divide others" (large).
    # Actually, large numbers are safer as divisors (fewer multiples in range).
    # Small numbers are dangerous divisors.
    # So, sort candidates by value descending? Or by number of divisors ascending?
    # Let's try sorting by value descending first, as it's a strong signal.
    
    def get_greedy_init():
        selected = []
        # Try a few different orderings
        for strategy in ['desc', 'prime_first', 'random']:
            sel = []
            if strategy == 'desc':
                candidates = list(range(n, 0, -1))
            elif strategy == 'prime_first':
                # Simple primality test
                def is_prime(x):
                    if x < 2: return False
                    for i in range(2, int(x**0.5)+1):
                        if x % i == 0: return False
                    return True
                candidates = [x for x in range(n, 0, -1) if is_prime(x)] + [x for x in range(n, 0, -1) if not is_prime(x)]
            else:
                candidates = list(range(1, n+1))
                random.shuffle(candidates)
                
            for x in candidates:
                if is_valid_add(x, sel):
                    sel.append(x)
            if len(sel) > len(selected):
                selected = sel
        return set(selected)

    best_set = get_greedy_init()
    best_size = len(best_set)
    
    # Tabu Search Parameters
    tabu_length = 50
    tabu_list = deque(maxlen=tabu_length)
    no_improve_count = 0
    max_no_improve = 200
    
    current_set = set(best_set)
    current_size = len(current_set)
    best_global_size = current_size
    
    while time.time() - start_time < max_time:
        # Generate neighbors
        # A move is (remove y, add x)
        # To increase size, we need to find x that is valid with current_set.
        # If we find one, we add it (size + 1).
        # If not, we swap.
        
        # 1. Try to add (Increase size)
        added = False
        candidates_to_add = [x for x in range(1, n+1) if x not in current_set]
        random.shuffle(candidates_to_add)
        
        for x in candidates_to_add:
            if is_valid_add(x, list(current_set)):
                current_set.add(x)
                current_size += 1
                if current_size > best_global_size:
                    best_global_size = current_size
                    best_set = set(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                added = True
                break
        
        if added:
            continue
            
        # 2. If no add, try swap (Maintain size, explore landscape)
        # Pick a random y to remove
        y = random.choice(list(current_set))
        temp_set = current_set - {y}
        temp_list = list(temp_set)
        
        candidates_to_add = [x for x in range(1, n+1) if x not in temp_set and x != y]
        random.shuffle(candidates_to_add)
        
        best_move = None
        best_score = -1
        best_x = None
        
        # We don't have a secondary score, so we just want any valid swap to move out of local min
        # But to be smarter, we can prefer swaps that bring in "safer" numbers?
        # For now, just find the first valid swap.
        
        for x in candidates_to_add:
            if is_valid_add(x, temp_list):
                # Check Tabu
                if (y, x) not in tabu_list:
                    best_x = x
                    best_move = (y, x)
                    break
                else:
                    # Aspiration: If this swap leads to a set that is "better" in some way?
                    # Since size is same, we can't really judge. 
                    # We'll just skip tabu moves unless we are stuck for a long time.
                    if no_improve_count > max_no_improve:
                        best_x = x
                        best_move = (y, x)
                        break
        
        if best_move:
            y, x = best_move
            current_set.remove(y)
            current_set.add(x)
            tabu_list.append((y, x))
            no_improve_count += 1
            
            # Diversification: If stuck for too long, random restart
            if no_improve_count > max_no_improve * 2:
                # Reset to best set found
                current_set = set(best_set)
                current_size = best_global_size
                tabu_list.clear()
                no_improve_count = 0
                # Or do a random perturbation of best_set
                # Let's just reset to best_set to ensure we don't degrade
                # Actually, a kick is better.
                # Remove 5 random elements from best_set, then greedily fill back up
                # For simplicity, just reset.
        
    return list(best_set)

def is_valid_add(x, selected_list):
    if not selected_list:
        return True
        
    S = np.array(selected_list)
    
    # Check 1: x does not divide sum of any 3 in S
    if len(S) >= 3:
        # Vectorized check
        S_3d = S[:, None, None] + S[None, :, None] + S[None, None, :]
        i_idx, j_idx, k_idx = np.indices((len(S), len(S), len(S)))
        mask = (i_idx != j_idx) & (j_idx != k_idx) & (i_idx != k_idx)
        
        if np.any(mask):
            triple_sums = S_3d[mask]
            if np.any(triple_sums % x == 0):
                return False
        
    # Check 2: For each a in S, a does not divide x + u + v
    for a in S:
        T = S[S != a]
        if len(T) < 2:
            continue
        T_2d = T[:, None] + T[None, :]
        i_idx, j_idx = np.indices((len(T), len(T)))
        mask = (i_idx != j_idx)
        if np.any(mask):
            pair_sums = T_2d[mask]
            if np.any((pair_sums + x) % a == 0):
                return False
                
    return True
