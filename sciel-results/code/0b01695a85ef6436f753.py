import math
import random
import time

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Target is L. We want sum(weights[S]) < L, maximizing sum.
    # Deficit = L - sum(weights[S])
    
    best_deficit = L  # Initial max deficit
    best_set = set()
    
    start_time = time.time()
    time_limit = 85  # Safety margin
    
    def get_deficit(S, current_sum):
        return L - current_sum

    def local_search(initial_set, initial_sum):
        S = set(initial_set)
        current_sum = initial_sum
        deficit = L - current_sum
        
        # If initial sum is already >= L, we must remove items
        while current_sum >= L:
            # Remove item with smallest weight? Or largest? 
            # We want to get < L. Removing any item reduces sum.
            # To minimize deficit (maximize sum), we should remove the item 
            # that leaves the sum closest to L from below.
            # This is equivalent to finding the item whose removal 
            # results in the largest sum < L.
            # i.e., minimize (current_sum - w[i]) subject to < L.
            # This is just current_sum - w[i] < L.
            # Since current_sum >= L, we need to remove enough.
            # Simplest: remove the item with the smallest weight? 
            # No, we want to keep as much weight as possible.
            # So we want to remove the item that is "least useful" or 
            # allows us to keep the sum high.
            # Actually, if current_sum >= L, we are invalid.
            # We should have ensured validity in the move generation.
            # For initialization, if we use greedy "add if fits", it's always valid.
            # If we use random, it might be invalid.
            # Let's enforce validity in initialization.
            pass
            
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try Add: Find j not in S such that current_sum + w[j] < L
            #    and w[j] is maximized (to reduce deficit most)
            best_add_j = -1
            best_add_w = 0
            for j in range(n):
                if j not in S:
                    if current_sum + weights[j] < L:
                        if weights[j] > best_add_w:
                            best_add_w = weights[j]
                            best_add_j = j
            
            if best_add_j != -1:
                S.add(best_add_j)
                current_sum += best_add_w
                deficit -= best_add_w
                improved = True
                continue # Restart loop as state changed significantly
            
            # 2. Try Swap 1-1: i in S, j not in S
            #    Condition: current_sum - w[i] + w[j] < L
            #    Gain: w[j] - w[i] > 0
            best_swap_i = -1
            best_swap_j = -1
            best_gain = 0
            for i in S:
                wi = weights[i]
                for j in range(n):
                    if j not in S:
                        wj = weights[j]
                        if wj > wi: # Necessary for gain
                            if current_sum - wi + wj < L:
                                gain = wj - wi
                                if gain > best_gain:
                                    best_gain = gain
                                    best_swap_i = i
                                    best_swap_j = j
            
            if best_swap_i != -1:
                S.remove(best_swap_i)
                S.add(best_swap_j)
                current_sum += best_gain
                deficit -= best_gain
                improved = True
                continue
            
            # 3. Try Swap 1-2: i in S, j,k not in S
            #    Condition: current_sum - w[i] + w[j] + w[k] < L
            #    Gain: w[j] + w[k] - w[i] > 0
            #    Precompute pairs? n=47, pairs ~ 1100. 
            #    Iterating S (up to 47) * Pairs (1100) is ~ 50k checks. Fast.
            best_swap_1_2_i = -1
            best_swap_1_2_j = -1
            best_swap_1_2_k = -1
            best_gain_1_2 = 0
            
            # To speed up, we can just iterate all j,k not in S
            # But we need to check if they are not in S.
            # Let's iterate j < k
            for i in S:
                wi = weights[i]
                # We need w[j] + w[k] > wi
                # And current_sum - wi + w[j] + w[k] < L
                # => w[j] + w[k] < L - current_sum + wi
                max_allowed = L - current_sum + wi
                # Find max w[j]+w[k] < max_allowed and > wi
                # We can just check all pairs
                for j in range(n):
                    if j in S: continue
                    wj = weights[j]
                    # Pruning: if wj >= max_allowed, no k can work? 
                    # Not necessarily, if w[k] is small. But w[k] >= 1.
                    # Let's just check all k > j
                    for k in range(j+1, n):
                        if k in S: continue
                        wk = weights[k]
                        sum_jk = wj + wk
                        if sum_jk > wi and sum_jk < max_allowed:
                            gain = sum_jk - wi
                            if gain > best_gain_1_2:
                                best_gain_1_2 = gain
                                best_swap_1_2_i = i
                                best_swap_1_2_j = j
                                best_swap_1_2_k = k
            
            if best_swap_1_2_i != -1:
                S.remove(best_swap_1_2_i)
                S.add(best_swap_1_2_j)
                S.add(best_swap_1_2_k)
                current_sum += best_gain_1_2
                deficit -= best_gain_1_2
                improved = True
                continue

            # 4. Try Swap 2-1: i,k in S, j not in S
            #    Condition: current_sum - w[i] - w[k] + w[j] < L
            #    Gain: w[j] - w[i] - w[k] > 0
            best_swap_2_1_i = -1
            best_swap_2_1_k = -1
            best_swap_2_1_j = -1
            best_gain_2_1 = 0
            
            # Iterate pairs in S
            S_list = list(S)
            for idx_i in range(len(S_list)):
                i = S_list[idx_i]
                wi = weights[i]
                for idx_k in range(idx_i+1, len(S_list)):
                    k = S_list[idx_k]
                    wk = weights[k]
                    sum_ik = wi + wk
                    # We need w[j] > sum_ik
                    # And current_sum - sum_ik + w[j] < L
                    # => w[j] < L - current_sum + sum_ik
                    max_allowed = L - current_sum + sum_ik
                    # Find max w[j] < max_allowed and > sum_ik
                    # Iterate all j not in S
                    for j in range(n):
                        if j in S: continue
                        wj = weights[j]
                        if sum_ik < wj < max_allowed:
                            gain = wj - sum_ik
                            if gain > best_gain_2_1:
                                best_gain_2_1 = gain
                                best_swap_2_1_i = i
                                best_swap_2_1_k = k
                                best_swap_2_1_j = j
            
            if best_swap_2_1_i != -1:
                S.remove(best_swap_2_1_i)
                S.remove(best_swap_2_1_k)
                S.add(best_swap_2_1_j)
                current_sum += best_gain_2_1
                deficit -= best_gain_2_1
                improved = True
                continue
                
        return S, deficit

    # Initialization Strategies
    # 1. Greedy: Sort by weight desc, add if fits
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    greedy_set = set()
    greedy_sum = 0
    for i in sorted_indices:
        if greedy_sum + weights[i] < L:
            greedy_set.add(i)
            greedy_sum += weights[i]
            
    # Run local search from greedy
    s, d = local_search(greedy_set, greedy_sum)
    if d < best_deficit:
        best_deficit = d
        best_set = s

    # 2. Random Restarts
    # Try random subsets of various sizes
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Random size between 5 and 20
        size = random.randint(5, 20)
        random_set = set(random.sample(range(n), size))
        random_sum = sum(weights[i] for i in random_set)
        
        # If invalid, try to fix by removing items? 
        # Or just skip if invalid? 
        # Better: if invalid, remove items until valid.
        if random_sum >= L:
            # Remove smallest weights until valid?
            # Or just skip this start?
            # Let's try to repair: remove item with largest weight? No, that reduces sum most.
            # We want to be close to L.
            # Let's just skip invalid starts to keep it simple, 
            # or use a "kick" from a valid state.
            # Actually, let's just use valid starts.
            # Generate valid random start:
            valid_set = set()
            valid_sum = 0
            # Random greedy
            random.shuffle(sorted_indices)
            for i in sorted_indices:
                if valid_sum + weights[i] < L and len(valid_set) < 30:
                    if random.random() < 0.7: # Probability to add
                        valid_set.add(i)
                        valid_sum += weights[i]
            
            s, d = local_search(valid_set, valid_sum)
            if d < best_deficit:
                best_deficit = d
                best_set = s
                # Optional: break if deficit is 1? (Optimal)
                if best_deficit == 1:
                    break

    # Return indices
    return list(best_set)
