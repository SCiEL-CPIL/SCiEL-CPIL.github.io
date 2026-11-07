import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 256
    
    # 1. Identify fixed elements: 1 and primes > n/2
    # Primes p > 128 have no multiples in 1..256 other than themselves, so they conflict with no other number.
    # 1 conflicts with nothing.
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    large_primes = [p for p in primes if p > n // 2]
    
    fixed = [1] + large_primes
    fixed_set = set(fixed)
    fixed_sum = sum(fixed)
    
    # Candidates are the rest: 2..128 and primes <= 128
    candidates = [i for i in range(2, n + 1) if i not in fixed_set]
    num_cands = len(candidates)
    
    # Map value to index in candidates
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if candidates[i] and candidates[j] are NOT coprime
    conflict_mask = [0] * num_cands
    for i in range(num_cands):
        vi = candidates[i]
        for j in range(i + 1, num_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    def get_sum(mask):
        s = 0
        for i in range(num_cands):
            if (mask >> i) & 1:
                s += candidates[i]
        return s

    def greedy_initial(order):
        mask = 0
        for v in order:
            idx = val_to_idx[v]
            # Check if idx conflicts with any currently selected
            if (conflict_mask[idx] & mask) == 0:
                mask |= (1 << idx)
        return mask

    def local_search(initial_mask, time_limit):
        current_mask = initial_mask
        current_sum = get_sum(current_mask)
        
        # Precompute sorted indices by value descending
        indices_sorted_desc = sorted(range(num_cands), key=lambda i: candidates[i], reverse=True)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # --- 1-for-1 Swap ---
            best_gain = 0
            best_move_11 = None
            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                mask_without_j = current_mask & ~(1 << j)
                
                for i in range(num_cands):
                    if ((current_mask >> i) & 1):
                        continue
                    val_i = candidates[i]
                    if val_i <= val_j:
                        continue
                    if (conflict_mask[i] & mask_without_j) == 0:
                        gain = val_i - val_j
                        if gain > best_gain:
                            best_gain = gain
                            best_move_11 = (i, j)
                            
            if best_move_11:
                i, j = best_move_11
                current_mask &= ~(1 << j)
                current_mask |= (1 << i)
                current_sum += best_gain
                improved = True
                continue
            
            # --- 1-for-2 Swap ---
            # Remove j, add i1, i2
            best_gain_12 = 0
            best_move_12 = None
            K = 25
            unselected_indices = [i for i in indices_sorted_desc if not ((current_mask >> i) & 1)]
            top_unselected = unselected_indices[:K]
            
            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                mask_without_j = current_mask & ~(1 << j)
                
                comp_unselected = []
                for i in top_unselected:
                    if (conflict_mask[i] & mask_without_j) == 0:
                        comp_unselected.append(i)
                        
                for a in range(len(comp_unselected)):
                    i1 = comp_unselected[a]
                    val_i1 = candidates[i1]
                    for b in range(a + 1, len(comp_unselected)):
                        i2 = comp_unselected[b]
                        val_i2 = candidates[i2]
                        if math.gcd(candidates[i1], candidates[i2]) == 1:
                            gain = val_i1 + val_i2 - val_j
                            if gain > best_gain_12:
                                best_gain_12 = gain
                                best_move_12 = (i1, i2, j)
                                
            if best_move_12:
                i1, i2, j = best_move_12
                current_mask &= ~(1 << j)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_12
                improved = True
                continue
            
            # --- 2-for-1 Swap ---
            # Remove j1, j2, add i
            best_gain_21 = 0
            best_move_21 = None
            top_unselected_1 = top_unselected[:8]
            
            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    
                    for i in top_unselected_1:
                        val_i = candidates[i]
                        if (conflict_mask[i] & mask_without_j1j2) == 0:
                            gain = val_i - (val_j1 + val_j2)
                            if gain > best_gain_21:
                                best_gain_21 = gain
                                best_move_21 = (i, j1, j2)
                                
            if best_move_21:
                i, j1, j2 = best_move_21
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i)
                current_sum += best_gain_21
                improved = True
                continue

            # --- 2-for-2 Swap ---
            # Remove j1, j2, add i1, i2.
            best_gain_22 = 0
            best_move_22 = None
            top_unselected_2 = top_unselected[:8]
            
            valid_unselected_pairs = []
            for a in range(len(top_unselected_2)):
                i1 = top_unselected_2[a]
                for b in range(a + 1, len(top_unselected_2)):
                    i2 = top_unselected_2[b]
                    if math.gcd(candidates[i1], candidates[i2]) == 1:
                        valid_unselected_pairs.append((i1, i2))
                        
            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    
                    for i1, i2 in valid_unselected_pairs:
                        if (conflict_mask[i1] & mask_without_j1j2) == 0 and \
                           (conflict_mask[i2] & mask_without_j1j2) == 0:
                            gain = candidates[i1] + candidates[i2] - val_j1 - val_j2
                            if gain > best_gain_22:
                                best_gain_22 = gain
                                best_move_22 = (i1, i2, j1, j2)
                                
            if best_move_22:
                i1, i2, j1, j2 = best_move_22
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_22
                improved = True
                continue

        return current_mask, current_sum

    # Main Loop: Multiple Restart
    best_mask = 0
    best_total_sum = fixed_sum
    
    # Initial Orders
    orders = []
    # 1. Descending Value
    orders.append(sorted(candidates, reverse=True))
    # 2. Ascending Value
    orders.append(sorted(candidates))
    # 3. Least Conflicts First
    conflict_counts = [bin(conflict_mask[i]).count('1') for i in range(num_cands)]
    order_least_conflict = sorted(candidates, key=lambda v: (conflict_counts[val_to_idx[v]], -v))
    orders.append(order_least_conflict)
    
    # 4. Random Shuffles
    rng = random.Random(42)
    for _ in range(15):
        rand_order = candidates.copy()
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # Allocate time per restart
    # We have ~55s. Precomputation is fast.
    # Let's do the first few restarts with more time, later ones with less?
    # Or just fixed time per restart.
    # 20 restarts * 2.5s = 50s.
    time_per_restart = 2.5
    
    for idx, order in enumerate(orders):
        if time.time() > deadline - 2.0: # Leave 2s for final construction
            break
            
        init_mask = greedy_initial(order)
        # Use a slightly shorter deadline for local search to ensure we return in time
        # But if it's the last one, use the remaining time.
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        ls_time_limit = min(time_per_restart, remaining - 0.5)
        
        mask, sum_cands = local_search(init_mask, start_time + ls_time_limit)
        total = sum_cands + fixed_sum
        if total > best_total_sum:
            best_total_sum = total
            best_mask = mask
            
    # Construct Result
    result = list(fixed)
    for i in range(num_cands):
        if (best_mask >> i) & 1:
            result.append(candidates[i])
            
    # Ensure types are int
    result = [int(x) for x in result]
    
    return result
