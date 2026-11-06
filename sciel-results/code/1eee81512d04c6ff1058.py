import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    # Precompute popcount for faster counting
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def fast_popcount(x):
        count = 0
        while x:
            count += popcount[x & 0xFFFF]
            x >>= 16
        return count
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = fast_popcount(current_mask)
    
    # Phase 1: Add evens with 0 conflicts, then try evens with minimal conflicts
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True
    
    # Phase 1b: Try adding evens that cause minimal conflicts (remove those conflicts, add even)
    changed = True
    while changed and time.time() < DEADLINE - 45:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 45:
                break
            if (current_mask >> e) & 1:
                continue
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            # Only add if net gain (remove k conflicts, add 1, net = 1-k)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << e)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True
    
    # Phase 2: Simulated Annealing with improved moves
    T = 2.0
    cooling = 0.9995
    max_iters = 200000
    
    current_size = best_size
    
    # Precompute list of all indices for faster random selection
    all_indices = list(range(N))
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        r = random.random()
        if r < 0.55:
            # Single add move: try to add a random element not in set
            # Prefer evens since they're more likely to have conflicts
            if random.random() < 0.7:
                e_candidates = [idx for idx in evens_indices if not ((current_mask >> idx) & 1)]
                if not e_candidates:
                    continue
                e = random.choice(e_candidates)
            else:
                o_candidates = [idx for idx in odds_indices if not ((current_mask >> idx) & 1)]
                if not o_candidates:
                    continue
                e = random.choice(o_candidates)
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            new_size = current_size - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.4:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
        elif r < 0.75:
            # Swap move: remove 1 element, add 1 element
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if not bits:
                continue
            rm_idx = random.choice(bits)
            
            # Find a candidate to add that doesn't conflict with set minus rm_idx
            temp_mask = current_mask & ~(1 << rm_idx)
            not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue
            e = random.choice(not_in_temp)
            
            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                continue
            
            # New size: same (remove 1, add 1)
            # This is a neutral move for exploration
            if random.random() < 0.3:
                current_mask = temp_mask | (1 << e)
                # current_size unchanged
        elif r < 0.9:
            # Multi-element exchange: remove k elements, add 1
            k = random.choice([2, 3])
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if len(bits) < k:
                continue
            removed = random.sample(bits, k)
            
            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)
            
            not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue
            
            # Prefer candidates with 0 conflicts with temp_mask
            zero_conflict = [idx for idx in not_in_temp if (conflict_mask[idx] & temp_mask) == 0]
            if zero_conflict:
                e = random.choice(zero_conflict)
            else:
                e = random.choice(not_in_temp)
            
            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                # Still might be worth trying if net gain
                num_conf = fast_popcount(conflicts_with_temp)
                new_size = current_size - k - num_conf + 1
                if new_size <= current_size:
                    continue
            
            new_size = current_size - k + 1
            
            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
        else:
            # Random removal to escape local minima
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if bits:
                r_idx = random.choice(bits)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final polish: greedy add any element with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 1:
                break
            if (best_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & best_mask) == 0:
                best_mask |= (1 << idx)
                best_size += 1
                changed = True
    
    # Second polish pass: try adding elements that cause at most 1 conflict if net gain
    changed = True
    while changed and time.time() < DEADLINE - 0.5:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 0.5:
                break
            if (best_mask >> idx) & 1:
                continue
            conflicts = conflict_mask[idx] & best_mask
            num_conf = fast_popcount(conflicts)
            if num_conf <= 1:
                new_mask = (best_mask & ~conflicts) | (1 << idx)
                new_size = best_size - num_conf + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                    changed = True

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
