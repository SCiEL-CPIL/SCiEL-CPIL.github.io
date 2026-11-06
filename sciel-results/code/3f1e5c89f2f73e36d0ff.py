"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    # (a*b) % (a+b) == 0 means (a+b) divides a*b
    # Equivalent: (a+b) divides a^2 (since a*b = a*(a+b) - a^2)
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Count conflicts for each node
    conflict_count = [0] * N
    for i in range(N):
        cm = conflict_mask[i]
        cnt = 0
        while cm:
            cm &= cm - 1
            cnt += 1
        conflict_count[i] = cnt
    
    # Strategy: Start with all, then remove elements to eliminate conflicts
    # Then try to add back
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    # Phase 1: Greedy removal - remove element with most conflicts among remaining
    current_mask = (1 << N) - 1  # all elements
    
    # Precompute: for each element, list of conflicting elements
    # We'll iteratively remove the element that has the most conflicts within current set
    
    removed = set()
    max_removals = 200  # limit to avoid infinite loop
    
    for _ in range(max_removals):
        if time.time() > DEADLINE - 20:
            break
        
        # Find element in current_mask with maximum conflict count within current_mask
        best_to_remove = -1
        best_conflicts = -1
        
        # Iterate through elements in current_mask
        cm = current_mask
        while cm:
            lsb = cm & (-cm)
            idx = lsb.bit_length() - 1
            cm ^= lsb
            
            # Count conflicts of idx within current_mask
            conflicts_in_set = conflict_mask[idx] & current_mask
            cnt = 0
            tmp = conflicts_in_set
            while tmp:
                tmp &= tmp - 1
                cnt += 1
            
            if cnt > best_conflicts:
                best_conflicts = cnt
                best_to_remove = idx
        
        if best_conflicts <= 0:
            break  # No conflicts left
        
        # Remove the element
        current_mask &= ~(1 << best_to_remove)
        removed.add(best_to_remove)
    
    # Phase 2: Try to add back removed elements
    for idx in sorted(removed, key=lambda x: conflict_count[x]):
        if time.time() > DEADLINE - 15:
            break
        if (current_mask >> idx) & 1:
            continue
        if (conflict_mask[idx] & current_mask) == 0:
            current_mask |= (1 << idx)
    
    best_mask = current_mask
    best_size = count_bits(current_mask)
    
    # Phase 3: Local search - try swaps
    def local_search(mask, deadline, max_iters=50000):
        best_m = mask
        best_sz = count_bits(mask)
        cur_m = mask
        cur_sz = best_sz
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            r = random.random()
            
            if r < 0.5:
                # Try adding a random element not in current
                not_in = ((1 << N) - 1) & ~cur_m
                if not_in == 0:
                    break
                e = random.randint(0, N - 1)
                if (cur_m >> e) & 1:
                    continue
                if (conflict_mask[e] & cur_m) == 0:
                    new_m = cur_m | (1 << e)
                    sz = count_bits(new_m)
                    if sz > best_sz:
                        best_m = new_m
                        best_sz = sz
                    cur_m = new_m
                    cur_sz = sz
            elif r < 0.8:
                # Try swap: remove one, add one
                # Pick random element to remove
                to_remove = random.randint(0, N - 1)
                if not ((cur_m >> to_remove) & 1):
                    continue
                trial = cur_m & ~(1 << to_remove)
                # Try to add a random element
                e = random.randint(0, N - 1)
                if (trial >> e) & 1:
                    continue
                if (conflict_mask[e] & trial) == 0:
                    new_m = trial | (1 << e)
                    sz = count_bits(new_m)
                    if sz > best_sz:
                        best_m = new_m
                        best_sz = sz
                    cur_m = new_m
                    cur_sz = sz
            else:
                # Try 2-swap: remove two, add two
                to_remove1 = random.randint(0, N - 1)
                to_remove2 = random.randint(0, N - 1)
                if to_remove1 == to_remove2:
                    continue
                if not ((cur_m >> to_remove1) & 1) or not ((cur_m >> to_remove2) & 1):
                    continue
                trial = cur_m & ~(1 << to_remove1) & ~(1 << to_remove2)
                added = 0
                for _ in range(20):
                    if added >= 2:
                        break
                    e = random.randint(0, N - 1)
                    if (trial >> e) & 1:
                        continue
                    if (conflict_mask[e] & trial) == 0:
                        trial |= (1 << e)
                        added += 1
                sz = count_bits(trial)
                if sz > best_sz:
                    best_m = trial
                    best_sz = sz
                cur_m = trial
                cur_sz = sz
        
        return best_m
    
    if time.time() < DEADLINE - 10:
        improved = local_search(best_mask, DEADLINE - 5, max_iters=100000)
        sz = count_bits(improved)
        if sz > best_size:
            best_mask = improved
            best_size = sz
    
    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
# EVOLVE-BLOCK-END
