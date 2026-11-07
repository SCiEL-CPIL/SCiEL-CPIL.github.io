import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 256
    
    # Sieve for primes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Key insight: Any prime p > 128 can only be in the set if it's alone with other
    # numbers not divisible by p. Since p > 128, no multiple of p exists in 1..256
    # except p itself. So all primes > 128 are "free" - they conflict with nothing
    # except themselves (which is excluded by distinctness).
    
    # Primes > 128: always include them (they have no conflicts with any other number in 1..256)
    free_primes = [p for p in range(129, n + 1) if is_prime[p]]
    
    # Also 1 is free (gcd(1, x) = 1 for all x)
    # Numbers in 2..128 need careful selection
    
    # Build conflict graph for numbers 2..128 (non-prime and prime <= 128)
    # A number x conflicts with y if gcd(x, y) > 1
    # We want max weight independent set in this graph
    # Weight = the number itself (since we're maximizing sum)
    
    # Strategy: Use a greedy + local search approach with bitmask representation
    # Numbers 2..128: 127 numbers
    # We'll use a recursive branch-and-bound with good heuristics
    
    # Actually, let's use a different approach:
    # The conflict graph has special structure. Let's use a greedy algorithm
    # with multiple restarts and local improvement.
    
    # Candidate pool: 1, all primes > 128, and a subset of 2..128
    # Let's work with the full range 1..256 but handle the structure
    
    # Precompute: for each number 1..256, which other numbers conflict with it
    # conflict_mask[x] = bitmask of all y in 1..256 where gcd(x,y) > 1 and y != x
    
    # Use a list of bitmasks (Python integers as arbitrary precision)
    all_numbers = list(range(1, n + 1))
    num_vals = n + 1  # 1..256
    
    # For efficiency, map value -> index (0..255 for values 1..256)
    val_to_idx = {v: v - 1 for v in range(1, n + 1)}
    
    # Build conflict bitmasks
    conflict_mask = [0] * num_vals
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                idx_i = val_to_idx[i]
                idx_j = val_to_idx[j]
                conflict_mask[idx_i] |= (1 << idx_j)
                conflict_mask[idx_j] |= (1 << idx_i)
    
    # Greedy initial solution: sort by value descending, add if no conflict
    def greedy_initial(order=None):
        if order is None:
            order = list(range(1, n + 1))
        selected_mask = 0
        selected_set = set()
        for v in order:
            idx = val_to_idx[v]
            # Check if v conflicts with any already selected
            # selected_mask has bits set for selected values
            # Check if conflict_mask[idx] intersects with selected_mask
            if conflict_mask[idx] & selected_mask == 0:
                selected_mask |= (1 << idx)
                selected_set.add(v)
        return selected_mask, selected_set
    
    # Try different orderings
    best_mask = 0
    best_sum = 0
    
    # Ordering 1: descending by value
    order1 = list(range(n, 0, -1))
    mask1, set1 = greedy_initial(order1)
    s1 = sum(set1)
    if s1 > best_sum:
        best_sum = s1
        best_mask = mask1
    
    # Ordering 2: ascending by number of conflicts (least constrained first)
    conflict_counts = [bin(conflict_mask[i]).count('1') for i in range(num_vals)]
    order2 = sorted(range(1, n + 1), key=lambda v: (conflict_counts[val_to_idx[v]], -v))
    mask2, set2 = greedy_initial(order2)
    s2 = sum(set2)
    if s2 > best_sum:
        best_sum = s2
        best_mask = mask2
    
    # Ordering 3: random-ish (use deterministic seed)
    import random
    rng = random.Random(42)
    for trial in range(5):
        order3 = list(range(1, n + 1))
        rng.shuffle(order3)
        mask3, set3 = greedy_initial(order3)
        s3 = sum(set3)
        if s3 > best_sum:
            best_sum = s3
            best_mask = mask3
    
    # Local search: try 1-for-1, 1-for-2, 2-for-1, 2-for-2 swaps
    def get_selected_values(mask):
        vals = []
        for i in range(num_vals):
            if (mask >> i) & 1:
                vals.append(i + 1)
        return vals
    
    def try_local_search(mask, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        current_mask = mask
        current_sum = sum(get_selected_values(current_mask))
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Build selected set and unselected set
            selected = set(get_selected_values(current_mask))
            unselected = set(range(1, n + 1)) - selected
            
            # 1-for-1 swap: remove j (selected), add i (unselected)
            best_gain = 0
            best_move = None
            for j in selected:
                idx_j = val_to_idx[j]
                for i in unselected:
                    if i <= j:
                        continue
                    idx_i = val_to_idx[i]
                    # Check: i must not conflict with any selected k != j
                    # i conflicts with selected if conflict_mask[idx_i] & (current_mask without j) != 0
                    temp_mask = current_mask & ~(1 << idx_j)
                    if conflict_mask[idx_i] & temp_mask == 0:
                        gain = i - j
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('1for1', i, j)
            
            if best_move:
                typ, i, j = best_move
                current_mask &= ~(1 << val_to_idx[j])
                current_mask |= (1 << val_to_idx[i])
                current_sum += best_gain
                improved = True
                continue
            
            # 1-for-2: remove j, add i1, i2
            best_gain_12 = 0
            best_move_12 = None
            # Limit search to top unselected by value
            top_unselected = sorted(unselected, reverse=True)[:30]
            for j in selected:
                idx_j = val_to_idx[j]
                temp_mask = current_mask & ~(1 << idx_j)
                # Find unselected i that don't conflict with temp_mask
                compatible_unselected = []
                for i in top_unselected:
                    idx_i = val_to_idx[i]
                    if conflict_mask[idx_i] & temp_mask == 0:
                        compatible_unselected.append(i)
                
                for a in range(len(compatible_unselected)):
                    i1 = compatible_unselected[a]
                    for b in range(a + 1, len(compatible_unselected)):
                        i2 = compatible_unselected[b]
                        # Check i1, i2 coprime
                        if math.gcd(i1, i2) > 1:
                            continue
                        gain = i1 + i2 - j
                        if gain > best_gain_12:
                            best_gain_12 = gain
                            best_move_12 = ('1for2', i1, i2, j)
            
            if best_move_12:
                typ, i1, i2, j = best_move_12
                current_mask &= ~(1 << val_to_idx[j])
                current_mask |= (1 << val_to_idx[i1])
                current_mask |= (1 << val_to_idx[i2])
                current_sum += best_gain_12
                improved = True
                continue
            
            # 2-for-1: remove j1, j2, add i
            best_gain_21 = 0
            best_move_21 = None
            top_unselected_1 = sorted(unselected, reverse=True)[:20]
            for j1 in selected:
                for j2 in selected:
                    if j2 <= j1:
                        continue
                    idx_j1 = val_to_idx[j1]
                    idx_j2 = val_to_idx[j2]
                    temp_mask = current_mask & ~(1 << idx_j1) & ~(1 << idx_j2)
                    for i in top_unselected_1:
                        idx_i = val_to_idx[i]
                        if conflict_mask[idx_i] & temp_mask == 0:
                            gain = i - j1 - j2
                            if gain > best_gain_21:
                                best_gain_21 = gain
                                best_move_21 = ('2for1', i, j1, j2)
            
            if best_move_21:
                typ, i, j1, j2 = best_move_21
                current_mask &= ~(1 << val_to_idx[j1])
                current_mask &= ~(1 << val_to_idx[j2])
                current_mask |= (1 << val_to_idx[i])
                current_sum += best_gain_21
                improved = True
                continue
            
            # 2-for-2: remove j1, j2, add i1, i2
            best_gain_22 = 0
            best_move_22 = None
            top_unselected_2 = sorted(unselected, reverse=True)[:15]
            for j1 in selected:
                for j2 in selected:
                    if j2 <= j1:
                        continue
                    idx_j1 = val_to_idx[j1]
                    idx_j2 = val_to_idx[j2]
                    temp_mask = current_mask & ~(1 << idx_j1) & ~(1 << idx_j2)
                    compatible_unselected_2 = []
                    for i in top_unselected_2:
                        idx_i = val_to_idx[i]
                        if conflict_mask[idx_i] & temp_mask == 0:
                            compatible_unselected_2.append(i)
                    
                    for a in range(len(compatible_unselected_2)):
                        i1 = compatible_unselected_2[a]
                        for b in range(a + 1, len(compatible_unselected_2)):
                            i2 = compatible_unselected_2[b]
                            if math.gcd(i1, i2) > 1:
                                continue
                            gain = i1 + i2 - j1 - j2
                            if gain > best_gain_22:
                                best_gain_22 = gain
                                best_move_22 = ('2for2', i1, i2, j1, j2)
            
            if best_move_22:
                typ, i1, i2, j1, j2 = best_move_22
                current_mask &= ~(1 << val_to_idx[j1])
                current_mask &= ~(1 << val_to_idx[j2])
                current_mask |= (1 << val_to_idx[i1])
                current_mask |= (1 << val_to_idx[i2])
                current_sum += best_gain_22
                improved = True
                continue
            
            # No improvement, break
            break
        
        return current_mask, current_sum
    
    # Run local search from best initial solution
    if time.time() < deadline - 10.0:
        improved_mask, improved_sum = try_local_search(best_mask)
        if improved_sum > best_sum:
            best_sum = improved_sum
            best_mask = improved_mask
    
    # Additional local search passes
    for _ in range(3):
        if time.time() >= deadline - 5.0:
            break
        improved_mask, improved_sum = try_local_search(best_mask)
        if improved_sum > best_sum:
            best_sum = improved_sum
            best_mask = improved_mask
        else:
            break
    
    # Extract result
    result = []
    for i in range(num_vals):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Final validation
    result = [int(x) for x in result]
    if len(set(result)) != len(result):
        # Fallback: just use 1 and all primes > 128
        result = [1] + free_primes
    
    # Verify pairwise gcd
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        result = [1] + free_primes
    
    return result

if __name__ == "__main__":
    res = construct()
    print(sum(res))
