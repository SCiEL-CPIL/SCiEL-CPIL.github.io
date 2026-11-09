# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin
    N = 512
    MAX_SIZE = 48
    
    # Helper: Check validity and count violations using bitset DP
    # Returns (is_valid, violation_count)
    def check_set(arr):
        m = len(arr)
        if m < 3:
            return True, 0
        
        # dp[k] is a bitset where bit s is 1 if sum s is achievable by k elements
        # We need to check for each element a, if there exists a subset T of others
        # such that sum(T) = a * len(T) for len(T) >= 2.
        
        # Optimization: Pre-compute DP for the whole set?
        # No, because "others" depends on which element is the mean.
        # But we can compute DP for the whole set, and then for each element a,
        # we need to know if sum a*k is achievable by k elements NOT including a.
        
        # Let's compute dp_all[k] for the whole set.
        # Then for element a, the sums achievable by others of size k are:
        # S_others(k) = S_all(k) - {sums that include a}
        # Sums including a of size k are: {a + s | s in S_others(k-1)}
        # This recursive removal is tricky with bitsets directly.
        
        # Alternative: Compute DP for each element's exclusion? Too slow for O(m) calls.
        
        # Let's stick to the efficient method used in the seed but optimized.
        # For each a in arr:
        #   Build DP for arr \ {a}
        #   Check if a*k is in DP[k] for k=2..m-1
        
        # This is O(m^2 * bit_ops). For m=48, m^2=2304. Bit ops are fast.
        # 2304 * 48 iterations per check? No, the inner loop is over k.
        # Total work per check: sum_{a in arr} (cost to build DP for m-1 items + check k)
        # Cost to build DP for m-1 items: (m-1)^2 bit shifts/ors.
        # Total: m * (m-1)^2 * C_bit.
        # 48 * 47^2 * C ~ 100k * C.
        # C is cost of big integer shift/or. Python big ints are fast for ~20k bits (max sum 48*512=24576).
        
        violations = 0
        max_sum = m * N
        
        # We can optimize by computing the "full" DP and then using inclusion-exclusion?
        # No, let's just do the per-element DP but reuse code.
        
        # Actually, a faster check:
        # An element a is a mean of a subset T of others iff sum(T) = a * |T|.
        # This is equivalent to: 0 is the sum of (T \cup {-a})? No.
        # It means a is in the convex hull of the others in 1D? No, just arithmetic mean.
        
        # Let's use the bitset approach.
        # To speed up, we can pre-compute the DP for the whole set, and then
        # for each a, we can try to "remove" a from the DP?
        # dp[k] = dp_without_a[k] | (dp_without_a[k-1] << a)
        # So dp_without_a[k] = dp[k] & ~(dp_without_a[k-1] << a) ?? No, this doesn't work simply due to overlaps.
        
        # So we must rebuild DP for each exclusion.
        # To make it faster, we can sort arr and maybe use symmetry? No.
        
        # Let's implement the check efficiently.
        arr_sorted = sorted(arr)
        for i, a in enumerate(arr_sorted):
            # Build DP for all elements except arr_sorted[i]
            # We can do this by iterating all elements and skipping i
            # But doing this from scratch for each i is O(m^3) bit ops.
            # m=48 -> 48^3 = 110k bit ops per check.
            # 110k bit ops is very fast in Python if using ints.
            
            d = [0] * m
            d[0] = 1
            # Iterate over all elements, skip i
            for j, v in enumerate(arr_sorted):
                if j == i:
                    continue
                for k in range(m - 2, -1, -1): # Max size of subset is m-1
                    if d[k]:
                        d[k+1] |= d[k] << v
            
            # Check for violations
            for k in range(2, m):
                target = a * k
                if target > max_sum:
                    break
                if (d[k] >> target) & 1:
                    violations += 1
                    break # Count each element as 1 violation max? 
                    # Actually, if it's a mean of multiple subsets, it's still 1 "bad" element.
                    # For SA, we want to minimize the number of bad elements.
        
        return violations == 0, violations

    # Initial state: Random set of size 48
    pool = list(range(1, N + 1))
    current_set = set(random.sample(pool, MAX_SIZE))
    current_list = list(current_set)
    
    is_valid, curr_violations = check_set(current_list)
    
    if is_valid:
        return current_list

    best_set = current_list[:]
    best_violations = curr_violations
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        # 1. Swap: Pick x in set, y not in set
        # 2. Add/Remove: If we allow size changes, but we want max 48.
        #    Let's stick to fixed size 48 swaps to keep it simple and valid.
        
        if rng.random() < 0.8:
            # Swap
            x = rng.choice(list(current_set))
            # Pick y from pool not in set
            # Efficient pick: sample from pool, if in set, resample
            y = rng.choice(pool)
            while y in current_set:
                y = rng.choice(pool)
            
            new_set = current_set.copy()
            new_set.remove(x)
            new_set.add(y)
            new_list = list(new_set)
            
            _, new_violations = check_set(new_list)
            
            delta = new_violations - curr_violations
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_set = new_set
                curr_violations = new_violations
                
                if curr_violations < best_violations:
                    best_violations = curr_violations
                    best_set = new_list[:]
                    
                if curr_violations == 0:
                    return new_list
        else:
            # Perturbation: Randomly replace 2 elements
            x1, x2 = rng.sample(list(current_set), 2)
            y1, y2 = rng.sample(pool, 2)
            # Ensure y1, y2 not in set and distinct
            attempts = 0
            while (y1 in current_set or y2 in current_set or y1 == y2) and attempts < 10:
                y1, y2 = rng.sample(pool, 2)
                attempts += 1
            if y1 in current_set or y2 in current_set or y1 == y2:
                continue
                
            new_set = current_set.copy()
            new_set.remove(x1)
            new_set.remove(x2)
            new_set.add(y1)
            new_set.add(y2)
            new_list = list(new_set)
            
            _, new_violations = check_set(new_list)
            
            delta = new_violations - curr_violations
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_set = new_set
                curr_violations = new_violations
                
                if curr_violations < best_violations:
                    best_violations = curr_violations
                    best_set = new_list[:]
                    
                if curr_violations == 0:
                    return new_list

        T *= cooling_rate
        
    # Final polish: Local search on best_set
    if best_violations > 0:
        # Try to fix violations by swapping out bad elements
        # Identify bad elements in best_set
        # Re-run check to get which elements are bad?
        # My check_set only returns count. Let's modify it to return bad elements?
        # For simplicity, let's do a few more random swaps around best_set.
        temp_set = set(best_set)
        temp_violations = best_violations
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            if temp_violations == 0:
                break
            x = rng.choice(list(temp_set))
            y = rng.choice(pool)
            while y in temp_set:
                y = rng.choice(pool)
            
            new_set = temp_set.copy()
            new_set.remove(x)
            new_set.add(y)
            new_list = list(new_set)
            
            _, new_violations = check_set(new_list)
            
            if new_violations < temp_violations:
                temp_set = new_set
                temp_violations = new_violations
                if temp_violations < best_violations:
                    best_violations = temp_violations
                    best_set = new_list[:]
                if temp_violations == 0:
                    break

    return best_set
# EVOLVE-BLOCK-END
