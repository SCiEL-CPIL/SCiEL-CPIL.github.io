"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_set(S):
    """Check if set S satisfies: no element is the arithmetic mean of any subset of >=2 OTHER elements."""
    S_list = list(S)
    n = len(S_list)
    if n < 2:
        return True
    
    # For each element a in S, check if a is the mean of any subset of OTHER elements
    # a = sum(T)/len(T) for some subset T of S\{a}, |T| >= 2
    # => sum(T) = len(T) * a
    # => sum(T) - len(T)*a = 0
    # Let b_i = x_i - a for x_i in S\{a}
    # Then we need sum(b_i) = 0 for some subset of size >= 2
    
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        m = len(others)
        # b_i = x_i - a
        b_vals = np.array([x - a for x in others], dtype=np.int64)
        
        # We need to check if any subset of size >= 2 has sum = 0
        # Use DP: dp[k][s] = whether we can pick k elements from b_vals[0..j] with sum = s
        # But sums can be negative. Let's shift.
        # Range of possible sums: min_sum to max_sum
        # b_vals are in range [-(511), 511] roughly
        # Max subset sum magnitude: m * 511
        # This is too large for direct DP.
        
        # Alternative: use a different approach.
        # For each pair of elements in others, check if their average equals a.
        # For larger subsets, it's harder.
        
        # Actually, let me think about this differently.
        # a is the mean of subset T iff sum(T) = |T| * a
        # Let's use a bitset DP where we track achievable sums for each subset size.
        
        # b_vals range: each element is in [1-a, 512-a], so in [-511, 511]
        # Number of elements m <= 47
        # Maximum absolute sum: 47 * 511 = 24017
        # We can offset by this to make non-negative
        
        max_abs_sum = m * 511
        offset = max_abs_sum
        
        # dp[k] = bitset of achievable sums for subsets of size k
        # Use Python integers as bitsets (they can be arbitrarily large)
        # bit s in dp[k] means we can achieve sum = s - offset with k elements
        
        dp = [0] * (m + 1)
        dp[0] = 1 << offset  # empty subset has sum 0, which is at position offset
        
        for bv in b_vals:
            # Update dp in reverse order of k
            for k in range(m - 1, -1, -1):
                if dp[k] != 0:
                    # Adding bv: sum increases by bv
                    # New bit position: old_position + bv
                    # Shift left by bv if bv >= 0, right by -bv if bv < 0
                    if bv >= 0:
                        new_bits = dp[k] << bv
                    else:
                        new_bits = dp[k] >> (-bv)
                    dp[k + 1] |= new_bits
        
        # Check if any dp[k] for k >= 2 has bit at position offset (sum = 0)
        for k in range(2, m + 1):
            if dp[k] & (1 << offset):
                return False
    
    return True

def _fast_is_valid_pairwise(S_list):
    """Quick check: just verify no element is the average of any pair of other elements."""
    S_set = set(S_list)
    n = len(S_list)
    for i, a in enumerate(S_list):
        for j in range(i + 1, n):
            b = S_list[j]
            # Check if a is mean of b and c for some c in S\{a}
            # a = (b+c)/2 => c = 2a - b
            c = 2 * a - b
            if c != a and c != b and c in S_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s margin before 90s kill
    
    random.seed(42)
    
    best_set = []
    best_size = 0
    
    # Initial greedy construction
    selected = []
    for x in range(1, 513):
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
    
    best_set = list(selected)
    best_size = len(selected)
    
    # Local search / Simulated Annealing to improve
    current_set = list(selected)
    current_size = len(current_set)
    
    # Try to add elements from outside
    outside = [x for x in range(1, 513) if x not in set(current_set)]
    random.shuffle(outside)
    
    # SA parameters
    temp = 100.0
    cooling = 0.995
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        if not outside:
            # Try removing and re-adding
            break
        
        # Random move: try to add a random outside element
        idx = random.randint(0, len(outside) - 1)
        elem = outside[idx]
        
        new_set = current_set + [elem]
        if _is_valid_set(new_set):
            current_set = new_set
            outside.pop(idx)
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            temp = 100.0  # Reset on improvement
        else:
            # Sometimes accept worse solutions to escape local optima
            # For now, just skip (greedy hill climbing)
            pass
        
        # Also try swaps: remove one from current, add one from outside
        if iters % 10 == 0 and outside:
            if current_set:
                rem_idx = random.randint(0, len(current_set) - 1)
                add_idx = random.randint(0, len(outside) - 1)
                rem_elem = current_set[rem_idx]
                add_elem = outside[add_idx]
                
                new_set = current_set.copy()
                new_set[rem_idx] = add_elem
                new_set.sort()
                
                if _is_valid_set(new_set):
                    current_set = new_set
                    outside[add_idx] = rem_elem
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
                    temp = 100.0
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    # Final greedy pass to try adding more
    remaining = [x for x in range(1, 513) if x not in set(best_set)]
    for x in remaining:
        candidate = best_set + [x]
        if _is_valid_set(candidate):
            best_set = candidate
    
    return best_set
# EVOLVE-BLOCK-END
