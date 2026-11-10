"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    1. Try arithmetic progressions from the top of the range (largest values first)
       since larger values create larger gaps between cardinality sum ranges.
    2. If AP doesn't work well enough, use greedy with exact cardinality-indexed
       subset-sum tracking.
    3. Use deadline guard to ensure we finish within time limits.
    
    The key insight: if we use large numbers in an arithmetic progression, the minimum
    sum of k elements will be strictly greater than the maximum sum of k-1 elements,
    which guarantees no cross-cardinality collisions.
    """
    deadline = time.time() + 50  # 50 second safety margin
    
    # First, try simple arithmetic progression from the top
    # {a, a+1, ..., a+n-1} where n is the size
    # Min sum of k elements: k*a + k*(k-1)/2
    # Max sum of k-1 elements: (k-1)*(a+n-1) - (k-1)*(k-2)/2
    # We need: k*a + k*(k-1)/2 > (k-1)*(a+n-1) - (k-1)*(k-2)/2 for all k=1..n
    
    best_result = None
    
    # Try APs starting from high values
    for start in range(256, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Find maximum n such that {start, start+1, ..., start+n-1} is valid
        # and all elements are <= 256
        max_n = 256 - start + 1
        
        for n in range(max_n, 0, -1):
            if time.time() > deadline - 5:
                break
            
            # Check if AP {start, start+1, ..., start+n-1} is valid
            # Min sum of k elements: sum of k smallest = k*start + k*(k-1)//2
            # Max sum of k-1 elements: sum of k-1 largest = (k-1)*(start+n-1) - (k-1)*(k-2)//2
            valid = True
            for k in range(1, n + 1):
                min_sum_k = k * start + k * (k - 1) // 2
                max_sum_k_minus_1 = (k - 1) * (start + n - 1) - (k - 1) * (k - 2) // 2
                if min_sum_k <= max_sum_k_minus_1:
                    valid = False
                    break
            
            if valid:
                result = list(range(start, start + n))
                if best_result is None or len(result) > len(best_result):
                    best_result = result
                break  # Found max n for this start, move to next start
    
    if best_result is not None and len(best_result) >= 30:
        return best_result
    
    # If AP approach didn't give good results, try greedy with subset-sum tracking
    # We'll try to build a set by adding elements one at a time, checking validity
    
    def is_valid_set(s):
        """Check if set s satisfies the cross-cardinality sum constraint."""
        if time.time() > deadline:
            return False
        
        n = len(s)
        if n == 0:
            return True
        
        # For each cardinality k, compute all possible sums
        # and check that sums from different cardinalities don't overlap
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        
        for k in range(1, n + 1):
            for combo in combinations(s, k):
                sums_by_card[k].add(sum(combo))
            if time.time() > deadline:
                return False
        
        # Check for cross-cardinality collisions
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_card[i] & sums_by_card[j]:
                    return False
        
        return True
    
    # Greedy: try to add elements from high to low
    candidate_set = []
    for val in range(256, 0, -1):
        if time.time() > deadline - 10:
            break
        
        test_set = candidate_set + [val]
        if is_valid_set(test_set):
            candidate_set.append(val)
        
        if len(candidate_set) >= 60:
            break
    
    if candidate_set:
        if best_result is None or len(candidate_set) > len(best_result):
            best_result = candidate_set
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed result
    return list(range(128, 136))

# EVOLVE-BLOCK-END
