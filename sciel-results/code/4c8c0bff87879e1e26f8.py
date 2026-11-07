import time

def _is_valid_cardinality_separated(arr):
    """Check if arr (sorted descending) satisfies cardinality separation."""
    n = len(arr)
    if n == 0:
        return True
    # For each k from 1 to n, check: min k-sum > max (k-1)-sum
    # max (k-1)-sum = sum of first (k-1) largest = prefix_sum[k-1]
    # min k-sum = sum of last k smallest = prefix_sum[n] - prefix_sum[n-k]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    total = prefix[n]
    for k in range(1, n + 1):
        max_k_minus_1 = prefix[k-1]  # sum of k-1 largest
        min_k = total - prefix[n-k]  # sum of k smallest
        if min_k <= max_k_minus_1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Try to find the largest valid set
    # Start with n=60 and work down
    best_result = []
    
    for n in range(60, 0, -1):
        if time.time() > deadline:
            break
        
        # Try to construct a valid set of size n
        # Use greedy: assign largest possible values
        # We need a_1 > a_2 > ... > a_n, all in 1..256
        # Constraint: for each k, sum of k smallest > sum of (k-1) largest
        
        # Greedy from smallest to largest
        # a_n (smallest) should be as large as possible to help satisfy constraints
        # But we also need distinct values
        
        # Let's try: assign values greedily from smallest index (largest value) to largest index (smallest value)
        # Actually, let's think differently.
        
        # For a valid set, the constraints are:
        # For k=1: a_n > 0 (always true since a_n >= 1)
        # For k=2: a_{n-1} + a_n > a_1
        # For k=3: a_{n-2} + a_{n-1} + a_n > a_1 + a_2
        # ...
        
        # Let's try a parametric approach: assume the set is close to an arithmetic progression
        # or use a simple greedy that tries to maximize each element
        
        # Greedy construction:
        # Start with the smallest element a_n, then a_{n-1}, etc.
        # But we want large values, so let's try from the top down.
        
        # Alternative: use the fact that for maximum size, we want values that are "well-separated"
        # in a specific way. Let's try a simple greedy:
        # a_1 = 256, then for each subsequent a_i, pick the largest value < a_{i-1} 
        # such that all constraints up to current point are satisfiable.
        
        # This is complex. Let's try a simpler approach:
        # For a given n, try to find ANY valid set by backtracking with pruning.
        
        # Given time constraints, let's try a heuristic:
        # Use values that grow exponentially or follow a specific pattern.
        
        # Actually, let's just try the original greedy but in reverse order of checking.
        # The original approach works but might not be optimal.
        
        # Let me try a different greedy: build from smallest to largest.
        # a_n = 1 (smallest possible)
        # a_{n-1} = 2
        # ... but this won't give large values.
        
        # Better: try to find the maximum n by checking if a valid set of size n exists.
        # Use a simple sufficient condition: if we use powers of 2, it works but gives small n.
        # For larger n, we need a more careful construction.
        
        # Let's use the constraint directly:
        # For the set to be valid, we need for all k:
        # sum_{i=n-k+1}^n a_i > sum_{i=1}^{k-1} a_i
        # Let S_k = sum of k smallest, T_k = sum of k largest
        # Then we need S_k > T_{k-1} for all k.
        
        # Greedy: start with a_n = 1, a_{n-1} = 2, ..., and increase values to satisfy constraints.
        
        # Actually, let's just try a simple approach: use the original greedy but try different starting points.
        
        # Given the time, let's just return the original greedy result if we haven't found better.
        # But let's try one more thing: a parametric family.
        
        # Try: a_i = c * 2^(n-i) for some c, but this exceeds 256 quickly.
        
        # Let's try: a_i = floor(256 * (n-i+1)/n) adjusted for distinctness.
        
        # This is getting complex. Let me just use the original approach which is proven to work.
        
        # Actually, let me try a cleaner reverse greedy:
        # For a target size n, try to assign values from largest to smallest.
        # a_1 = 256
        # a_2 = 255
        # ...
        # a_n = 256 - n + 1
        # Check if this satisfies the constraints. If not, adjust.
        
        arr = [256 - i for i in range(n)]  # [256, 255, ..., 256-n+1]
        if _is_valid_cardinality_separated(arr):
            best_result = arr[:]
            break
        
        # If the simple consecutive range doesn't work, try to find a valid set
        # by adjusting values. This is a search problem.
        
        # For now, if the simple range doesn't work, we'll fall back to the original greedy.
    
    # If we didn't find a valid set via the simple range, use the original greedy
    if not best_result:
        sum_to_card = {0: 0}
        result = []
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            new_sum_to_card = dict(sum_to_card)
            ok = True
            for s, k in sum_to_card.items():
                s_new = s + x
                k_new = k + 1
                if s_new in new_sum_to_card:
                    if new_sum_to_card[s_new] != k_new:
                        ok = False
                        break
                else:
                    new_sum_to_card[s_new] = k_new
            if ok:
                sum_to_card = new_sum_to_card
                result.append(x)
        best_result = result
    
    return list(best_result)
