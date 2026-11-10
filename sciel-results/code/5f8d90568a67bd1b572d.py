import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def is_valid(set_vals):
        """Check if subset sums of different cardinalities are all distinct."""
        n = len(set_vals)
        # For each cardinality k, compute all possible sums
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for i, v in enumerate(set_vals):
            # Update in reverse to avoid using same element twice
            for k in range(i + 1, 0, -1):
                for s in sums_by_card[k - 1]:
                    sums_by_card[k].add(s + v)
        # Check that sums for different cardinalities are disjoint
        for k in range(1, n + 1):
            for j in range(k):
                if sums_by_card[k] & sums_by_card[j]:
                    return False
        return True
    
    def is_valid_fast(set_vals):
        """Faster check using the min/max separation sufficient condition."""
        n = len(set_vals)
        s = sorted(set_vals)
        # Check: for each k >= 1, min k-sum > max (k-1)-sum
        for k in range(1, n + 1):
            min_k = sum(s[:k])
            max_k_minus_1 = sum(s[n - (k - 1):]) if k - 1 > 0 else 0
            if min_k <= max_k_minus_1:
                return False
        return True
    
    # Strategy: try to find the largest set by searching over structured families
    # First, try arithmetic progressions {a, a+d, a+2d, ..., a+(n-1)d}
    best_set = list(range(1, 9))  # fallback: {1,2,...,8}
    
    # Try to maximize n by trying different starting points and gaps
    # For an AP {a, a+d, ..., a+(n-1)d} with max element <= 256:
    # a + (n-1)*d <= 256, a >= 1, d >= 1
    
    # The separation condition for AP: 
    # min k-sum = k*a + d*k*(k-1)/2
    # max (k-1)-sum = (k-1)*(a + (n-1)*d) - d*(k-1)*(k-2)/2
    # Need: k*a + d*k*(k-1)/2 > (k-1)*(a + (n-1)*d) - d*(k-1)*(k-2)/2
    
    # Let's just try many APs and check with the fast condition, then verify
    best_n = 0
    best_ap = None
    
    for n in range(60, 0, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for d in range(1, 10):
            if found:
                break
            # a + (n-1)*d <= 256 => a <= 256 - (n-1)*d
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            for a in range(1, max_a + 1):
                if time.time() > deadline - 5:
                    break
                s = [a + i * d for i in range(n)]
                if is_valid_fast(s):
                    # Verify with full check
                    if is_valid(s):
                        best_n = n
                        best_ap = s
                        found = True
                        break
        if found:
            break
    
    if best_ap is not None:
        return best_ap
    
    # If no AP found, try greedy construction
    # Start with small numbers and try to add larger ones
    # Actually, let's try a different structured approach:
    # Use numbers that grow geometrically but fit in 1..256
    
    # Try: {2^0, 2^1, ..., 2^(n-1)} scaled - but these go up to 2^7=128, 2^8=256
    # {1, 2, 4, 8, 16, 32, 64, 128} - 8 elements, all subset sums unique
    # Can we do better?
    
    # Try mixed: some small, then larger gaps
    # Let's try to build greedily: start with smallest possible, add next element
    # that maintains the separation property
    
    def greedy_build():
        s = []
        for candidate in range(1, 257):
            if time.time() > deadline - 5:
                break
            trial = s + [candidate]
            if is_valid_fast(trial):
                s = trial
        return s
    
    greedy_result = greedy_build()
    if len(greedy_result) > len(best_set):
        best_set = greedy_result
    
    # Also try: start with a base and add elements that are large enough
    # The condition min_k > max_{k-1} means each new element a_n must satisfy:
    # For the new largest k = n: sum of n smallest > sum of (n-1) largest
    # This is automatically checked by is_valid_fast
    
    # Try another strategy: use the fact that we want large elements to dominate
    # If we pick elements where each is > sum of all previous, sums are fully unique
    # Max such set in 1..256: {1,2,4,8,16,32,64,128} = 8 elements
    # But we can be less strict: just need different-cardinality separation
    
    # Let me try: {a, 2a, 3a, ..., na} won't work well
    # Try: {1, 3, 5, 7, ...} odd numbers?
    
    # Let's try a more systematic search for small n first, then expand
    # Actually, the AP search should have found good solutions. Let me also try
    # non-AP sets by perturbation
    
    # Try: take best AP and try to replace elements to fit more
    if best_ap is not None and len(best_ap) < 60:
        # Try adding one more element by reducing the gap
        pass
    
    return best_set

# Pre-compute a good solution to avoid runtime search
# Let's analyze: for AP {a, a+d, ..., a+(n-1)d}
# Condition for k=1: a > 0 (always true)
# Condition for k=2: 2a+d > (a+(n-1)d) => a > (n-2)d
# Condition for k=3: 3a+3d > 2(a+(n-1)d) - d = 2a + 2(n-1)d - d => a + 3d > (2n-3)d => a > (2n-6)d
# In general, the tightest constraint is for k=n: sum of all > 0, and k=n-1 vs k=n-2, etc.
# Actually the tightest is often k=2 or small k.
# For k=2: a > (n-2)d. So n < a/d + 2.
# Max a = 256 - (n-1)d, so 256 - (n-1)d > (n-2)d => 256 > (2n-3)d => n < (256/d + 3)/2
# For d=1: n < 129.5, so n up to 129? But a > n-2, and a >= 1, so n-2 < a <= 256-n+1
# a > n-2 and a <= 256-n+1 => n-2 < 256-n+1 => 2n < 259 => n <= 129
# But we also need all other k conditions. Let me just test n=10, 15, 20 with d=1.

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    def is_valid_fast(set_vals):
        n = len(set_vals)
        s = sorted(set_vals)
        for k in range(1, n + 1):
            min_k = sum(s[:k])
            if k - 1 > 0:
                max_k_minus_1 = sum(s[n - (k - 1):])
            else:
                max_k_minus_1 = 0
            if min_k <= max_k_minus_1:
                return False
        return True
    
    def is_valid_full(set_vals):
        n = len(set_vals)
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for i, v in enumerate(set_vals):
            for k in range(i + 1, 0, -1):
                for s in sums_by_card[k - 1]:
                    sums_by_card[k].add(s + v)
        for k in range(1, n + 1):
            for j in range(k):
                if sums_by_card[k] & sums_by_card[j]:
                    return False
        return True
    
    # Try APs with d=1, increasing n
    best_set = None
    for n in range(20, 0, -1):
        if time.time() > deadline - 10:
            break
        max_a = 256 - (n - 1)
        if max_a < 1:
            continue
        # Need a > (n-2) for k=2 condition (approximately)
        min_a = max(1, n - 1)
        for a in range(max_a, min_a - 1, -1):
            if time.time() > deadline - 5:
                break
            s = list(range(a, a + n))
            if is_valid_fast(s):
                if is_valid_full(s):
                    best_set = s
                    break
        if best_set is not None:
            break
    
    if best_set is None:
        # Fallback: try greedy
        s = []
        for candidate in range(1, 257):
            if time.time() > deadline - 5:
                break
            trial = s + [candidate]
            if is_valid_fast(trial):
                s = trial
        best_set = s if s else [1]
    
    return best_set
