"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a linear sieve for speed."""
    sigma = [0] * (n + 1)
    sigma[1] = 1
    primes = []
    is_composite = [False] * (n + 1)
    # sigma for prime powers: if p^k || n, contribution is (p^(k+1)-1)/(p-1)
    # We'll use a different approach: for each prime p, for each multiple m of p,
    # add the appropriate factor.
    # Actually, let's use the standard sieve approach but optimized.
    # sigma(n) = product over p^k || n of (1 + p + ... + p^k)
    
    # Better: use the fact that sigma is multiplicative.
    # Linear sieve: sigma(n) for prime n is n+1.
    # For composite: if n = p^k * m where gcd(m,p)=1, sigma(n) = sigma(m) * (1+p+...+p^k)
    
    # Let's use a modified sieve:
    # For each d from 1 to n, add d to all multiples of d.
    # This is O(n log n) which for n=10M is about 10M * 16.1 ≈ 161M operations.
    # In Python this might be too slow. Let's try numpy.
    
    import numpy as np
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # safety margin before 60s kill
    
    N = 10_000_000
    target_frac = Fraction(3, 2)
    
    # Compute sigma for all numbers up to N
    sigma = _sigma_sieve(N)
    
    # Group numbers by their sigma value
    groups = defaultdict(set)
    for i in range(1, N + 1):
        groups[int(sigma[i])].add(i)
    
    # Filter to groups with at least 2 elements
    # Keep as sets for O(1) lookup
    filtered_groups = {}
    for s, arr in groups.items():
        if len(arr) >= 2:
            filtered_groups[s] = arr
    
    # Search for best pair (a, b) with sigma(a) == sigma(b) and |a/b - 3/2| minimized
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group (set of numbers with same sigma), for each b in the set,
    # check if a = (3b + k)/2 is in the set for small k.
    # We want to minimize |k| / (2b), so small |k| and large b.
    
    best_error = Fraction(1, 1)
    best_pair = None
    
    # Strategy: for each group, iterate b in descending order (larger b = smaller error)
    # For each b, check k = 1, -1, 3, -3, 5, -5, ... (odd k only since 2a-3b must be odd
    # when a,b are integers... wait, 2a is even, 3b is odd if b is odd, even if b is even.
    # 2a - 3b: if b is even, 3b is even, so 2a-3b is even, k is even.
    # If b is odd, 3b is odd, so 2a-3b is odd, k is odd.
    # So k can be any integer. But a = (3b+k)/2 must be integer, so 3b+k must be even.
    # If b is even: 3b is even, so k must be even.
    # If b is odd: 3b is odd, so k must be odd.
    
    # For efficiency, for each b, check a few small |k| values.
    # The error is |k|/(2b). To get a good result, we want small |k| and large b.
    
    # Let's iterate groups by size (larger groups more likely to have close pairs)
    # But first, let's just do a straightforward search with time limit.
    
    import numpy as np
    
    # Sort groups by size descending, process larger groups first (more likely to have good pairs)
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for s, group_set in sorted_groups:
        if time.time() > deadline - 5.0:
            break
        
        # Convert to sorted list for potential binary search, but we'll use set for O(1) lookup
        # For each b in the group (iterate in some order), check nearby a values
        
        # To be efficient, iterate b values and check a = (3b + k)/2 for small k
        # We want to check k values that make a integer and a in range [1, N]
        
        # Let's sample: check all b values but with limited k checks
        # For each b, check k in [-5, -3, -1, 1, 3, 5] (or appropriate parity)
        
        # Actually, let's be smarter. For each b, the best a is near 1.5*b.
        # Check a = floor(1.5*b) and ceil(1.5*b), and a few neighbors.
        
        # Convert set to list for iteration
        group_list = list(group_set)
        
        for b in group_list:
            if time.time() > deadline - 2.0:
                break
            
            # a should be close to 1.5 * b
            # a = (3b + k) / 2 for small k
            # Check k values with correct parity
            if b % 2 == 0:
                # b even, need k even
                k_values = [0, 2, -2, 4, -4, 6, -6, 8, -8]
            else:
                # b odd, need k odd
                k_values = [1, -1, 3, -3, 5, -5, 7, -7, 9, -9]
            
            for k in k_values:
                if k == 0:
                    continue  # invalid (error = 0)
                
                num_3b_plus_k = 3 * b + k
                if num_3b_plus_k <= 0 or num_3b_plus_k % 2 != 0:
                    continue
                
                a = num_3b_plus_k // 2
                if a < 1 or a > N or a == b:
                    continue
                
                if a in group_set:
                    # Valid pair!
                    err = Fraction(abs(k), 2 * b)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    # If we found a very good pair (error < 1/1000), we can stop early
    # Otherwise, continue searching
    
    if best_pair is None:
        # Fallback: find any valid pair
        for s, group_set in filtered_groups.items():
            if len(group_set) >= 2:
                arr = sorted(group_set)
                a, b = arr[0], arr[1]
                if a != b:
                    best_pair = (a, b)
                    break
        if best_pair is None:
            best_pair = (11, 6)
    
    a, b = best_pair
    
    # Verify
    if int(sigma[a]) != int(sigma[b]):
        best_pair = (11, 6)
        a, b = best_pair
    
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
