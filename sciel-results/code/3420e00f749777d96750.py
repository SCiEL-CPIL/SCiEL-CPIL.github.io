import time
import math
import sys
from array import array

MAX_N = 10_000_000
TIME_LIMIT = 85

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    n = MAX_N
    
    # Linear sieve for sigma(n)
    # sigma[1] = 1
    # For prime p, sigma[p] = 1 + p
    # If i % p == 0:
    #   let i = p^k * m, gcd(m,p)=1
    #   sigma[i] = sigma[m] * (1 + p + ... + p^k)
    #   sigma[i*p] = sigma[m] * (1 + p + ... + p^(k+1))
    #   We can store the power of p dividing i, or the sum of powers.
    #   Let spf[i] be smallest prime factor.
    #   Let power_sum[i] be sum of powers of spf[i] dividing i.
    #   If spf[i*p] == spf[i]:
    #       power_sum[i*p] = power_sum[i] + p^k * p  (where p^k is highest power of spf[i] in i)
    #       sigma[i*p] = sigma[i] * (power_sum[i*p] / power_sum[i])
    #   Else:
    #       power_sum[i*p] = 1 + p
    #       sigma[i*p] = sigma[i] * (1 + p)
    
    spf = array('I', [0]) * (n + 1)
    sigma = array('Q', [0]) * (n + 1)
    power_sum = array('Q', [0]) * (n + 1) # Sum of geometric series for spf part
    p_power = array('I', [0]) * (n + 1)    # Highest power of spf dividing n
    
    sigma[1] = 1
    power_sum[1] = 1
    p_power[1] = 1
    
    primes = []
    is_composite = bytearray(n + 1)
    
    for i in range(2, n + 1):
        if not is_composite[i]:
            primes.append(i)
            spf[i] = i
            sigma[i] = 1 + i
            power_sum[i] = 1 + i
            p_power[i] = i
            
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            is_composite[i * p] = 1
            spf[i * p] = p
            
            if i % p == 0:
                # p is spf of i
                # p_power[i] is p^k
                # p_power[i*p] is p^(k+1)
                p_power[i * p] = p_power[i] * p
                power_sum[i * p] = power_sum[i] + p_power[i * p]
                # sigma[i*p] = sigma[i] * (power_sum[i*p] / power_sum[i])
                # Since power_sum[i*p] is divisible by power_sum[i]?
                # 1 + p + ... + p^(k+1) = (1 + p + ... + p^k) + p^(k+1)
                # Not necessarily divisible.
                # But sigma is integer.
                # sigma[i] = sigma[m] * power_sum[i]
                # sigma[i*p] = sigma[m] * power_sum[i*p]
                # So sigma[i*p] = sigma[i] * power_sum[i*p] // power_sum[i]
                sigma[i * p] = (sigma[i] * power_sum[i * p]) // power_sum[i]
                break
            else:
                # p does not divide i
                p_power[i * p] = p
                power_sum[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
                
    t1 = time.time()
    print(f"Sigma computation: {t1-t0:.2f}s", file=sys.stderr)
    
    if time.time() > deadline - 20:
        return [6, 11]
        
    # Now we have sigma[1..n]
    # We need to find a, b in [1..n] such that sigma[a] == sigma[b] and |a/b - 3/2| is minimized.
    # |a/b - 3/2| = |2a - 3b| / (2ab)
    # Minimize |2a - 3b| / (2ab).
    
    # Group indices by sigma value.
    # Using a dictionary might be slow for 10^7 entries if we append to lists.
    # Sorting is more robust.
    
    import numpy as np
    
    # Convert sigma to numpy array for sorting
    # Note: sigma[0] is 0, we care about 1..n
    sigma_np = np.array(sigma[1:], dtype=np.uint64)
    indices = np.arange(1, n + 1, dtype=np.int32)
    
    order = np.argsort(sigma_np, kind='stable')
    sorted_sigma = sigma_np[order]
    sorted_indices = indices[order]
    
    t2 = time.time()
    print(f"Sorting: {t2-t1:.2f}s", file=sys.stderr)
    
    if time.time() > deadline - 10:
        return [6, 11]
        
    best_err = float('inf')
    best_a = 6
    best_b = 11
    
    i = 0
    m = len(sorted_sigma)
    while i < m:
        j = i + 1
        while j < m and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        group_size = j - i
        if group_size >= 2:
            # Extract group
            # For efficiency, if group is small, just do O(G^2) or O(G log G)
            # If group is large, use two pointers.
            # Two pointers works on sorted A=2*x and B=3*x.
            # Since group is sorted, A and B are sorted.
            
            # We can process the group using two pointers in O(G)
            # But we need to be careful: we are looking for min |2a - 3b| / (2ab).
            # The two pointer finds min |2a - 3b|.
            # Does min |2a - 3b| imply min |2a - 3b| / (2ab)?
            # Not necessarily. A small difference with large a,b might have larger error than a slightly larger difference with small a,b.
            # Example: |2a-3b|=1, a=100, b=67 -> err = 1/13400
            # |2a-3b|=2, a=10, b=6 -> err = 2/120 = 1/60
            # 1/60 > 1/13400. So smaller difference is better IF a,b are similar magnitude.
            # However, a/b ~ 3/2 implies a ~ 1.5 b.
            # So 2ab ~ 3 b^2.
            # Error ~ |2a-3b| / (3 b^2).
            # If we minimize |2a-3b|, we are minimizing the numerator.
            # But the denominator varies.
            
            # Let's stick to the definition: minimize |a/b - 3/2|.
            # This is equivalent to finding a/b closest to 1.5.
            
            # For a fixed group, we want to find a, b in group such that a/b is closest to 1.5.
            # Let ratio = a/b. We want ratio ~ 1.5.
            # a ~ 1.5 b.
            
            # We can iterate through b in group, and find a in group closest to 1.5 b.
            # Since group is sorted, we can use binary search for a.
            # This is O(G log G).
            
            # The two pointer method for min |2a - 3b| finds the pair with smallest absolute difference in the linear combination.
            # Is it guaranteed to be the best ratio?
            # Counter-example:
            # Group: {10, 11, 15, 16}
            # Pairs:
            # a=10, b=11 -> 10/11 ~ 0.9 (bad)
            # a=15, b=10 -> 1.5 (perfect) -> |2*15 - 3*10| = 0.
            # a=16, b=11 -> 16/11 ~ 1.45
            # a=11, b=10 -> 1.1
            
            # If we have a pair with 2a=3b exactly, error is 0.
            # If not, we want min |2a-3b| / (2ab).
            
            # Let's just use binary search for each element. It's safer for the ratio objective.
            # G is usually small. Sigma collisions are rare for small numbers, but for large numbers?
            # Average group size for N=10^7?
            # Number of distinct sigma values is roughly N / log N ? No, sigma(n) grows like n log log n.
            # Distinct values are dense?
            # Actually, sigma(n) is not that sparse.
            # But most groups are size 1 or 2.
            # So O(G log G) is effectively O(N).
            
            group = sorted_indices[i:j]
            # For each a in group, find b closest to 2/3 a
            # Wait, a/b ~ 3/2 => b ~ 2/3 a.
            for k in range(group_size):
                a = group[k]
                target_b = (2 * a) / 3.0
                # Binary search in group
                pos = np.searchsorted(group, target_b)
                # Check pos and pos-1
                for p in [pos - 1, pos]:
                    if 0 <= p < group_size:
                        b = group[p]
                        # Check a != b? No, a and b can be same?
                        # "a, b positive integers". If a=b, sigma(a)=sigma(b) is true.
                        # |a/a - 3/2| = |1 - 1.5| = 0.5.
                        # We want to minimize this.
                        # If a=b, error is 0.5.
                        # We can do much better.
                        diff_num = abs(2 * a - 3 * b)
                        if diff_num == 0:
                            continue # Invalid
                        err = diff_num / (2 * a * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
        i = j
        
    t3 = time.time()
    print(f"Search: {t3-t2:.2f}s, best_err={best_err:.2e}", file=sys.stderr)
    
    if best_err == float('inf'):
        return [6, 11]
        
    best_a = int(best_a)
    best_b = int(best_b)
    
    # Verify
    def sigma_check(x):
        s = 0
        for d in range(1, int(math.isqrt(x)) + 1):
            if x % d == 0:
                s += d
                if d != x // d:
                    s += x // d
        return s
        
    if sigma_check(best_a) != sigma_check(best_b):
        return [6, 11]
        
    return [best_a, best_b]
