import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # We need sigma(a) == sigma(b) and minimize |a/b - 3/2|
    # The error is |2a - 3b| / (2b)
    # So we want 2a ≈ 3b, i.e., a ≈ 1.5*b
    # And sigma(a) == sigma(b)

    # Strategy: Precompute sigma for all numbers up to N using a sieve.
    # Then for each b, check a candidates near 1.5*b.
    # But N=10^7 means we need sigma up to 10^7.
    
    # Actually, let's think about the constraint more carefully.
    # sigma(a) = sigma(b). We want |2a - 3b| minimized relative to 2b.
    # So |2a - 3b| should be small. Let 2a - 3b = k for small integer k.
    # Then a = (3b + k) / 2, so (3b + k) must be even.
    # 3b + k even means b and k have same parity (since 3b has same parity as b).
    
    # For a given b and small k, a = (3b+k)//2 if (3b+k) is even.
    # We need a <= 10^7 and a >= 1.
    # Also sigma(a) == sigma(b).
    
    # Let's precompute sigma up to 10^7.
    N = 10000000
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now search for best (a, b)
    # For each b from 1 to N, check small k values.
    # k can be negative too.
    # a = (3b + k) // 2, need 3b + k >= 2 (so a >= 1) and a <= N.
    # Also 3b + k must be even.
    
    best_error_num = None  # |2a - 3b|
    best_error_den = None  # 2b
    best_a = None
    best_b = None
    
    # We'll iterate over b and try k in a range.
    # The error is |k| / (2b). To minimize, we want small |k| and large b.
    # But sigma(a) == sigma(b) is the hard constraint.
    
    # Let's try k = 0, ±1, ±2, ... up to some limit.
    # For each b, a = (3b + k) // 2 must be integer and in [1, N].
    
    # To be efficient, let's iterate over b in descending order (larger b gives smaller error for same k).
    # But we need to check many b's. Let's see how long the sigma computation took.
    
    # Actually, the sigma sieve above is O(N log N) which for N=10^7 is about 10^7 * ln(10^7) ≈ 1.6 * 10^8 operations.
    # In Python this might be slow. Let me check timing.
    
    if time.time() > deadline:
        # Fallback: return a known valid pair
        # sigma(1) = 1, sigma(2) = 3, sigma(3) = 4, sigma(4) = 7, sigma(5) = 6, sigma(6) = 12
        # sigma(11) = 12, sigma(6) = 12. So a=11, b=6 works. |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3
        return [11, 6]
    
    # Now search. Let's try k values from 0 outward.
    # For each k, find b such that a = (3b+k)//2 is valid and sigma(a) == sigma(b).
    
    # Let's try a range of k values. Start with small |k|.
    # For each k, iterate over valid b values.
    
    # Actually, iterating over all b for each k is too slow. Let's think differently.
    
    # For each b, the best k is the one that makes a = (3b+k)/2 closest to being an integer with matching sigma.
    # The closest k to make 3b+k even: if 3b is even (b even), k=0 works. If 3b is odd (b odd), k=±1 works.
    # So for b even, try k=0 first. For b odd, try k=1 or k=-1 first.
    
    # Let's iterate b from N down to 1, and for each b try a few k values.
    
    max_k_try = 20  # try k in [-max_k_try, max_k_try]
    
    for b in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
        sb = sigma[b]
        # Try k values in order of |k|
        # Determine parity: 3b + k must be even => k ≡ 3b (mod 2) => k ≡ b (mod 2) since 3≡1 mod 2
        # So k must have same parity as b.
        for k in range(0, max_k_try + 1):
            for sign in ([1] if k == 0 else [1, -1]):
                kk = sign * k
                if (b + kk) % 2 != 0:
                    continue
                a = (3 * b + kk) // 2
                if a < 1 or a > N:
                    continue
                if sigma[a] == sb:
                    # Found a match!
                    err_num = abs(kk)
                    err_den = 2 * b
                    if best_error_num is None or (err_num * best_error_den < best_error_num * err_den):
                        best_error_num = err_num
                        best_error_den = err_den
                        best_a = a
                        best_b = b
                    break  # No need to try larger k for this b
            if k > 0:
                break  # Already checked both signs for this |k|
    
    if best_a is not None:
        return [int(best_a), int(best_b)]
    
    # Fallback
    return [11, 6]
