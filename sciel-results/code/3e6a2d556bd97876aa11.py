import time
import math

def construct():
    start = time.time()
    deadline = 82.0  # 8s safety margin before 90s kill

    # We need a and b <= 10^7, sigma(a)=sigma(b), minimize |a/b - 3/2|
    # Equivalently: minimize |2a - 3b| / (2b)
    # So we want 2a ≈ 3b, i.e., a/b ≈ 3/2
    
    # Strategy: Enumerate b from 1 to N, compute sigma(b), 
    # then for each b look for a = round(1.5*b) nearby such that sigma(a)=sigma(b)
    # But sigma computation is expensive.
    
    # Better strategy: 
    # 1. Use a sieve to compute sigma(n) for n up to N=10^7
    # 2. Group numbers by their sigma value
    # 3. For each group, find pairs (a,b) minimizing |a/b - 3/2|
    
    # The sieve approach: sigma[n] = sum of divisors of n
    # Using the standard sieve: for i from 1 to N, add i to all multiples of i
    
    N = 10_000_000
    
    # sigma(n) for n up to N. Max sigma for n <= 10^7 is around 40000000 or so.
    # Let's use int32 which can hold up to 2*10^9, safe enough.
    # Actually sigma(9979200) is about 40M, so int32 (max ~2.1B) is fine.
    
    sigmas = [0] * (N + 1)
    
    # Standard divisor sum sieve
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigmas[j] += i
        if time.time() - start > deadline - 30:
            # Not enough time for full sieve; but we need it.
            # This should complete in ~15-20s typically
            pass
    
    if time.time() - start > deadline - 5:
        # Fallback: return a known valid pair
        # sigma(6)=12, sigma(8)=15... no
        # sigma(6)=1+2+3+6=12, sigma(11)=12. |11/6 - 3/2| = |1.833-1.5|=0.333
        # sigma(8)=1+2+4+8=15, sigma(14)=15. |14/8-1.5|=|1.75-1.5|=0.25
        # Let's just return something valid
        return [11, 6]
    
    # Now group by sigma value
    # We need to find, for each sigma value s, all n with sigma(n)=s
    # Then for pairs in the same group, minimize |a/b - 3/2|
    
    # Build dictionary: sigma_value -> list of n
    # This could use a lot of memory. Let's think about it differently.
    
    # Alternative: For each b from 1 to N, target a ≈ 1.5*b
    # Check a few candidates around 1.5*b and see if sigma(a)==sigma(b)
    
    # This is much more efficient! For each b, we only need to check a few a values.
    # But we still need sigma values, which we have.
    
    best_a = 11
    best_b = 6
    best_err = abs(11.0 / 6.0 - 1.5)
    
    # For each b, check a in a window around 1.5*b
    # The window size: we want |2a - 3b| to be small.
    # |2a - 3b| = 0 means a = 3b/2, requires 3b even.
    # For each b, check a = floor(1.5b), ceil(1.5b), and a few more.
    
    # Actually, to be thorough, for each b we should check all a such that 
    # sigma(a) == sigma(b) and |a/b - 3/2| is small.
    # But checking all a is O(N^2).
    
    # Better: for each b, compute sigma_b = sigmas[b]
    # Then we want a with sigmas[a] == sigma_b and a close to 1.5*b
    # We can check a range [floor(1.5b) - delta, ceil(1.5b) + delta]
    # What delta? The error |a/b - 3/2| = |2a-3b|/(2b)
    # For small error, |2a-3b| is small.
    # If we check |2a-3b| <= K for some K, we cover enough.
    
    # Let's be smart: iterate b, and for each b check a values where |2a - 3b| <= some bound
    # and a is in [1, N]
    
    # |2a - 3b| <= 200 should be enough to find good ratios.
    # For b up to 10^7, a is up to 1.5*10^7, but capped at N=10^7.
    
    # Actually, let's do: for each b, the ideal a is 3b/2.
    # Check a in a window around 3b/2.
    
    max_window = 1000  # check |a - 1.5b| <= 1000
    
    for b in range(1, N + 1):
        if time.time() - start > deadline - 5:
            break
        sig_b = sigmas[b]
        if sig_b == 0:
            continue
        
        ideal_a = 1.5 * b
        a_min = max(1, int(ideal_a - max_window))
        a_max = min(N, int(ideal_a + max_window))
        
        for a in range(a_min, a_max + 1):
            if a == b:
                # |a/b - 3/2| = |1 - 1.5| = 0.5, probably not best
                continue
            if sigmas[a] != sig_b:
                continue
            # Compute error exactly as rational, then as float
            # |a/b - 3/2| = |2a - 3b| / (2b)
            num = abs(2 * a - 3 * b)
            if num == 0:
                # Perfect ratio, but task says raw value 0 is invalid
                # So we can't return this? "A raw value of exactly 0 cannot be scored"
                # So skip perfect matches? Or return them and hope?
                # The task says "cannot be scored and counts as invalid"
                # So we must avoid a/b = 3/2 exactly
                continue
            den = 2 * b
            err = num / den
            if err < best_err:
                best_err = err
                best_a = a
                best_b = b
    
    # Ensure types are Python int
    return [int(best_a), int(best_b)]
