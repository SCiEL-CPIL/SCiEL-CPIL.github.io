import time
import numpy as np
from math import isqrt

def _sieve_primes(n):
    """Sieve of Eratosthenes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _is_powerful(n, primes):
    """Check if n is powerful using precomputed primes."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt < 2:
                return False
    if x > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit

    # Generate powerful numbers up to L.
    # A powerful number n = prod p_i^{e_i} with e_i >= 2.
    # Generation: for each squarefree part s (product of distinct primes),
    # n = s * k^2 where s is squarefree and for each p|s, the exponent in n is 2*floor(e_p(k)/1)+1 >= 2... 
    # Actually simpler: n = a^2 * b^3 where b is squarefree doesn't work directly.
    #
    # Correct generation: n is powerful iff n = prod p^{e_p} with e_p >= 2.
    # We can generate by: for each "cube-free" base... 
    #
    # Simplest correct method: iterate over possible forms.
    # A powerful number can be written as n = u^2 * v where v is a product of primes
    # each appearing exactly once in v, but then in n they appear with exponent 2 (from u^2) 
    # plus possibly more. This is getting complex.
    #
    # Most practical: generate all numbers of form a^2 * b^3 <= L, then filter.
    # But this over-generates. Better: just generate and check.
    #
    # Even simpler for our purposes: generate powerful numbers by iterating
    # over "square" and "cube" components carefully.
    #
    # Actually the cleanest: a powerful number n satisfies n = m^2 * k^3 where
    # gcd(m, k) = 1 is NOT correct either.
    #
    # Let me use: n is powerful iff for every prime p dividing n, v_p(n) >= 2.
    # Generate by: pick a set of primes with exponents >= 2.
    # For small L, we can enumerate:
    #   For each "squarefree kernel" s (product of distinct primes, each to power 1 in the kernel),
    #   n = s * t^2 where for each p|s, v_p(t) >= 0, and total v_p(n) = 1 + 2*v_p(t) >= 2 ✓
    #   Wait, that gives v_p(n) = 1 + 2*v_p(t) which is odd >= 1. We need >= 2.
    #   So if p|s, v_p(n) = 1 + 2*v_p(t) >= 3 (odd). 
    #   If p ∤ s, v_p(n) = 2*v_p(t) >= 2 (even) or 0.
    #   So n = s * t^2 where s is squarefree gives all powerful numbers? 
    #   Check: n=4=2^2. s=1, t=2. 1*4=4. ✓
    #   n=8=2^3. s=2, t=1. 2*1=2. ✗! 8 != 2.
    #   So that doesn't work.
    #
    # Correct: n = a^2 * b^3 where a, b are positive integers. 
    # NOT every powerful number has this form uniquely, but every powerful number
    # CAN be written this way: given n = prod p^{e_p}, e_p >= 2.
    # Write e_p = 3*q_p + r_p where r_p in {0,1,2}.
    # If r_p = 0: p^{3q_p} = (p^{q_p})^3, contributes to b.
    # If r_p = 1: p^{3q_p+1} = p * (p^{q_p})^3. The p goes to... hmm.
    # If r_p = 2: p^{3q_p+2} = (p^{q_p+1})^2 * (p^{q_p})^... 
    #
    # Actually: n = a^2 * b^3 works if we let for each p:
    #   e_p = 2*alpha_p + 3*beta_p where alpha_p, beta_p >= 0.
    #   We need e_p >= 2. For e_p=2: alpha=1,beta=0. For e_p=3: alpha=0,beta=1.
    #   For e_p=4: alpha=2,beta=0. For e_p=5: alpha=1,beta=1. Etc.
    #   So yes, every powerful number = a^2 * b^3 for some a,b.
    #   But the representation isn't unique (e.g., 36=6^2*1^3=2^2*3^3... wait 2^2*3^3=108≠36).
    #   36 = 2^2*3^2. a^2*b^3: a=6,b=1 gives 36. a=2,b=? 4*b^3=36, b^3=9, no.
    #   So it IS unique if we require specific conditions, but for generation it's fine.
    #
    # Generation: for b from 1 to L^(1/3), for a from 1 to sqrt(L/b^3):
    #   n = a^2 * b^3
    #   Add to set.
    # Then filter with _is_powerful.
    #
    # Problem: this generates many non-powerful numbers too. E.g., a=2,b=1: n=4 (powerful).
    # a=2,b=2: n=4*8=32=2^5 (powerful). a=3,b=1: n=9 (powerful).
    # a=1,b=2: n=8 (powerful). a=2,b=3: n=4*27=108=2^2*3^3 (powerful).
    # a=6,b=1: n=36 (powerful). a=2,b=5: n=4*125=500=2^2*5^3 (powerful).
    # Hmm, actually a^2*b^3 is ALWAYS powerful? No! a=2,b=3: 108=2^2*3^3 ✓.
    # a=1,b=6: 216=2^3*3^3 ✓. a=4,b=3: 16*27=432=2^4*3^3 ✓.
    # Wait, is a^2*b^3 always powerful? a=2,b=1: 4=2^2 ✓.
    # a=1,b=1: 1 ✓. a=3,b=2: 9*8=72=2^3*3^2 ✓.
    # a=2,b=4: 4*64=256=2^8 ✓.
    # Hmm, what about a=6,b=5: 36*125=4500=2^2*3^2*5^3 ✓.
    # I think a^2*b^3 is ALWAYS powerful! Because for any prime p:
    #   v_p(a^2*b^3) = 2*v_p(a) + 3*v_p(b).
    #   This is 0, 2, 3, 4, 5, 6, ... but never 1!
    #   Because 2a+3b can be 0,2,3,4,5,6,... but not 1.
    #   So yes, a^2*b^3 is always powerful (or 1).
    #
    # BUT: not every powerful number is of the form a^2*b^3 with a,b having no shared primes
    # in a way that... actually every powerful number IS of the form a^2*b^3.
    # Proof: n=prod p^{e_p}, e_p>=2. For each p, write e_p = 2*alpha + 3*beta.
    # Since e_p >= 2, we can always find non-negative alpha, beta:
    #   e_p=2: (1,0), e_p=3: (0,1), e_p=4: (2,0), e_p=5: (1,1), e_p=6: (0,2) or (3,0), etc.
    # So n = (prod p^{alpha_p})^2 * (prod p^{beta_p})^3. ✓
    #
    # So generating all a^2*b^3 <= L gives us ALL powerful numbers <= L (plus duplicates).
    # We just need to deduplicate with a set.

    L = 2 * 10**7  # Upper bound for individual powerful numbers

    powerful_set = set()
    max_b = int(L ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            powerful_set.add(n)

    powerful_list = sorted(powerful_set)
    powerful_set_list = set(powerful_list)

    elapsed = time.time() - start_time
    print(f"Generated {len(powerful_list)} powerful numbers up to {L}, took {elapsed:.1f}s")

    if time.time() > deadline - 20:
        # Not enough time for the counting step
        # Just return a small known result
        return {'N': 2, 'pairs': [[1, 1]]}

    # Now find N maximizing count of pairs (a,b) with a<=b, a+b=N, both powerful.
    # Use numpy: create indicator array, use convolution.
    # But max_N = 2*L = 4*10^7, array of that size is 40MB for int32, manageable.
    
    max_N = 2 * L
    arr = np.zeros(max_N + 1, dtype=np.int32)
    for p in powerful_list:
        if p <= max_N:
            arr[p] = 1

    elapsed2 = time.time() - start_time
    print(f"Array setup took {elapsed2:.1f}s total")

    if time.time() > deadline - 15:
        # Try a smaller approach: just check a few candidate N values
        # Candidate N values: sums involving small powerful numbers
        best_N = 2
        best_pairs = [[1, 1]]
        for N in [2, 4, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 40, 48, 54, 60, 72, 80, 90, 100]:
            pairs = []
            for a in powerful_list:
                if a > N // 2:
                    break
                b = N - a
                if b in powerful_set_list:
                    pairs.append([a, b])
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_N = N
        return {'N': best_N, 'pairs': best_pairs}

    # Use FFT convolution
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')

    elapsed3 = time.time() - start_time
    print(f"Convolution took {elapsed3:.1f}s total")

    # Find best N
    best_N = 2
    best_count = 0
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(counts[N])
        if ordered == 0:
            continue
        if N % 2 == 0 and (N // 2) in powerful_set_list:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N

    print(f"Best N = {best_N}, count = {best_count}, total time {time.time()-start_time:.1f}s")

    # Generate pairs for best_N
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_list:
            pairs.append([a, b])

    pairs.sort()

    return {'N': best_N, 'pairs': pairs}
