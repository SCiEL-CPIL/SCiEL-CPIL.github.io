import math
import time

def _squarefree_kernel(n):
    """Return integer kernel (product of primes with odd exponent) of n."""
    f = n
    result = 1
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if f > 1:
        result *= f
    return result

def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n

def construct():
    N = 210
    MAX = 10210
    MAX_SIZE = 100

    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin before 90s kill

    # Precompute squarefree kernels for all candidates
    # Use a sieve-based approach for speed
    kernels = [0] * (MAX + 1)
    for i in range(211, MAX + 1):
        kernels[i] = _squarefree_kernel(i)

    # Target: we need product of selected * 210 to be a perfect square
    # 210 = 2 * 3 * 5 * 7, so squarefree kernel of 210 is 210
    # We need: XOR (in GF(2) sense, i.e., product with odd exponents) of
    #   kernel(210) * product(selected kernels) == 1 (perfect square)
    # Equivalently: the multiset of prime factors (with odd multiplicity)
    # of 210 * product(selected) must be empty.
    
    # Better: track the "squarefree part" as a set of primes with odd exponent.
    # 210 contributes primes {2, 3, 5, 7} each with odd exponent.
    # We need to select numbers so that the combined odd-exponent primes cancel out.
    
    # For small solutions (1-2 numbers), we can just check directly.
    # 1 number: need kernel(M) == 210 (so M*210 has all even exponents)
    #   Actually: kernel(210 * M) == 1, i.e., 210*M is a perfect square.
    #   kernel(210*M) = squarefree(210*M). Since 210 = 2*3*5*7,
    #   210*M is a square iff M = 210 * k^2 for some integer k.
    #   So M = 210*k^2. Need 210 < M <= 10210, so k^2 < 10210/210 ≈ 48.6, k <= 6.
    #   k=1: M=210 (not > 210), k=2: M=840, k=3: M=1890, k=4: M=3360, k=5: M=5250, k=6: M=7560
    
    # Check single-number solutions
    for k in range(1, 7):
        M = 210 * k * k
        if 210 < M <= MAX:
            # Verify
            prod = M * N
            if _is_perfect_square(prod):
                return [M]

    # 2 numbers: need kernel(M1 * M2) == kernel(210) = 210
    # i.e., M1 * M2 * 210 is a perfect square.
    # This means: the squarefree part of M1*M2 must equal 210.
    # Equivalently: M1 * M2 = 210 * t^2 for some t.
    # Or: M1 / 210 = (t^2) / M2 ... 
    # Better: M1 * M2 must have the same squarefree kernel as 210.
    # kernel(M1 * M2) = 210.
    
    # For each pair (M1, M2) with M1 < M2, check if M1*M2*210 is a perfect square.
    # But that's O(n^2) which is ~50M pairs. Too slow.
    
    # Alternative: for each M1, we need M2 such that M1*M2*210 is a square.
    # M1*M2*210 = s^2 => M2 = s^2 / (M1*210)
    # So for each M1, we can iterate over possible s values.
    # s^2 = M1 * M2 * 210, with M2 in (210, 10210], M2 != M1
    # s^2 in range (M1*210*211, M1*210*10210]
    # s in range (sqrt(M1*210*211), sqrt(M1*210*10210)]
    
    # Let's try a different approach: precompute all kernels, then for each M1,
    # find M2 such that kernel(M1) * kernel(M2) / gcd stuff... 
    # Actually, kernel(a*b) is not simply kernel(a)*kernel(b) due to shared primes.
    
    # Let me use the direct check but be smarter.
    # For 2 numbers: M1 * M2 * 210 must be a perfect square.
    # Let g = gcd(M1, M2). Write M1 = g*a, M2 = g*b, gcd(a,b)=1.
    # Then g^2 * a * b * 210 must be a square.
    # So a * b * 210 must be a square (since g^2 is already a square).
    # gcd(a, b) = 1, so a and b must each be "squarefree-compatible" with 210.
    # Specifically, a * b * 210 = square, gcd(a,b)=1.
    # This means: for each prime p, the exponent of p in (a * b * 210) is even.
    # Since gcd(a,b)=1, p can appear in at most one of a, b.
    
    # This is getting complex. Let me just do the direct O(n^2) but with early termination
    # and only check pairs where the max is small (since we want to minimize max).
    
    # Actually, let me think about this differently.
    # We want to minimize max(M1, M2, ..., Mk).
    # So we should search in order of increasing maximum.
    # For max = T, we can only use numbers in (210, T].
    # Start with T = 211, 212, ... and for each T, check if there's a valid subset.
    
    # For T up to ~1000, the number of candidates is small (~800), so O(n^2) pairs is ~640k, fast.
    # For larger T, we need to be smarter.
    
    # Let's start with small T and increase.
    # For each T, check:
    #   - Single number: already handled above (T = 840, 1890, etc.)
    #   - Two numbers: any pair from (211..T]
    #   - Three numbers: any triple (too many for large T)
    
    # Let me just do the two-number search efficiently.
    # For each M1 in (211..T], for each M2 in (M1+1..T], check if M1*M2*210 is a square.
    # But we want to find the smallest T where such a pair exists.
    
    # Precompute: for each M, kernel(M). Then M1*M2*210 is a square iff
    # kernel(M1*M2*210) == 1.
    # kernel(M1*M2*210) can be computed from the prime factorizations.
    
    # Let me use a different strategy: for each M1, compute what M2 needs to be.
    # M1 * M2 * 210 = s^2
    # M2 = s^2 / (M1 * 210)
    # For M2 to be an integer in (210, 10210], we need M1*210 | s^2.
    # Let M1*210 = A. We need s^2 ≡ 0 (mod A), so A | s^2.
    # This means: for each prime p^e || A, we need 2e' >= e where e' is exponent in s.
    # So s must be divisible by the "square root" of A in the squarefree sense.
    
    # Let me just iterate over possible M2 values for each M1.
    # For fixed M1, M2 = s^2 / (M1 * 210) for some s.
    # s^2 = M1 * M2 * 210, so s = sqrt(M1 * M2 * 210).
    # M2 = s^2 / (M1 * 210) must be an integer in (210, 10210].
    
    # Let A = M1 * 210. We need s^2 / A to be an integer in (210, 10210].
    # So s^2 = A * M2, where M2 in (210, 10210].
    # s^2 in range (A*211, A*10210]
    # s in range (ceil(sqrt(A*211)), floor(sqrt(A*10210))]
    
    # For each s in that range, check if A * M2 = s^2, i.e., M2 = s^2 // A,
    # and verify s^2 % A == 0 and 210 < M2 <= 10210 and M2 != M1.
    
    # Let's implement this.
    
    # First, let's try to find a 2-number solution with small max.
    # We'll iterate M1 from 211 upward, and for each M1, find valid M2.
    # To minimize max(M1, M2), we want both M1 and M2 to be small.
    
    # Let's collect all valid pairs and pick the one with smallest max.
    best_max = MAX + 1
    best_pair = None
    
    # For efficiency, let's limit our search.
    # If we find a pair with max <= some bound, we can stop early if we're confident.
    # But let's just find the optimal 2-number solution first.
    
    # Actually, let me reconsider. The single-number solution M=840 gives max=840, raw=630.
    # Can we do better with 2 numbers? We need M1*M2*210 to be a square, with both < 840.
    # Let's search for pairs with max < 840.
    
    # For M1 in (211..839], M2 in (M1+1..839], check if M1*M2*210 is a square.
    # That's about 628*628/2 ≈ 197k pairs. Fast.
    
    # Let me just do a full search for 2-number solutions and pick the best.
    
    # Precompute kernels for fast checking
    # Actually, let me just compute M1*M2*210 and check if it's a perfect square.
    
    # For speed, let's use the s-based approach.
    # For each M1 from 211 to MAX:
    #   A = M1 * 210
    #   s_min = isqrt(A * 211) + 1  # first s where s^2 > A*210
    #   s_max = isqrt(A * 10210)
    #   For s in range(s_min, s_max+1):
    #       if s*s % A == 0:
    #           M2 = s*s // A
    #           if 210 < M2 <= 10210 and M2 != M1:
    #               # Valid pair (M1, M2)
    #               if max(M1, M2) < best_max:
    #                   best_max = max(M1, M2)
    #                   best_pair = (M1, M2)
    
    # But this is O(n * sqrt(n)) which might be slow for n=10000.
    # Let me estimate: for M1=211, A=44310, s range is about sqrt(44310*10210)-sqrt(44310*211) ≈ 21290-3060 ≈ 18230.
    # For M1=10210, A=2144100, s range is about sqrt(2144100*10210)-sqrt(2144100*211) ≈ 147800-21300 ≈ 126500.
    # Total iterations: sum over M1 of (s_max - s_min) ≈ 10000 * 50000 = 500M. Too slow.
    
    # Let me use a different approach. Precompute all squarefree kernels, then use a dictionary.
    # For 2 numbers: kernel(M1) * kernel(M2) must have the same squarefree part as 210.
    # Actually, let me think in terms of prime factorization.
    
    # Let me represent each number's "odd-exponent prime set" as a frozenset or a bitmask.
    # 210 = 2 * 3 * 5 * 7.
    # For M1*M2*210 to be a square, the XOR (symmetric difference) of the prime sets
    # {primes with odd exponent in M1} XOR {primes with odd exponent in M2} XOR {2,3,5,7} must be empty.
    # I.e., {odd primes in M1} XOR {odd primes in M2} = {2,3,5,7}.
    
    # Let me represent each kernel as a frozenset of primes.
    # Then for each M1, I need M2 such that kernel_set(M2) = kernel_set(M1) XOR {2,3,5,7}.
    
    # Let me precompute kernel_set for all M.
    
    def _kernel_prime_set(n):
        """Return frozenset of primes with odd exponent in n."""
        f = n
        primes = set()
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    primes.add(d)
            d += 1 if d == 2 else 2
        if f > 1:
            primes.add(f)
        return frozenset(primes)
    
    # Precompute kernel prime sets
    kernel_sets = [None] * (MAX + 1)
    for i in range(211, MAX + 1):
        kernel_sets[i] = _kernel_prime_set(i)
    
    target_set = frozenset({2, 3, 5, 7})
    
    # For each M1, find M2 such that kernel_sets[M2] = kernel_sets[M1] ^ target_set
    # (XOR of sets = symmetric difference)
    
    # Build a dictionary: kernel_set -> list of numbers with that kernel set
    from collections import defaultdict
    set_to_nums = defaultdict(list)
    for i in range(211, MAX + 1):
        set_to_nums[kernel_sets[i]].append(i)
    
    # For each M1, compute needed_set = kernel_sets[M1] ^ target_set
    # Then look up set_to_nums[needed_set] and find M2 != M1 with smallest max(M1, M2)
    
    best_max = MAX + 1
    best_pair = None
    
    for M1 in range(211, MAX + 1):
        needed = kernel_sets[M1] ^ target_set
        if needed in set_to_nums:
            for M2 in set_to_nums[needed]:
                if M2 != M1:
                    m = max(M1, M2)
                    if m < best_max:
                        best_max = m
                        best_pair = (M1, M2)
    
    # Check time
    if time.time() > deadline - 20:
        if best_pair is not None:
            return [best_pair[0], best_pair[1]]
        return [840]
    
    if best_pair is not None:
        # Verify
        M1, M2 = best_pair
        prod = M1 * M2 * N
        if _is_perfect_square(prod):
            return [M1, M2]
    
    # No 2-number solution found? Try 3 numbers.
    # For 3 numbers: kernel_set(M1) ^ kernel_set(M2) ^ kernel_set(M3) = target_set
    # This is more complex. Let me try a different approach.
    
    # Actually, let me just return the best 2-number solution or the single-number solution.
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    
    # Fallback to single number
    return [840]

# EVOLVE-BLOCK-END
