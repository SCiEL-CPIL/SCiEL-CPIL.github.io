import math
import time
import numpy as np
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit."""
    powerful = set([1])
    
    def dfs(start, current):
        if current > limit:
            return
        for i in range(start, int(math.isqrt(limit // current)) + 1):
            # i is the base prime (we'll use all integers, filter by being prime later? No, powerful requires prime factors squared)
            # Actually, powerful numbers: every prime factor appears with exponent >= 2.
            # So we need to build from primes.
            pass
    
    # Better: use primes
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    # Powerful numbers up to 10^18. The largest prime we need: p^2 <= 10^18 => p <= 10^9.
    # But we also need combinations. Let's use a recursive generation.
    
    # Primes up to 10^6 are enough for most, but we need up to 10^9 for single-prime squares.
    # For p^2 <= 10^18, p <= 10^9. That's too many primes to sieve.
    # Alternative: generate powerful numbers by iterating over possible "square parts".
    
    # A powerful number n = s * t^2 where s is squarefree... no, that's not right.
    # A powerful number has all prime exponents >= 2. So n = prod p_i^{e_i}, e_i >= 2.
    # We can write n = a^2 * b where b is powerful and squarefree... no.
    # Actually: n is powerful iff n = x^2 * y where y is powerful and squarefree? No.
    # If n = prod p_i^{e_i} with e_i >= 2, then n = (prod p_i^{floor(e_i/2)})^2 * (prod p_i^{e_i mod 2}).
    # Since e_i >= 2, e_i mod 2 is 0 or 1. If e_i is odd, e_i >= 3, so p_i appears in the squarefree part with exponent 1, but also in the square part.
    # This is getting complicated. Let me just do DFS over primes.
    
    # For efficient generation, note that powerful numbers <= X are O(X^{1/2} + X^{1/3} + ...) which is small.
    # Known: number of powerful numbers <= 10^18 is about 2,740,000.
    
    # Use the standard algorithm:
    # n is powerful iff n = a^2 * b where b is powerful and squarefree... 
    # Actually the cleanest: n powerful <=> n = prod p^e, e>=2.
    # Generate by: for each squarefree d, and for each powerful m, if gcd(d,m)=1 then d*m... no.
    
    # Standard approach: n powerful iff n = x^2 * y where y is powerful and squarefree, and gcd(x,y)=1? 
    # Let's just do: iterate over all possible "core" powerful numbers built from small primes,
    # and multiply by squares.
    
    # Simpler: DFS over primes with exponent >= 2.
    # We need primes up to 10^9 for p^2 <= 10^18. But we can't sieve up to 10^9.
    # Instead: any powerful number n <= 10^18 can be written as n = a^2 * b where b is powerful and b <= 10^9 (since a^2 * b <= 10^18 and b >= 1, a <= 10^9).
    # And b is powerful, so b = prod p_i^{e_i}, e_i >= 2. The largest prime factor of b: p^2 <= b <= 10^9, so p <= 31623.
    # So we only need primes up to ~31623 for the "b" part, and then a can be up to 10^9.
    
    # Algorithm:
    # 1. Find all powerful numbers b <= 10^9 (using primes up to 31623).
    # 2. For each such b, for each a such that a^2 * b <= 10^18, i.e., a <= isqrt(10^18 / b),
    #    check if a^2 * b is powerful. But a^2 * b is powerful iff all prime factors of a also have exponent >= 2 in the total,
    #    which means a must be powerful too (since a^2 contributes exponent 2*exp_a, and b contributes exp_b, total = 2*exp_a + exp_b >= 2 always).
    #    Wait: if a = prod q_j^{f_j}, then a^2 = prod q_j^{2f_j}. Combined with b = prod p_i^{e_i}, 
    #    the total exponent for prime q is 2f_q + e_q (where e_q=0 if q not in b). We need 2f_q + e_q >= 2 for all q.
    #    If q is in a but not in b: 2f_q >= 2, OK since f_q >= 1.
    #    If q is in b but not in a: e_q >= 2, OK.
    #    If q in both: 2f_q + e_q >= 2 + 2 = 4, OK.
    #    So a^2 * b is ALWAYS powerful for any positive integer a and powerful b!
    #    
    #    This is a key insight: n is powerful iff n = a^2 * b for some integer a and powerful b.
    #    And every such product is powerful.
    #    
    #    So: generate all powerful b <= 10^9, then for each b, all a with a^2*b <= 10^18 give powerful numbers.
    #    But we'd get duplicates. We need unique powerful numbers.
    #    
    #    Actually this gives a bijection? No, because different (a,b) can give same n.
    #    E.g., n = 36 = 6^2 * 1 = 2^2 * 9 = 3^2 * 4.
    #    
    #    So this approach generates duplicates. We need a different method.
    
    # Let's just do the standard DFS with primes up to 10^6 (which covers most cases) and handle large primes separately.
    # For a powerful number n <= 10^18, if it has a prime factor p > 10^6, then p^2 > 10^12, so n >= p^2 > 10^12.
    # And n = p^2 * k where k is powerful and k <= 10^18/p^2 < 10^6.
    # So k is a powerful number < 10^6, which we can generate with small primes.
    # And p^2 < 10^18, so p < 10^9.
    # 
    # Strategy:
    # 1. Generate all powerful numbers <= 10^6 using primes <= 1000.
    # 2. For each such small powerful k, find all primes p such that p^2 * k <= 10^18, i.e., p <= isqrt(10^18/k).
    #    For each such prime p, p^2*k is powerful.
    # 3. Also, powerful numbers with all prime factors <= 10^6 are already covered in step 1 (extended).
    # 
    # This is still complex. Let me use a simpler but effective approach for this time budget.
    
    # Given the 150s limit, let me use a practical generation:
    # Generate powerful numbers by: for each square s = a^2 (a up to 10^9), and for each powerful b,
    # n = s * b if n <= 10^18. But this is too many.
    
    # Practical approach: use the known fact that there are ~2.7M powerful numbers <= 10^18.
    # Generate them via DFS with pruning, using primes up to 10^5 (sufficient for the "core" part).
    
    def gen_powerful(limit):
        result = set()
        
        # Sieve primes up to 10^6
        max_prime = 10**6
        sieve = bytearray(b'\x01') * (max_prime + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(max_prime)) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        primes = [i for i in range(2, max_prime+1) if sieve[i]]
        
        def dfs(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                result.add(current)
                return
            p = primes[idx]
            # Try exponent 0 (skip this prime)
            dfs(idx+1, current)
            # Try exponent >= 2
            val = p * p
            if val > limit // current:
                return
            while current * val <= limit:
                dfs(idx+1, current * val)
                if val > limit // current // p:
                    break
                val *= p
        
        dfs(0, 1)
        return sorted(result)
    
    return gen_powerful(limit)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Generate all powerful numbers up to 10^18
    # This is the expensive part. Let's time it.
    t0 = time.time()
    powerful_list = _generate_powerful(10**18)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    powerful_set = set(powerful_list)
    t1 = time.time()
    print(f"Generated {len(powerful_list)} powerful numbers in {t1-t0:.2f}s")
    
    if time_left() < 30:
        # Not enough time, return a fallback
        # Find a good N by sampling
        return _fallback_construct(powerful_arr, powerful_set, time_left, deadline)
    
    # Now find N with maximum number of representations.
    # For each powerful number a, for each powerful number b >= a, s = a+b.
    # Count frequencies of s.
    # But O(n^2) is too slow for n=2.7M.
    
    # Alternative: for a candidate N, count = #{a in powerful : a <= N/2 and N-a in powerful}.
    # We can do this in O(n) per N using a set lookup, or O(n log n) with binary search.
    
    # Strategy: try many candidate N values and count representations.
    # Good candidates: N values that are sums of many powerful numbers.
    # Heuristic: N near the median of pairwise sums, or N = 2 * (some powerful number).
    
    # Better: use the structure. If N is even and N/2 is powerful, that's one representation.
    # But we want MANY representations.
    
    # Let's try: for a range of N values around 10^18, count representations.
    # But we need to be smart about which N to try.
    
    # Key insight: the number of representations r(N) = #{(a,b) : a+b=N, a,b powerful, a<=b}.
    # This is maximized when N is such that many powerful numbers a have N-a also powerful.
    
    # Approach: sample a subset of powerful numbers, compute their pairwise sums, and find the most common sum.
    # With 2.7M numbers, we can't do all pairs. But we can do a random sample.
    
    # Better approach: for each powerful number a, compute N - a for a fixed N and check if it's powerful.
    # Vectorize this with numpy.
    
    # Let's try several candidate N values:
    # 1. N = 2 * powerful_number (so a=b=powerful_number is one rep)
    # 2. N values that are "round" numbers
    # 3. N values found by sampling pairwise sums
    
    # First, let's try to find a good N by sampling.
    n_powerful = len(powerful_arr)
    
    # Sample-based approach: pick random pairs of powerful numbers, compute sums, find mode.
    np.random.seed(42)
    sample_size = min(50000, n_powerful)
    
    # For efficiency, let's focus on N values in a specific range.
    # The maximum possible N is 2 * max(powerful) = 2 * 10^18, but we need N <= 10^18.
    # So a + b <= 10^18 with a,b powerful.
    
    # Let's try N values that are multiples of small powerful numbers, or N = p^2 * k for various p,k.
    
    # Actually, let's just try a set of candidate N values and count representations for each.
    # Candidate N values:
    # - N = 2 * x where x is powerful (gives at least one rep: x+x)
    # - N = a + b for random pairs (a,b) of powerful numbers
    # - N values that are highly composite in some sense
    
    # Let's generate candidates from random pairs and from 2*powerful.
    candidates = set()
    
    # Add 2 * powerful numbers (where 2*powerful <= 10^18)
    for p in powerful_arr:
        if p * 2 <= 10**18:
            candidates.add(p * 2)
        if len(candidates) > 100000:
            break
    
    # Add random pair sums
    for _ in range(200000):
        i = np.random.randint(0, n_powerful)
        j = np.random.randint(0, n_powerful)
        s = powerful_arr[i] + powerful_arr[j]
        if s <= 10**18 and s >= 2:
            candidates.add(s)
    
    candidates = list(candidates)
    print(f"Trying {len(candidates)} candidate N values")
    
    # For each candidate N, count representations.
    # Use numpy: for a given N, a is powerful and N-a is powerful, a <= N/2.
    # powerful_arr is sorted. For a in powerful_arr with a <= N/2, check if N-a in powerful_set.
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Sort candidates by some heuristic: prefer N values that might have many reps.
    # N values where N/2 is powerful might be good.
    # Let's just iterate and keep the best.
    
    # To speed up, use numpy binary search instead of set lookup.
    # For a given N:
    #   a_candidates = powerful_arr[powerful_arr <= N//2]
    #   b_candidates = N - a_candidates
    #   Check which b_candidates are in powerful_arr using searchsorted.
    
    for idx, N in enumerate(candidates):
        if time_left() < 15:
            break
        
        if N < 2 or N > 10**18:
            continue
        
        # Find all a in powerful_arr with a <= N//2
        half = N // 2
        # Binary search for the end of a <= half
        end_idx = np.searchsorted(powerful_arr, half, side='right')
        if end_idx == 0:
            continue
        
        a_vals = powerful_arr[:end_idx]
        b_vals = N - a_vals
        
        # Check which b_vals are in powerful_arr
        # Use searchsorted: for each b, find if it exists in powerful_arr
        # b_vals is sorted in reverse order (since a_vals is sorted ascending, b_vals = N - a_vals is descending)
        # Let's sort b_vals and search.
        
        # Actually, b_vals = N - a_vals where a_vals is ascending, so b_vals is descending.
        # Let's reverse to make it ascending for binary search.
        b_sorted = b_vals[::-1].copy()
        
        # For each b in b_sorted, check if it's in powerful_arr
        # Use searchsorted with a tolerance
        indices = np.searchsorted(powerful_arr, b_sorted)
        # Check if the found index points to the correct value
        valid = (indices < n_powerful) & (powerful_arr[np.clip(indices, 0, n_powerful-1)] == b_sorted)
        
        count = np.sum(valid)
        
        if count > best_count:
            best_count = count
            best_N = N
            # Extract pairs
            valid_b = b_sorted[valid]
            # For each valid b, a = N - b
            valid_a = N - valid_b
            # Ensure a <= b
            pairs_tmp = []
            for a_val, b_val in zip(valid_a, valid_b):
                if a_val <= b_val and a_val >= 1 and b_val >= 1:
                    pairs_tmp.append([int(a_val), int(b_val)])
            best_pairs = pairs_tmp
        
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(candidates)} candidates, best so far: N={best_N}, count={best_count}")
    
    if best_count == 0:
        # Fallback
        return _fallback_construct(powerful_arr, powerful_set, time_left, deadline)
    
    return {'N': int(best_N), 'pairs': best_pairs}


def _fallback_construct(powerful_arr, powerful_set, time_left, deadline):
    """Fallback: try a few specific N values."""
    best_N = 0
    best_pairs = []
    best_count = 0
    
    # Try N = 2 * p for powerful p
    for p in powerful_arr[:1000]:
        N = p * 2
        if N > 10**18 or N < 2:
            continue
        pairs = []
        half = N // 2
        end_idx = np.searchsorted(powerful_arr, half, side='right')
        a_vals = powerful_arr[:end_idx]
        b_vals = N - a_vals
        b_sorted = b_vals[::-1].copy()
        indices = np.searchsorted(powerful_arr, b_sorted)
        valid = (indices < len(powerful_arr)) & (powerful_arr[np.clip(indices, 0, len(powerful_arr)-1)] == b_sorted)
        count = np.sum(valid)
        if count > best_count:
            best_count = count
            best_N = N
            valid_b = b_sorted[valid]
            valid_a = N - valid_b
            pairs = []
            for a_val, b_val in zip(valid_a, valid_b):
                if a_val <= b_val and a_val >= 1 and b_val >= 1:
                    pairs.append([int(a_val), int(b_val)])
            best_pairs = pairs
    
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
    
    return {'N': int(best_N), 'pairs': best_pairs}


if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
