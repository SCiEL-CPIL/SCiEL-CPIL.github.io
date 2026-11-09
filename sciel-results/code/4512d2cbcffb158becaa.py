import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    # 140s deadline to be safe (150s limit, buffer for overhead)
    deadline = start_time + 140.0
    
    # 1. Generate powerful numbers up to a limit L
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # We want L large enough to capture many candidates, but small enough to generate quickly.
    # L = 2 * 10^8 seems reasonable.
    L = 2 * 10**8
    
    def is_square_free(n):
        if n < 2:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 1
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        return sorted(powerful)

    # Phase 1: Generate powerful numbers up to L
    p_small = generate_powerful(L)
    M = len(p_small)
    
    # Phase 2: Probabilistic Heuristic to select candidate N
    # We estimate the expected number of representations for N.
    # E[R(N)] = sum_{a in P, a <= N/2} Pr(N-a is powerful)
    # Pr(x is powerful) ~ C * x^{-1/2} (heuristic density)
    # So E[R(N)] ~ C * sum_{a in P, a <= N/2} (N-a)^{-1/2}
    
    # We can't sum over all N up to 2L efficiently for all a.
    # Instead, we can sample N values or use a coarse grid.
    # Let's use a coarse grid of N values in [L, 2L].
    # Step size for grid: 1000
    # Number of points: L / 1000 = 2 * 10^5.
    # For each N, we sum over a in P such that a <= N/2.
    # This is O(M * NumGridPoints) which is 4000 * 200000 = 8*10^8. Too slow in Python.
    
    # Alternative:
    # Iterate over powerful numbers a.
    # For each a, it contributes to E[R(N)] for N in [2a, 2L] (assuming b >= a, so N >= 2a? No, b can be < a? No, a<=b implies a <= N/2).
    # Wait, the pair is (a,b) with a<=b. So a <= N/2.
    # For a fixed a, it contributes to N such that N >= 2a and N <= 2L.
    # The contribution is Pr(N-a is powerful) ~ (N-a)^{-1/2}.
    # We want to find N that maximizes sum_{a <= N/2} (N-a)^{-1/2}.
    
    # Let's use a different heuristic.
    # The density of powerful numbers is ~ C x^{-1/2}.
    # The number of pairs summing to N is roughly the integral of the product of densities.
    # But powerful numbers are sparse.
    
    # Let's stick to the "hot spot" idea but with a better selection.
    # The current program uses a small L1=10^7 and finds the max count there.
    # The insight from the knowledge cell is: "Increasing the search limit M from 2e+08 to 5e+08 allows the algorithm to explore a larger range of N values... Measured: 7 -> 7.733."
    # This suggests that larger N (within the range of generated powerful numbers) might have more representations, or that finding a "hot" N in a larger range is better.
    
    # Let's try to find the "hottest" N in the range [L, 2L] using a sampling approach.
    # We can't do full O(M^2) for L=2e8.
    # But we can sample.
    
    # Actually, let's look at the current program's weakness.
    # It finds candidates in [0, 2*10^7] and verifies them.
    # The max N it considers is ~2*10^7.
    # If we generate powerful numbers up to L=2*10^8, we can consider N up to 4*10^8.
    # But verifying N requires checking if N-a is powerful.
    # If we have the set of powerful numbers up to L, we can only verify N if N-a <= L for all a?
    # No, if N is large, say 4*10^8, and a is small, N-a is large.
    # We need powerful numbers up to N.
    # If we generate up to L=2*10^8, we can only verify N up to 2*10^8 (if a=0, b=N, but a>=1).
    # Actually, if a+b=N and a,b are powerful, and max(a,b) <= L, then N <= 2L.
    # So if we generate powerful numbers up to L, we can verify any N <= 2L.
    
    # So, let's generate powerful numbers up to L=2*10^8.
    # Then we can search for N in [1, 2L].
    # How to find the best N?
    # The heuristic: E[R(N)] is maximized when N is large?
    # The density of powerful numbers decreases as x^{-1/2}.
    # The number of pairs for N is sum_{a} I(N-a in P).
    # This is the additive energy.
    
    # Let's try a different heuristic:
    # Pick N from the top "hot" spots found in a smaller range, but scale them?
    # No, the structure is not scale invariant.
    
    # Let's use the "probabilistic heuristic" as requested.
    # We want to maximize E[R(N)] = sum_{a in P, a <= N/2} Pr(N-a in P).
    # Approximate Pr(x in P) by 0 if x > L (since we don't know), or use the density model.
    # If we assume the density model holds up to 2L, we can estimate.
    
    # Let's compute the score for a sample of N values.
    # Sample N values: 1000 random values in [L, 2L].
    # For each N, estimate E[R(N)] using the density model.
    # Then verify the top few.
    
    # But the density model is crude.
    # Let's try a hybrid:
    # 1. Generate P up to L=2*10^8.
    # 2. Use the small-range hot spots (from L1=10^7) as seeds.
    # 3. Also sample some large N values.
    # 4. Verify all candidates.
    
    # Actually, the most effective strategy might be to just generate P up to L=2*10^8 and find the N with the most representations in [1, 2L].
    # But finding the max N in [1, 2L] is O(M^2) which is too slow.
    
    # Let's go back to the "hot spot" idea but with a larger L1.
    # If L1 = 2*10^7, M ~ 4000. M^2 = 16*10^6. Fast.
    # Find top K hot spots in [0, 2*L1].
    # Then verify them using the full P up to L=2*10^8.
    # This allows us to find pairs (a,b) where a,b can be up to 2*10^8.
    # So N can be up to 4*10^8.
    # But the hot spots are in [0, 4*10^7].
    # Are there better N's in [4*10^7, 4*10^8]?
    # The knowledge cell says "Increasing the search limit M from 2e+08 to 5e+08 allows the algorithm to explore a larger range of N values... Measured: 7 -> 7.733."
    # This implies that the best N found was in the larger range.
    # So we should search for hot spots in a larger range.
    
    # Let's use L1 = 5*10^7 for the hot spot search.
    # M ~ 5000. M^2 = 25*10^6. Still fast.
    # Find top K hot spots in [0, 2*L1] = [0, 10^8].
    # Then verify them using P up to L=2*10^8.
    
    # Wait, if we verify N=10^8, we need a+b=10^8.
    # a and b are powerful.
    # If a=1, b=10^8-1. Is 10^8-1 powerful? Unlikely.
    # We need a and b to be in P.
    # So we need P to contain a and b.
    # If we generate P up to L=2*10^8, we can verify any N <= 2*10^8.
    # So if we find a hot spot N in [0, 10^8], we can verify it.
    
    # Let's try this.
    
    L1 = 5 * 10**7
    p_small = generate_powerful(L1)
    
    counts_small = defaultdict(int)
    for i, a in enumerate(p_small):
        for b in p_small[i:]:
            s = a + b
            if s <= 2 * L1:
                counts_small[s] += 1
    
    sorted_ns = sorted(counts_small.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, c in sorted_ns[:50]]
    
    # Also add some random candidates in [L1, 2*L1] to explore larger N
    import random
    random.seed(42)
    for _ in range(50):
        candidates.append(random.randint(L1, 2 * L1))
    
    # Generate powerful numbers up to L=2*10^8 for verification
    L = 2 * 10**8
    p_full = generate_powerful(L)
    p_set = set(p_full)
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Check each candidate
    for N in candidates:
        if time.time() > deadline - 10:
            break
            
        pairs = []
        # Iterate through powerful numbers a <= N/2
        # We can use bisect to find the range in p_full
        import bisect
        mid = N // 2
        idx = bisect.bisect_right(p_full, mid)
        for i in range(idx):
            a = p_full[i]
            b = N - a
            if b in p_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs

    # If we have time, we can try a few more random candidates
    if time.time() < deadline - 20:
        for _ in range(100):
            N = random.randint(1, 2 * L)
            if time.time() > deadline - 5:
                break
            pairs = []
            mid = N // 2
            idx = bisect.bisect_right(p_full, mid)
            for i in range(idx):
                a = p_full[i]
                b = N - a
                if b in p_set:
                    pairs.append([a, b])
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N
                best_pairs = pairs

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
