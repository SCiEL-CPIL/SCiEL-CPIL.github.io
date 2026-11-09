import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

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

    # Phase 1: Generate powerful numbers up to a moderate limit L1
    # This is used to generate candidate N values and to estimate the score.
    # L1 = 10^7 gives ~2000 powerful numbers.
    L1 = 10**7
    p_small = generate_powerful(L1)
    
    # Phase 2: Candidate Generation via Probabilistic Heuristic
    # Instead of brute-force counting pairs (which is O(M^2)), we use a heuristic 
    # to estimate the "density" of powerful numbers around N.
    # The heuristic is based on the idea that if N is a sum of two powerful numbers 
    # in the small range, it is likely a "good" target. 
    # However, the directive asks for a probabilistic heuristic: 
    # Score(N) = sum_{a in P} P(N-a is powerful)
    # Where P(x is powerful) is approximated.
    # A common heuristic for powerful numbers density near x is roughly x^{-1/3} or similar.
    # But a better "structural" heuristic is to look at the "additive energy" in a 
    # slightly larger window or use the fact that powerful numbers are sparse.
    
    # Let's try a different structural approach: 
    # We know that powerful numbers are of the form a^2 b^3.
    # Let's generate candidates N by summing pairs from p_small, but instead of 
    # counting frequency, we rank them by a heuristic that predicts how many 
    # representations they might have in the larger range.
    
    # Actually, the most direct interpretation of the directive "maximizing the 
    # expected number of powerful representations, estimated via the sum of the 
    # probability that N - a is powerful for each powerful number a" suggests:
    
    # 1. Pick a set of candidate N's.
    # 2. For each N, estimate E[N] = sum_{a in Powerful} P(N-a is powerful).
    # 3. Select N with highest E[N].
    
    # What is P(x is powerful)?
    # The density of powerful numbers near x is approximately 1/sqrt(x) * constant? 
    # Actually, the count of powerful numbers <= X is ~ sqrt(X). So density is ~ 1/(2*sqrt(X)).
    # So P(x is powerful) ~ 1/(2*sqrt(x)).
    
    # So Score(N) = sum_{a in P, a <= N} 1/(2*sqrt(N-a)).
    
    # To get candidates, we can still use the sums from p_small, but we can 
    # expand the candidate pool by also considering sums from a slightly larger 
    # set, or just use the top sums from p_small and score them.
    
    # Let's generate candidate N's from p_small sums (top 50 by frequency is a good 
    # proxy for "structurally rich" N, but let's score them all).
    
    counts_small = defaultdict(int)
    # We can optimize the pair counting a bit by breaking early if needed, but 
    # 2000^2 / 2 = 2M ops is fast.
    for i, a in enumerate(p_small):
        for b in p_small[i:]:
            s = a + b
            if s <= 2 * L1:
                counts_small[s] += 1
    
    # Get all unique candidates from the small range sums
    # We might miss candidates that are not sums of small powerful numbers but 
    # have many large representations. However, "hot" numbers in small range 
    # are good candidates.
    
    candidates = list(counts_small.keys())
    
    # To ensure we have enough candidates, let's also add some random large N's?
    # No, let's stick to the "hot" ones first.
    
    # Score each candidate N using the probabilistic heuristic
    # Score(N) = sum_{a in p_small, a <= N} 1/sqrt(N-a)
    # Note: We only sum over a in p_small because p_small is our "probe" set.
    # This is an approximation of the sum over all powerful a.
    
    scored_candidates = []
    for N in candidates:
        score = 0.0
        # Sum over a in p_small
        for a in p_small:
            if a >= N:
                break
            b = N - a
            if b > 0:
                # Heuristic probability that b is powerful
                # Density of powerful numbers ~ 1/sqrt(b)
                score += 1.0 / math.sqrt(b)
        
        scored_candidates.append((N, score, counts_small[N]))
        
        if time.time() > deadline - 20:
            break

    # Sort by heuristic score, then by small-range count as tie-breaker
    scored_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Phase 3: Verification
    # Take the top K candidates and verify them exactly.
    # We need to generate powerful numbers up to the maximum candidate N.
    
    top_k = 10
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for N, score, small_count in scored_candidates[:top_k]:
        if time.time() > deadline - 5:
            break
            
        # Generate powerful numbers up to N
        # This might be expensive if N is large.
        # Max N is ~ 2 * 10^7.
        # generate_powerful(2*10^7) is fast.
        
        p_full = generate_powerful(N)
        p_set = set(p_full)
        
        pairs = []
        for a in p_full:
            if a > N // 2:
                break
            b = N - a
            if b in p_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs

    # Fallback if no candidates found or verification failed
    if best_N == 0:
        # Try a simple fallback
        # N=2, pairs [[1,1]]
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
