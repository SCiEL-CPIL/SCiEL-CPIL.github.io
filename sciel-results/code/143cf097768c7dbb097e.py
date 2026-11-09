import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe
    deadline = start_time + 110.0
    
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
        max_b = int(limit ** (1.0/3.0))
        # Optimize square-free check by precomputing primes? 
        # For limit 10^7, max_b ~ 2154. Simple trial division is fine.
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

    # Phase 1: Find a candidate N with many representations in a small range
    # Small range L1 = 10^6. M ~ 100. M^2 = 10^4. Very fast.
    L1 = 10**6
    p_small = generate_powerful(L1)
    
    counts_small = defaultdict(int)
    for i, a in enumerate(p_small):
        for b in p_small[i:]:
            counts_small[a + b] += 1
            
    # Sort N by count
    sorted_ns = sorted(counts_small.items(), key=lambda x: x[1], reverse=True)
    
    # Take top K candidates
    candidates = [n for n, c in sorted_ns[:50]]
    
    # Phase 2: For each candidate N, count representations in a much larger range
    # We need powerful numbers up to max(N) to count pairs a+b=N.
    # Max N from phase 1 is 2*L1 = 2*10^6.
    # But we want to find representations using larger powerful numbers.
    # So we need powerful numbers up to max(candidates).
    # Wait, if a+b=N and a,b powerful, then a <= N.
    # So we need powerful numbers up to max(candidates).
    
    max_N_candidate = max(candidates) if candidates else 2
    # We can actually search for powerful numbers up to 10^18, but we only need those <= max_N_candidate.
    # However, if we pick a small N, we might miss a better N that has many reps with large a,b.
    # But the "hot" N in small range is likely a good seed.
    
    # Let's generate powerful numbers up to max_N_candidate.
    # If max_N_candidate is 2*10^6, this is small.
    # What if the best N is larger?
    # We can also consider that N might be large.
    # Let's add a few "large" candidates: e.g., sums of two large powerful numbers?
    # No, let's stick to the small-range heuristic first.
    
    p_large = generate_powerful(max_N_candidate)
    p_set = set(p_large)
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for N in candidates:
        if time.time() > deadline - 5:
            break
            
        pairs = []
        for a in p_large:
            if a > N // 2:
                break
            b = N - a
            if b in p_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs

    # Fallback if no candidates found or all empty
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
