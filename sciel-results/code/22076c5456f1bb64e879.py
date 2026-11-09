import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    def generate_square_free(limit):
        """Generate all square-free numbers up to limit using sieve."""
        if limit < 2:
            return [1]
        is_sf = bytearray(b'\x01') * (limit + 1)
        is_sf[0] = 0
        for p in range(2, int(math.isqrt(limit)) + 1):
            p2 = p * p
            for m in range(p2, limit + 1, p2):
                is_sf[m] = 0
        return [i for i in range(1, limit + 1) if is_sf[i]]

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = generate_square_free(max_v)
        
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    def is_powerful(n):
        """Check if n is powerful by exact factorization."""
        if n <= 1:
            return True
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                count = 0
                while temp % d == 0:
                    temp //= d
                    count += 1
                if count == 1:
                    return False
            d += 1 if d == 2 else 2
        return True

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    # Generate powerful numbers up to a moderate limit
    M = 100_000_000
    if time.time() > deadline - 10:
        return best_result
    
    try:
        P = generate_powerful(M)
        L = len(P)
        P_np = np.array(P, dtype=np.int64)
        P_set = set(P)
        
        # Strategy: Use probabilistic heuristic to select candidate N
        # For each powerful number a, estimate P(N - a is powerful)
        # Then sum over a to get expected representations for N
        
        # We'll sample several candidate N values and estimate their scores
        # The key insight: powerful numbers have density ~ c * n^(-1/2) for large n
        # But for our range, we can use empirical density
        
        # First, compute the empirical density of powerful numbers in ranges
        # to build a better estimator
        
        # Sample candidate N values: try multiples of small numbers, 
        # and values near 2*M where density might be different
        
        candidate_ns = set()
        
        # Try N values that are sums of pairs of small powerful numbers
        # This is likely to yield many representations
        small_P = P_np[:min(1000, L)]
        
        # Generate candidate N values from sums of small powerful numbers
        for i in range(min(200, len(small_P))):
            for j in range(i, min(200, len(small_P))):
                candidate_ns.add(int(small_P[i] + small_P[j]))
        
        # Also add some random candidates
        for _ in range(50):
            candidate_ns.add(int(np.random.randint(10_000, 2 * M)))
        
        # Sort candidates and evaluate top ones
        candidates = sorted(candidate_ns)
        
        # For each candidate, estimate the expected number of representations
        # by checking how many a in P have N - a also in P
        # But do this efficiently using numpy
        
        best_estimated_score = 0
        best_N_candidate = None
        best_pairs = []
        
        # Evaluate candidates in batches for efficiency
        batch_size = 50
        for start in range(0, len(candidates), batch_size):
            if time.time() > deadline - 20:
                break
            
            batch = candidates[start:start + batch_size]
            batch_np = np.array(batch, dtype=np.int64)
            
            # For each N in batch, count how many a in P have N - a in P
            # Use broadcasting: (batch_size, L) matrix of N - a
            # Then check membership
            
            # To save memory, process in smaller sub-batches
            sub_batch_size = 10
            for sub_start in range(0, len(batch), sub_batch_size):
                if time.time() > deadline - 10:
                    break
                
                sub_batch = batch_np[sub_start:sub_start + sub_batch_size]
                
                # For each N, compute N - P and check if in P_set
                # This is O(sub_batch_size * L) set lookups
                for idx, N_val in enumerate(sub_batch):
                    N_int = int(N_val)
                    if N_int < 2 or N_int > 2 * M:
                        continue
                    
                    # Count pairs
                    count = 0
                    for a in P_np:
                        b = N_int - int(a)
                        if b < a:
                            continue
                        if b in P_set:
                            count += 1
                    
                    if count > best_estimated_score:
                        best_estimated_score = count
                        best_N_candidate = N_int
                        # Recover pairs for this N
                        best_pairs = []
                        for a in P:
                            b = N_int - a
                            if b >= a and b in P_set:
                                best_pairs.append([a, b])
        
        if best_N_candidate is not None and len(best_pairs) > len(best_result['pairs']):
            best_result = {'N': best_N_candidate, 'pairs': best_pairs}
            
    except Exception:
        pass

    return best_result
