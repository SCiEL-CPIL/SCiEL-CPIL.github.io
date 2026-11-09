import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit using u^2 * v^3 form."""
        powerful = set([1])
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
                
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

    best_result = None
    best_score = 0

    # Try progressively larger limits, but with a smarter approach:
    # Use chunked outer-sum to handle larger P without OOM.
    limits_to_try = [10**9, 5*10**8, 2*10**8, 10**8, 5*10**7]

    for M in limits_to_try:
        if time.time() > deadline - 15:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            # For L > 15000, use chunked approach
            if L > 15000:
                # Chunked sum counting to avoid memory blowup
                P_np = np.array(P, dtype=np.int64)
                max_sum = 2 * M
                counts = np.zeros(max_sum + 1, dtype=np.int32)
                chunk_size = max(1, 10**8 // L)  # Aim for ~10^8 elements per chunk
                
                for start in range(0, L, chunk_size):
                    end = min(start + chunk_size, L)
                    chunk = P_np[start:end]
                    sums = np.add.outer(chunk, P_np)
                    np.add.at(counts, sums.ravel(), 1)
                    del sums
                    
                    if time.time() > deadline - 15:
                        break
                
                if time.time() > deadline - 15:
                    break
                    
                max_count_idx = np.argmax(counts)
                best_N = int(max_count_idx)
                C = int(counts[best_N])
                
                d = 0
                if best_N % 2 == 0:
                    half_N = best_N // 2
                    P_set = set(P)
                    if half_N in P_set:
                        d = 1
                
                num_pairs = (C + d) // 2
                
                if num_pairs > best_score:
                    best_score = num_pairs
                    pairs = []
                    P_set = set(P)
                    for a in P:
                        b = best_N - a
                        if b >= a:
                            if b in P_set:
                                pairs.append([a, b])
                    best_result = {'N': best_N, 'pairs': pairs}
            else:
                # Direct approach for smaller L
                P_np = np.array(P, dtype=np.int64)
                sums = np.add.outer(P_np, P_np)
                max_sum_val = int(sums.max())
                counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
                
                max_count_idx = np.argmax(counts)
                best_N = int(max_count_idx)
                C = int(counts[best_N])
                
                d = 0
                if best_N % 2 == 0:
                    half_N = best_N // 2
                    P_set = set(P)
                    if half_N in P_set:
                        d = 1
                
                num_pairs = (C + d) // 2
                
                if num_pairs > best_score:
                    best_score = num_pairs
                    pairs = []
                    P_set = set(P)
                    for a in P:
                        b = best_N - a
                        if b >= a:
                            if b in P_set:
                                pairs.append([a, b])
                    best_result = {'N': best_N, 'pairs': pairs}
                    
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result is not None:
        return best_result
    
    # Fallback
    try:
        P = generate_powerful(1000)
        P_np = np.array(P, dtype=np.int64)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=int(sums.max()) + 1)
        best_N = int(np.argmax(counts))
        C = int(counts[best_N])
        d = 1 if (best_N % 2 == 0 and best_N // 2 in P) else 0
        num_pairs = (C + d) // 2
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
