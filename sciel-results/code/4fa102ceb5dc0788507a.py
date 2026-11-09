import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

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

    # Try different limits, starting high to maximize the search space
    # L ~ C * M^(1/2), so for L=10000, M~1e8
    limits_to_try = [200_000_000, 150_000_000, 100_000_000, 50_000_000]
    
    best_result = None
    
    for M in limits_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            L = len(P)
            
            # Convert to numpy array
            P_np = np.array(P, dtype=np.int64)
            
            # Generate sums using outer addition
            sums = np.add.outer(P_np, P_np)
            
            # Find max sum for bincount size
            max_sum_val = int(sums.max())
            
            # Count occurrences
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            # Find the sum with max count
            max_count_idx = int(np.argmax(counts))
            best_N = max_count_idx
            C = int(counts[best_N])
            
            # Calculate distinct pairs count
            d = 0
            if best_N % 2 == 0:
                half_N = best_N // 2
                if half_N in P:
                    d = 1
            
            num_pairs = (C + d) // 2
            
            # Recover pairs
            pairs = []
            P_set = set(P)
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            current_result = {'N': best_N, 'pairs': pairs}
            
            # Update best result if this one has more pairs
            if best_result is None or len(pairs) > len(best_result['pairs']):
                best_result = current_result
            
            # Return the first successful result (largest M)
            return current_result
            
        except MemoryError:
            # Try next smaller limit
            continue
        except Exception:
            continue

    # Fallback if all failed or time ran out
    if best_result:
        return best_result
        
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
