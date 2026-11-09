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

    # Try different limits, starting high to maximize the search space
    # We estimate that M=5e8 gives ~60k numbers, which is manageable with chunking
    limits_to_try = [500_000_000, 300_000_000, 200_000_000]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    for M in limits_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            L = len(P)
            P_np = np.array(P, dtype=np.int64)
            P_set = set(P)
            
            # We need to count sums a+b for a,b in P.
            # Max sum is 2*M.
            max_sum = 2 * M
            
            # Using chunked addition to save memory
            # We process P in chunks of size CHUNK_SIZE
            # For each chunk P_chunk, we compute P_chunk[:, None] + P[None, :]
            # This creates a (CHUNK_SIZE x L) array of sums.
            # We bincount these and add to global counts.
            
            CHUNK_SIZE = 500  # Adjust based on memory. 500 * 60000 * 8 bytes = 240 MB. Safe.
            counts = np.zeros(max_sum + 1, dtype=np.int32)
            
            for i in range(0, L, CHUNK_SIZE):
                if time.time() > deadline - 5:
                    break
                end = min(i + CHUNK_SIZE, L)
                P_chunk = P_np[i:end]
                # Compute sums for this chunk
                sums = P_chunk[:, None] + P_np[None, :]
                # Flatten and count
                chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
                counts += chunk_counts.astype(np.int32)
                
            # Find the sum with max count
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            # Recover pairs
            pairs = []
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            if len(pairs) > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we found a very good result, we might stop?
            # But larger M usually yields better results.
            # However, if M=5e8 works, it's likely the best we can do in time.
            # So we break after the first successful large M.
            if M >= 500_000_000:
                break

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result
