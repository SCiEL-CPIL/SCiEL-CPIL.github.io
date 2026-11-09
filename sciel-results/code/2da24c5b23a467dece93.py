import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def generate_square_free(limit):
        """Sieve for square-free numbers up to limit."""
        if limit < 1:
            return []
        is_sq_free = [True] * (limit + 1)
        is_sq_free[0] = False
        p = 2
        while p * p <= limit:
            if is_sq_free[p]:
                sq = p * p
                for j in range(sq, limit + 1, sq):
                    is_sq_free[j] = False
            p += 1
        return [i for i in range(1, limit + 1) if is_sq_free[i]]

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

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    # Try different limits for powerful numbers
    # For M=10^8: ~10k powerful numbers, P^2=10^8 sums, 800MB int64
    # For M=5*10^7: ~7k powerful numbers, P^2=5*10^7 sums, 400MB
    limits_to_try = [100_000_000, 50_000_000, 20_000_000]
    
    for M in limits_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            if L < 10:
                continue

            P_np = np.array(P, dtype=np.int64)
            
            # Check if P^2 fits in memory (8 bytes per element)
            if L * L * 8 > 3 * 1024**3:  # 3GB limit
                continue

            # Compute all pair sums
            sums = np.add.outer(P_np, P_np)
            max_sum_val = int(sums.max())
            
            # Count occurrences of each sum
            # Use int32 if possible to save memory
            if max_sum_val <= 2**31 - 1:
                sums_int = sums.astype(np.int32)
                counts = np.bincount(sums_int.ravel(), minlength=max_sum_val + 1)
            else:
                counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            max_count_idx = int(np.argmax(counts))
            best_N = max_count_idx
            C = int(counts[best_N])
            
            # Calculate distinct pairs (a <= b)
            P_set = set(P)
            pairs = []
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            current_score = len(pairs)
            if current_score > best_score:
                best_score = current_score
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we successfully processed the largest limit, return
            if M == limits_to_try[0]:
                break

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result
