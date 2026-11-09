import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def generate_square_free(limit):
        """Sieve for square-free numbers up to limit."""
        if limit < 1:
            return []
        is_sq_free = [True] * (limit + 1)
        is_sq_free[0] = False
        # Mark multiples of p^2 as not square-free
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

    # Try limits in descending order of size to maximize potential pairs
    # M=5e7 -> |P|~15k, P^2~2.25e8, sums~1.8GB, bincount~800MB. ~2.6GB total.
    # M=8e7 -> |P|~18k, P^2~3.2e8, sums~2.5GB, bincount~1.3GB. ~3.8GB total. Risky.
    limits_to_try = [80_000_000, 50_000_000, 30_000_000]
    
    for M in limits_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            if L < 10:
                continue

            P_np = np.array(P, dtype=np.int64)
            
            # Check memory estimate: sums array (L*L*8) + bincount array (2*M*8)
            mem_sums = L * L * 8
            mem_bincount = 2 * M * 8
            if mem_sums + mem_bincount > 4 * 1024**3: # 4GB limit
                continue

            sums = np.add.outer(P_np, P_np)
            max_sum_val = int(sums.max())
            
            # Use bincount without minlength to save memory if possible,
            # but we need indices up to max_sum_val. Use minlength to be safe.
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            
            # Calculate distinct pairs
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
                
            # If we found a good score from the largest M, return early
            if M == limits_to_try[0]:
                return best_result

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result
