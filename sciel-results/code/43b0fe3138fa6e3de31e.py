import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        # Precompute square-free v
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

    # Try different limits, starting high to maximize the search space
    # Larger M allows larger N and potentially more representations
    limits_to_try = [500_000_000, 300_000_000, 200_000_000, 100_000_000]
    
    best_result = None
    
    for M in limits_to_try:
        if time.time() > deadline:
            break
            
        try:
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            L = len(P)
            
            # Convert to numpy array
            P_np = np.array(P, dtype=np.int64)
            
            # Generate sums
            # Using outer addition
            sums = np.add.outer(P_np, P_np)
            
            # Find max sum for bincount size
            max_sum_val = sums.max()
            
            # Count occurrences
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            # Find the sum with max count
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
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
                
            # If we found a good number of pairs, we might stop early to save time?
            # But we want the max, so let's try the next (smaller) limit only if we failed or if we want to be safe.
            # Actually, larger M is better. If 500M works, it's likely better than 300M.
            # So if the first one succeeds, we can return it?
            # No, it's possible that a smaller M finds a "lucky" N with more pairs? Unlikely.
            # The density of powerful numbers is higher for smaller numbers, but the range is smaller.
            # Generally, max reps increases with N.
            # So the largest M that fits should give the best result.
            
            # We can break if we successfully processed the largest limit
            # But we should verify it didn't hit a timeout or error.
            # If we are here, it succeeded.
            # Let's assume the largest M is best.
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
        counts = np.bincount(sums.ravel(), minlength=sums.max() + 1)
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
