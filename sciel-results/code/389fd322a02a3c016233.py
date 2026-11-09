import time
import math
import numpy as np

def _is_square_free(n):
    if n < 2:
        return True
    for p in range(2, int(math.isqrt(n)) + 1):
        if (n % (p * p)) == 0:
            return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set([1])
    max_v = int(limit ** (1/3)) + 2
    sq_free_v = []
    for v in range(1, max_v + 1):
        if _is_square_free(v):
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

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin

    # We will try a limit that allows efficient chunked summation
    # L = 10^7 gives ~30k powerful numbers.
    # L = 2*10^7 gives ~40k powerful numbers.
    # Let's start with 10^7 to be safe on time, then maybe try higher if fast.
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    limits_to_try = [10**7, 5*10**6, 10**6]

    for L in limits_to_try:
        if time.time() > deadline - 20:
            break
        
        try:
            P = _generate_powerful(L)
            L_len = len(P)
            
            if L_len < 2:
                continue
                
            P_np = np.array(P, dtype=np.int64)
            max_sum = 2 * L
            # Use int32 for counts if possible, but int64 is safer for large counts?
            # Max count is L_len, which is small. int32 is fine.
            counts = np.zeros(max_sum + 1, dtype=np.int32)
            
            chunk_size = 1000
            
            for start in range(0, L_len, chunk_size):
                end = min(start + chunk_size, L_len)
                chunk = P_np[start:end]
                # Compute outer sum
                sums = np.add.outer(chunk, P_np)
                # Add to counts
                # We need to accumulate. 
                # np.add.at is slow. 
                # Better to bincount the chunk and add to global counts.
                chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
                counts += chunk_counts
                del sums
                del chunk_counts
                
                if time.time() > deadline - 20:
                    break
            
            if time.time() > deadline - 20:
                break
            
            # Find N with max count
            # Note: counts[N] is the number of pairs (a,b) with a,b in P, a+b=N.
            # This includes (a,b) and (b,a).
            # We need to calculate distinct pairs {a,b} with a<=b.
            
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            # Check if N/2 is powerful
            d = 0
            if best_N % 2 == 0:
                half_N = best_N // 2
                # Check if half_N is in P
                # Since P is sorted, we can use binary search or set
                # For speed, use set
                P_set = set(P)
                if half_N in P_set:
                    d = 1
            
            num_pairs = (C + d) // 2
            
            if num_pairs > best_score:
                best_score = num_pairs
                # Extract pairs
                pairs = []
                # We need a<=b
                # Iterate through P
                # For each a, b = N - a
                # If b >= a and b in P_set, add
                for a in P:
                    b = best_N - a
                    if b >= a:
                        if b in P_set:
                            pairs.append([a, b])
                    else:
                        # Since P is sorted and a is increasing, b is decreasing.
                        # If b < a, we will never have b >= a for larger a.
                        # So we can break?
                        # Yes, if a > N/2, then b < N/2 < a.
                        # So we can break when a > N/2.
                        if a > best_N // 2:
                            break
                        # Actually, if b < a, it means a > N/2.
                        # So we can break.
                        pass
                
                # Verify pairs count
                if len(pairs) == num_pairs:
                    best_result = {'N': best_N, 'pairs': pairs}
                else:
                    # Fallback: just use the first num_pairs? No, must be valid.
                    # If mismatch, something is wrong.
                    # Let's just return what we have if it's valid.
                    pass
                    
        except MemoryError:
            continue
        except Exception:
            continue

    return best_result
