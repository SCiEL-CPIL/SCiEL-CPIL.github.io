import time
import bisect
import numpy as np

MAXV_LIMIT = 10000000
WALL_LIMIT = 90.0
SAFETY = 0.85

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    try:
        N = MAXV_LIMIT

        # ---------- Compute sigma(n) using vectorized sieve ----------
        # sigma[n] = sum of divisors of n
        sigma = np.zeros(N + 1, dtype=np.int64)
        # This loop is O(N log N) but vectorized inner loop
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        if time_left() < 20.0:
            # Fallback if sieve took too long (unlikely)
            return [11, 6]

        # ---------- Group indices by sigma value ----------
        # We want to group n by sigma[n]
        # Create an array of indices 1..N
        indices = np.arange(1, N + 1, dtype=np.int64)
        sigma_vals = sigma[1:]
        
        # Sort indices by sigma value
        # lexsort sorts by last key first, so (indices, sigma_vals) sorts by sigma_vals then indices
        order = np.lexsort((indices, sigma_vals))
        sorted_sigma = sigma_vals[order]
        sorted_indices = indices[order]

        # Find group boundaries
        # changes[i] is True if sorted_sigma[i] != sorted_sigma[i-1]
        changes = np.empty(len(sorted_sigma), dtype=bool)
        changes[0] = True
        changes[1:] = sorted_sigma[1:] != sorted_sigma[:-1]
        starts = np.nonzero(changes)[0]
        ends = np.append(starts[1:], len(sorted_sigma))

        best_val = float('inf')
        best_pair = None

        # Process groups
        # We want to find pairs (a, b) in the same group minimizing |a/b - 1.5|
        # which is |2a - 3b| / (2b)
        
        # To improve performance, we can skip groups that are too small
        # Or process groups with larger elements first?
        # Actually, the error depends on b. Larger b allows smaller error for same numerator.
        # So we should check large b values first.
        # However, iterating over all groups is necessary unless we prune.
        
        for gi in range(len(starts)):
            if time_left() < 5.0:
                break

            s0 = starts[gi]
            e0 = ends[gi]
            length = e0 - s0
            if length < 2:
                continue

            # Extract the group of numbers
            # They are already sorted by index (due to lexsort secondary key)
            gn = sorted_indices[s0:e0]
            L = len(gn)
            
            # Convert to list for bisect? 
            # Bisect on numpy array works but is slower than on list.
            # For small L, list is fine. For large L, numpy searchsorted is better.
            # Let's use numpy searchsorted for consistency and speed on large arrays.
            
            # We want to iterate b from largest to smallest to find small errors early?
            # Or just iterate all.
            # Let's iterate all b in the group.
            # For each b, find a in gn closest to 1.5 * b.
            
            # Vectorized approach for the group?
            # targets = 1.5 * gn
            # Find insertion points
            # But we need to check neighbors.
            
            # Let's stick to loop for clarity and bisect efficiency, 
            # but use numpy searchsorted which is O(log N) per call.
            
            # Actually, since gn is sorted, we can do:
            # for b in gn:
            #    target = 1.5 * b
            #    pos = np.searchsorted(gn, target)
            #    check gn[pos] and gn[pos-1]
            
            # This is O(L log L) per group.
            # Total complexity depends on sum of L log L over all groups.
            # Since sum L = N, and log L <= log N, this is roughly O(N log N).
            
            for bi in range(L):
                b = int(gn[bi])
                target = 1.5 * b
                pos = np.searchsorted(gn, target)
                
                # Check candidate at pos
                if pos < L:
                    a = int(gn[pos])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check candidate at pos - 1
                if pos > 0:
                    a = int(gn[pos - 1])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                                
            # Optional: Early termination if best_val is extremely small
            # Theoretical min is 1/(2*10^7) = 5e-8
            if best_val < 1e-7:
                # We found a very good pair. 
                # Can we do better? Maybe 1/(2*10^7) is the floor for b=10^7.
                # If we find 5e-8, we can't get much better unless b is larger (impossible) 
                # or we find a pair with numerator 1 and larger b.
                # But b is bounded by 10^7.
                # So if we find val < 5.1e-8, it's likely optimal or close.
                # Let's continue to be safe, but we can break if we hit the theoretical limit.
                if best_val <= 1.0 / (2 * N):
                    break

        if best_pair is None:
            return [11, 6]

        a, b = best_pair
        if not (1 <= a <= N and 1 <= b <= N):
            return [11, 6]
        if sigma[a] != sigma[b]:
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]

        return [int(a), int(b)]

    except Exception:
        return [11, 6]
