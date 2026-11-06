import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 78
    n = 256

    def can_add(S_arr, x):
        nS = len(S_arr)
        if nS == 0:
            return True

        # x as 'a': for all b,c in S with x<b<=c, check x ∤ (b+c)
        mask_b = S_arr > x
        if np.any(mask_b):
            B = S_arr[mask_b]
            nB = len(B)
            if nB >= 2:
                sums = B[:, None] + B[None, :]
                upper = np.triu(sums)
                if np.any(upper % x == 0):
                    return False

        # x as 'b': for all a,c in S with a<x<=c, check a ∤ (x+c)
        mask_a = S_arr < x
        mask_c = S_arr >= x
        if np.any(mask_a) and np.any(mask_c):
            A = S_arr[mask_a]
            C = S_arr[mask_c]
            vals = x + C[None, :]
            div_check = vals % A[:, None]
            if np.any(div_check == 0):
                return False

        # x as 'c': for all a,b in S with a<b<=x, check a ∤ (b+x)
        mask_b = S_arr <= x
        if np.any(mask_b):
            B = S_arr[mask_b]
            nB = len(B)
            if nB >= 2:
                for i in range(nB - 1):
                    a = B[i]
                    remainder = (B[i+1:] + x) % a
                    if np.any(remainder == 0):
                        return False

        return True

    best_set = set(range(192, 257))
    best_size = len(best_set)

    # Try ascending extension from various upper tails
    for k in range(1, 257):
        if time.time() - start_time > DEADLINE:
            break
        S = set(range(k, 257))
        S_arr = np.array(sorted(S))
        for x in range(1, k):
            if time.time() - start_time > DEADLINE:
                break
            if can_add(S_arr, x):
                S.add(x)
                S_arr = np.array(sorted(S))
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Try descending extension from promising k values
    if time.time() - start_time < DEADLINE - 10:
        min_best = min(best_set)
        lo = max(1, min_best - 30)
        hi = min(257, min_best + 30)
        for k in range(lo, hi):
            if time.time() - start_time > DEADLINE - 5:
                break
            S = set(range(k, 257))
            S_arr = np.array(sorted(S))
            for x in range(k-1, 0, -1):
                if time.time() - start_time > DEADLINE - 5:
                    break
                if can_add(S_arr, x):
                    S.add(x)
                    S_arr = np.array(sorted(S))
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)

    return sorted(best_set) if best_set else list(range(192, 257))
