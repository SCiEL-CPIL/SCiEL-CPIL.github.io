import time
import numpy as np
from math import isqrt

def construct():
    t0 = time.time()
    T_LIMIT = 55.0
    LIM = 10000

    # Sieve via bytearray slicing
    is_p = bytearray(b'\x01') * (LIM + 1)
    is_p[0] = is_p[1] = 0
    for m in range(2, isqrt(LIM) + 1):
        if is_p[m]:
            is_p[m * m::m] = bytearray(len(range(m * m, LIM + 1, m)))
    P = np.array([x for x in range(2, LIM + 1) if is_p[x]], dtype=np.int64)
    n = len(P)
    if n < 2:
        return [int(x) for x in P]

    # dp[s] running value for current start, and back[s]
    # For each start s, iterate j from s+1 to n-1 with two-pointer
    dp_row = np.zeros(n, dtype=np.int32)   # dp_row[j] = best length ending at j for current s... 
    # Actually let me think differently.
    
    # State: dp[s, j] = length of longest valid subseq starting at P[s], second element P[j]
    # Store as 2D or flat. For memory, flat is fine: n*n = ~4.9M for n~2262 -> OK
    # dp[s*n + j], back[s*n + j]
    
    total = n * n
    D = np.zeros(total, dtype=np.int32)
    B = np.full(total, -1, dtype=np.int32)

    best_len = 2
    best_s_idx = 0
    best_j_idx = 1

    for s in range(n - 1):
        Ps = P[s]
        # j2 pointer: for fixed s, as j increases, threshold=2*Ps-P[j] decreases
        # j2min (smallest j2 with P[j2] >= threshold) is non-increasing
        # j2 starts at s+1, j starts at s+1
        j2 = s + 1  # pointer for predecessor gap index
        # For each j > s:
        #   threshold = 2*Ps - P[j]
        #   while j2 < j and P[j2] < threshold: j2 += 1
        #   valid j2 range: [j2, j-1] intersect [s+1, j-1]
        #   D[s,j] = max(D[s,j2'] + 1 for j2' in [j2, j-1]) if such range non-empty, else 2
        
        # To get max quickly, maintain a running max from the right
        # As j increases, j2 is non-decreasing (threshold decreases, so j2 moves right)
        # Wait: threshold = 2*Ps - P[j], as j increases P[j] increases, threshold decreases
        # We need P[j2] >= threshold, so as threshold decreases, more j2 qualify
        # j2min decreases. So j2 (our pointer) should decrease? 
        # No: j2 is the LEFT boundary of valid range. As threshold drops, left boundary moves left.
        # But j2 must be >= s+1.
        # So j2 = max(s+1, searchsorted(P, threshold)) 
        # As j increases, threshold decreases, searchsorted gives smaller values.
        # So j2 is non-increasing. We can use a two-pointer moving LEFT as j increases? No...
        #
        # Let me re-examine: for fixed s, iterate j from s+1 to n-1.
        # threshold(j) = 2*P[s] - P[j], decreasing in j.
        # j2min(j) = smallest j2 such that P[j2] >= threshold(j). Since P is increasing,
        # as threshold decreases, j2min decreases (or stays). So j2min is non-increasing in j.
        # But we also need j2 >= s+1 and j2 < j.
        # So valid range is [max(s+1, j2min(j)), j-1].
        #
        # For the max D[s, j2'] over j2' in [max(s+1, j2min(j)), j-1]:
        # As j increases by 1, we add j'=j-1 to the candidate set AND the left boundary may move left.
        # This is not a simple sliding window; the left boundary moves left while right boundary moves right.
        # 
        # Alternative: precompute suffix max of D[s, *] once per s.
        # For fixed s, D[s, s+1], D[s, s+2], ..., D[s, n-1]
        # Suffix max: suff[j] = max(D[s, j], suff[j+1])
        # Then D[s, j] = suff[j2min(j)] + 1 if j2min(j) < j, else 2
        #
        # But D[s, j] depends on D[s, j2'] for j2' < j, so we need to compute in order.
        # Suff max can only be computed after all D[s,*] are known.
        # So we can't use suffix max directly.
        
        # Let me just do: for fixed s, for j from s+1 to n-1:
        #   Compute j2min via binary search
        #   Need max of D[s, s+1..j-1] where index >= j2min
        #   Maintain a data structure or just scan
        #
        # For n~2262, inner loop is O(n) per j, total O(n^3) ~ 11 billion. Too slow.
        
        # The original used: for fixed i (middle element), iterate j (right element),
        # and use suffix max over k (left element) which is precomputable per i.
        # That's O(n^2) total. Let me follow that pattern but with different code structure.
        
        break  # This approach won't work, go back to original pattern

    # Let me just write it cleanly in the original algorithm style
    # but with numpy arrays and different variable names/structure
    
    # Reinitialize
    D2 = np.zeros((n, n), dtype=np.int32)
    B2 = np.full((n, n), -1, dtype=np.int32)
    
    best_len = 2
    best_mid = 0
    best_right = 1
    
    for mid in range(n - 1):
        Pm = P[mid]
        
        # For mid=0, all D2[0, j] = 2
        if mid == 0:
            D2[0, 1:] = 2
            B2[0, 1:] = -1
            continue
        
        # Build suffix max arrays for column mid (rows 0..mid-1)
        # suff_val[k] = max(D2[k, mid], suff_val[k+1]) for k from mid-1 down to 0
        # suff_idx[k] = argmax
        if mid > 0:
            sv = np.empty(mid, dtype=np.int32)
            si = np.empty(mid, dtype=np.int32)
            sv[mid - 1] = D2[mid - 1, mid]
            si[mid - 1] = mid - 1
            for k in range(mid - 2, -1, -1):
                if D2[k, mid] >= sv[k + 1]:
                    sv[k] = D2[k, mid]
                    si[k] = k
                else:
                    sv[k] = sv[k + 1]
                    si[k] = si[k + 1]
        
        # Two-pointer: for j from mid+1 to n-1
        # threshold = 2*Pm - P[j]
        # k_ptr = smallest k such that P[k] >= threshold (using binary search)
        # Valid k range: [k_ptr, mid-1]
        # D2[mid, j] = sv[k_ptr] + 1 if k_ptr < mid, else 2
        
        # As j increases, threshold decreases, k_ptr is non-increasing
        # But k_ptr can't go below 0. Start k_ptr at mid (invalid), decrease as needed.
        k_ptr = mid  # Start at mid (invalid, means no valid k)
        for j in range(mid + 1, n):
            thresh = 2 * Pm - P[j]
            # Decrease k_ptr while P[k_ptr - 1] >= thresh
            while k_ptr > 0 and P[k_ptr - 1] >= thresh:
                k_ptr -= 1
            
            if k_ptr < mid:
                nv = sv[k_ptr] + 1
                D2[mid, j] = nv
                B2[mid, j] = si[k_ptr]
                if nv > best_len:
                    best_len = nv
                    best_mid = mid
                    best_right = j
            else:
                D2[mid, j] = 2
                B2[mid, j] = -1
        
        if time.time() - t0 > T_LIMIT:
            break
    
    # Reconstruct
    if best_len <= 2:
        return [int(P[0]), int(P[1])]
    
    result = [int(P[best_right]), int(P[best_mid])]
    cm = best_mid
    cr = best_right
    while True:
        bk = int(B2[cm, cr])
        if bk < 0:
            break
        result.append(int(P[bk]))
        cr = cm
        cm = bk
    result.reverse()
    return result
