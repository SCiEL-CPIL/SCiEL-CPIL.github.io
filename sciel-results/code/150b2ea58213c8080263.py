import numpy as np
import time

_P = 1009
_M = 5

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # 1. Compute quadratic residues using NumPy vectorized Legendre symbol
    # Legendre symbol (a/p) = a^((p-1)/2) mod p
    # If result is 1, it's a QR. If p-1 (which is -1 mod p), it's a non-QR.
    # We exclude 0 explicitly.
    
    a = np.arange(1, _P)
    # (p-1)/2 = 1008/2 = 504
    exp = (_P - 1) // 2
    
    # np.power supports modular exponentiation with 'mod' parameter
    # Note: np.power(base, exp, mod) is available in recent NumPy versions
    # If not, we can do (base ** exp) % mod, but that's slower for large exp.
    # Let's use the standard modular exponentiation.
    chars = np.power(a, exp, _P)
    
    # chars will contain 1 for QRs and _P-1 for non-QRs
    is_qr = (chars == 1)
    
    # is_qr is a boolean array of length _P-1 (indices 0 to _P-2 corresponding to values 1 to _P-1)
    # We need to find the smallest s (1-indexed value) such that s, s+1, s+2, s+3, s+4 are all True.
    # In the array is_qr, value v is at index v-1.
    # So we need is_qr[s-1], is_qr[s], is_qr[s+1], is_qr[s+2], is_qr[s+3] to all be True.
    
    # Convert to integer array for cumulative sum
    # 1 if non-residue, 0 if residue
    non_residues = (~is_qr).astype(np.int32)
    
    # Cumulative sum of non-residues
    # cumsum[i] = number of non-residues in 1..i
    # To check if s..s+M-1 are all residues, we check if cumsum[s+M-1] - cumsum[s-1] == 0
    # Let's compute prefix sums.
    # prefix[0] = 0
    # prefix[i] = sum(non_residues[0:i]) = number of non-residues in 1..i
    
    prefix = np.zeros(_P, dtype=np.int32)
    np.cumsum(non_residues, out=prefix[1:])
    
    # We want to find smallest s >= 1 such that prefix[s+M-1] - prefix[s-1] == 0
    # s ranges from 1 to _P - _M
    # Let i = s-1. i ranges from 0 to _P - _M - 1
    # Condition: prefix[i+M] - prefix[i] == 0
    
    # Vectorized check
    # indices i from 0 to _P - _M - 1
    # max_i = _P - _M - 1
    # We need prefix[i+_M] - prefix[i]
    
    # Create array of indices
    # i_vals = np.arange(0, _P - _M)
    # diffs = prefix[i_vals + _M] - prefix[i_vals]
    # valid = (diffs == 0)
    # If valid.any(), s = np.argmax(valid) + 1
    
    # Optimization: We can just check the slice directly
    # prefix_slice_end = prefix[_M:_P]  # indices M to P-1
    # prefix_slice_start = prefix[0:_P-_M] # indices 0 to P-M-1
    # diffs = prefix_slice_end - prefix_slice_start
    
    diffs = prefix[_M:_P] - prefix[0:_P-_M]
    
    valid_indices = np.where(diffs == 0)[0]
    
    if len(valid_indices) == 0:
         # Fallback to linear scan if something goes wrong, though it shouldn't
         # This is a safety net
         for s in range(1, _P - _M + 1):
             if all(is_qr[s-1 : s-1+_M]):
                 # Found it
                 pass
             else:
                 continue
             break
         else:
             raise RuntimeError("No solution found in vectorized search, fallback also failed")
         # If we got here, we found s in the loop above? No, the loop logic above is messy.
         # Let's stick to the vectorized result. If empty, it's an error.
         raise RuntimeError("No solution found")
    
    best_s = valid_indices[0] + 1  # +1 because index 0 corresponds to s=1
    
    # 2. Find roots for s, s+1, ..., s+M-1
    # We need r_i such that r_i^2 % _P == s+i
    # We can find roots by scanning x from 1 to _P-1 and checking if x^2 % _P is in {s..s+M-1}
    # Since M is small (5), we can just compute squares for all x and pick the ones that match.
    # Or simpler: for each target value v in s..s+M-1, find a root.
    # We can precompute a map of value -> root.
    
    roots = [0] * _M
    # We need to find x for each v in s..s+M-1
    # Efficient way:
    # Generate all squares x^2 % _P for x in 1.._P-1
    # This is O(P)
    # But we only need 5 specific values.
    # We can iterate x from 1 to _P-1, compute sq = x*x % _P, and if sq is in the target set, record it.
    
    targets = set(range(best_s, best_s + _M))
    found_roots = {}
    
    for x in range(1, _P):
        sq = (x * x) % _P
        if sq in targets:
            found_roots[sq] = x
            if len(found_roots) == _M:
                break
    
    for i in range(_M):
        v = best_s + i
        if v not in found_roots:
             raise RuntimeError(f"Root not found for {v}")
        roots[i] = found_roots[v]
        
    return {'start': best_s, 'roots': roots}

