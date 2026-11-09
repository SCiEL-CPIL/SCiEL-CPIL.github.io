import numpy as np

def construct():
    p = 1009
    # Vectorized computation of quadratic residues and roots
    x = np.arange(1, p)
    v = (x * x) % p
    # roots[v] = smallest x such that x^2 mod p = v
    roots = np.zeros(p, dtype=np.int32)
    # Use np.minimum.at to keep the minimum root for each residue
    np.minimum.at(roots, v, x)
    
    # Mark residues: roots[v] > 0 means v is a non-zero quadratic residue
    is_res = roots > 0
    
    # Find the smallest s in 1..1004 such that s..s+4 are all residues
    # Use a sliding window sum of is_res
    # conv with [1,1,1,1,1] gives sum over window of size 5
    kernel = np.ones(5)
    window_sum = np.convolve(is_res, kernel, mode='valid')
    
    # window_sum[i] corresponds to window starting at index i (0-based)
    # We need window_sum[i] == 5 for s = i+1 (since s is 1-based, index 0 is value 0)
    # Actually is_res[0] is for residue 0, is_res[1] for residue 1, etc.
    # window starting at index s (1-based) means is_res[s], is_res[s+1], ..., is_res[s+4]
    # In 0-based numpy, that's indices s, s+1, ..., s+4
    # np.convolve(is_res, kernel, 'valid')[i] = sum of is_res[i:i+5]
    # So for s (1-based), we want window_sum[s-1] == 5? No.
    # Let's think: is_res is indexed 0..1008.
    # We want s in 1..1004 such that is_res[s], is_res[s+1], is_res[s+2], is_res[s+3], is_res[s+4] are all True.
    # This is a window of length 5 starting at index s (0-based index s).
    # np.convolve(is_res, ones(5), 'valid') has length p-4 = 1005.
    # result[i] = sum(is_res[i:i+5]) for i in 0..1004.
    # So for s (1-based), the 0-based start index is s.
    # We need result[s] == 5, where s is in 1..1004.
    # So we look at result[1], result[2], ..., result[1004].
    
    # Find the first index in result[1:] that equals 5
    candidates = np.where(window_sum[1:] == 5)[0]
    if len(candidates) > 0:
        s = candidates[0] + 1  # 1-based start
        # Get roots for s, s+1, s+2, s+3, s+4
        r = roots[s:s+5]
        return {'start': int(s), 'roots': [int(ri) for ri in r]}
    
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
