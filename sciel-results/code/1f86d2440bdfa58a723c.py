<<<<<<< SEARCH
def _get_tight_ap_info(signs, top_k=40):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max_sum - 1
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = s
        while idx < n:
            idx_count[idx] += 1
            idx += d
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()
=======
def _get_tight_ap_info(signs, top_k=40):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for idx_arr in _AP_INDICES:
        cumsum = np.cumsum(signs_arr[idx_arr])
        current_max = int(np.max(cumsum) - np.min(cumsum))
        if current_max > 0:
            ap_sums.append((current_max, idx_arr))
        if current_max > max_sum:
            max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max_sum - 1
    tight_aps = [idx_arr for (val, idx_arr) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for idx_arr in tight_aps:
        idx_count[idx_arr] += 1
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()
>>>>>>> REPLACE

Now let me improve the main search loop to do more iterations and use smarter moves:

