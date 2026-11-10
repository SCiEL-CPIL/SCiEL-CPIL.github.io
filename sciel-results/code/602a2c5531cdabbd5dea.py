    is_res = root_arr != 0
    # p is 1009
    # We need s such that s..s+4 are valid indices and True
    # s ranges 1..1004
    # is_res indices 0..1008
    # We check windows starting at index s-1 (0-indexed)
    # Window covers indices s-1, s, s+1, s+2, s+3
    # Corresponding to values s, s+1, s+2, s+3, s+4
    
    # Vectorized check
    # all_5[i] is True if is_res[i], is_res[i+1], ..., is_res[i+4] are all True
    all_5 = is_res[0:p-4] & is_res[1:p-3] & is_res[2:p-2] & is_res[3:p-1] & is_res[4:p]
    
    hits = np.where(all_5)[0]
    if len(hits) == 0:
        raise RuntimeError('No run found')
        
    s_0 = hits[0]
    s = s_0 + 1
    
    roots = [int(root_arr[s+i]) for i in range(5)]
    
    return {'start': s, 'roots': roots}
