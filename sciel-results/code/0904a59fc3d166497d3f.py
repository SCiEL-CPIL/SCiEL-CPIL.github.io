    if not valid:
        return cands_r2
    
    return np.around(best_arr).astype(np.int64)
