    bad_cache = {}
    def get_bad_indices_cached(a_idx, b_idx):
        if a_idx > b_idx:
            a_idx, b_idx = b_idx, a_idx
        key = (a_idx, b_idx)
        if key not in bad_cache:
            bad_cache[key] = get_bad_indices(a_idx, b_idx)
        return bad_cache[key]
