<<<<<<< SEARCH
def _local_search_remove(selected, N, deadline):
    best = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() > deadline:
                break
            cand = [v for v in best if v != x]
            if not _is_valid_sidon(cand):
                continue
            # Rebuild mask for cand to check maximality efficiently
            cand_mask = _build_sum_mask(cand)
            if _is_maximal(cand, N, cand_mask):
                best = cand
                improved = True
                break
    return best
=======
def _local_search_remove(selected, N, deadline):
    best = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() > deadline:
                break
            cand = [v for v in best if v != x]
            if not _is_valid_sidon(cand):
                continue
            # Rebuild mask for cand to check maximality efficiently
            cand_mask = _build_sum_mask(cand)
            if _is_maximal(cand, N, cand_mask):
                best = cand
                improved = True
                break
    return best
>>>>>>> REPLACE

