<<<<<<< SEARCH
    # If we have time and scipy, try to find rank-23
    elapsed = time.time() - start_time
    if HAVE_SCIPY and elapsed < 30:
        result_r23 = _try_rank23(start_time, deadline, idx_u, idx_v, idx_w, n, m, p)
        if result_r23 is not None:
            return result_r23

    # Return the verified rank-26
    return result_r26
=======
    # If we have time and scipy, try to find lower rank via merge-and-refine
    elapsed = time.time() - start_time
    if HAVE_SCIPY and elapsed < 20:
        # Try Rank 24 first (merge 2 columns from 26)
        result_r24 = _try_merge_and_refine(24, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, result_r26)
        if result_r24 is not None:
            return result_r24
        
        # Try Rank 23 (merge 3 columns from 26)
        if time.time() < deadline - 10:
            result_r23 = _try_merge_and_refine(23, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, result_r26)
            if result_r23 is not None:
                return result_r23

    # Return the verified rank-26
    return result_r26
>>>>>>> REPLACE

We need to define `_try_merge_and_refine` and remove the old `_try_rank23` or replace it.

