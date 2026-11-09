<<<<<<< SEARCH
def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]
=======
def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]
>>>>>>> REPLACE

