def local_search(perm, p, deadline=None):
    n = len(perm)
    if n == 0:
        return perm
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline and time.time() > deadline:
                return perm
            job = perm[i]
            perm.pop(i)
            best_pos = i
            best_val = float('inf')
            
            for j in range(n):
                perm.insert(j, job)
                ms = calc_makespan_fast(perm, p)
                if ms < best_val:
                    best_val = ms
                    best_pos = j
                perm.pop(j)
            
            perm.insert(best_pos, job)
            
            if best_pos != i:
                improved = True
                break
    return perm
