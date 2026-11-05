import numpy as np
import time

def order_jobs(processing_times):
    P = np.asarray(processing_times, dtype=np.float64)
    n, m = P.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
        
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def _calc_makespan(perm, P, n, m):
        if n == 0: return 0.0
        c = np.zeros(m, dtype=np.float64)
        P_local = P
        m_local = m
        for job in perm:
            p = P_local[job]
            nc = np.empty(m_local, dtype=np.float64)
            nc[0] = c[0] + p[0]
            for k in range(1, m_local):
                if c[k] > nc[k-1]:
                    nc[k] = c[k] + p[k]
                else:
                    nc[k] = nc[k-1] + p[k]
            c = nc
        return float(c[-1])

    def neh_init(P):
        n, m = P.shape
        totals = P.sum(axis=1)
        order = np.argsort(-totals)
        seq = []
        for job in order:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                ms = _calc_makespan(new_seq, P, n, m)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq.insert(best_pos, job)
        return seq

    def local_search_insertion(perm, P, n, m):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                job = perm[i]
                best_j = -1
                best_ms = _calc_makespan(perm, P, n, m)
                
                for j in range(n):
                    if j == i: continue
                    new_perm = perm[:i] + perm[i+1:]
                    new_perm.insert(j, job)
                    ms = _calc_makespan(new_perm, P, n, m)
                    if ms < best_ms:
                        best_ms = ms
                        best_j = j
                if best_j != -1:
                    perm.pop(i)
                    perm.insert(best_j, job)
                    improved = True
        return perm

    # 1. Init
    current_perm = neh_init(P)
    best_perm = current_perm.copy()
    best_ms = _calc_makespan(best_perm, P, n, m)
    
    # 2. Local Search
    t_start_ls = time.time()
    current_perm = local_search_insertion(current_perm, P, n, m)
    ls_time = time.time() - t_start_ls
    cur_ms = _calc_makespan(current_perm, P, n, m)
    if cur_ms < best_ms:
        best_ms = cur_ms
        best_perm = current_perm.copy()
        
    # 3. ILS with SA
    T = 1.0
    cooling = 0.99
    min_T = 1e-5
    
    # Estimate time per ILS iteration (Perturb + LS)
    # Perturb is negligible. LS is dominant.
    # If ls_time is too large, we might not fit many iterations.
    
    while time.time() < deadline:
        # Check if we have enough time for another LS pass
        # If time remaining < ls_time * 1.1, break
        if time.time() + ls_time * 1.1 > deadline:
            break
            
        # Perturb
        k = np.random.randint(2, 6)
        perturbed = current_perm.copy()
        for _ in range(k):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
        # Local Search
        new_perm = local_search_insertion(perturbed, P, n, m)
        new_ms = _calc_makespan(new_perm, P, n, m)
        
        # Acceptance
        delta = new_ms - cur_ms
        if delta < 0:
            cur_ms = new_ms
            current_perm = new_perm
            if cur_ms < best_ms:
                best_ms = cur_ms
                best_perm = current_perm.copy()
        else:
            if T > min_T:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    cur_ms = new_ms
                    current_perm = new_perm
        
        T *= cooling

    return best_perm
