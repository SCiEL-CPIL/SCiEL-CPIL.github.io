# EVOLVE-BLOCK-START
import numpy as np

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Uses multi-start NEH heuristic followed by fast adjacent-swap local search.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    p = processing_times
    n = n_jobs
    M = n_machines
    
    def compute_makespan_and_C(perm, p):
        """
        Compute makespan and completion matrix C.
        """
        n_seq = len(perm)
        if n_seq == 0:
            return 0.0, np.zeros((0, M))
        
        C = np.zeros((n_seq, M))
        p_perm = p[perm]
        
        for i in range(n_seq):
            for m in range(M):
                if i == 0 and m == 0:
                    C[i, m] = p_perm[i, m]
                elif i == 0:
                    C[i, m] = C[i, m-1] + p_perm[i, m]
                elif m == 0:
                    C[i, m] = C[i-1, m] + p_perm[i, m]
                else:
                    C[i, m] = max(C[i-1, m], C[i, m-1]) + p_perm[i, m]
        
        return C[-1, -1], C

    def neh_initial(p, sort_key='total'):
        """
        NEH heuristic.
        """
        if sort_key == 'total':
            totals = p.sum(axis=1)
            order = list(np.argsort(-totals))
        elif sort_key == 'min':
            mins = p.min(axis=1)
            order = list(np.argsort(-mins))
        elif sort_key == 'max':
            maxs = p.max(axis=1)
            order = list(np.argsort(-maxs))
        else:
            order = list(np.arange(len(p)))
            np.random.shuffle(order)
            order = list(order)
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                candidate = seq[:pos] + [job] + seq[pos:]
                ms, _ = compute_makespan_and_C(candidate, p)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq

    def local_search_adjacent(perm, p, max_passes=50):
        """
        Fast local search using adjacent swaps.
        """
        n_seq = len(perm)
        best_ms, best_C = compute_makespan_and_C(perm, p)
        improved = True
        
        while improved and max_passes > 0:
            improved = False
            max_passes -= 1
            best_swap = None
            best_swap_ms = best_ms
            best_swap_C = None
            
            for i in range(n_seq - 1):
                # Evaluate swap of i and i+1
                # We need to recompute C from i onwards
                new_perm = perm[:]
                new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
                
                new_C = np.zeros((n_seq, M))
                if i > 0:
                    new_C[:i, :] = best_C[:i, :]
                
                p_perm = p[new_perm]
                
                # Recompute from i
                for pos in range(i, n_seq):
                    for m in range(M):
                        if pos == 0 and m == 0:
                            new_C[pos, m] = p_perm[pos, m]
                        elif pos == 0:
                            new_C[pos, m] = new_C[pos, m-1] + p_perm[pos, m]
                        elif m == 0:
                            new_C[pos, m] = new_C[pos-1, m] + p_perm[pos, m]
                        else:
                            new_C[pos, m] = max(new_C[pos-1, m], new_C[pos, m-1]) + p_perm[pos, m]
                
                swap_ms = new_C[-1, -1]
                if swap_ms < best_swap_ms - 1e-10:
                    best_swap_ms = swap_ms
                    best_swap = i
                    best_swap_C = new_C
            
            if best_swap is not None:
                i = best_swap
                perm[i], perm[i+1] = perm[i+1], perm[i]
                best_ms = best_swap_ms
                best_C = best_swap_C
                improved = True
                
        return perm, best_ms

    # Multi-start
    best_overall_perm = None
    best_overall_ms = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Start 1: NEH Total
    perm = neh_initial(p, 'total')
    perm_ls, ms_ls = local_search_adjacent(perm, p, max_passes=30)
    if ms_ls < best_overall_ms:
        best_overall_ms = ms_ls
        best_overall_perm = perm_ls
        
    # Start 2: NEH Min
    perm = neh_initial(p, 'min')
    perm_ls, ms_ls = local_search_adjacent(perm, p, max_passes=30)
    if ms_ls < best_overall_ms:
        best_overall_ms = ms_ls
        best_overall_perm = perm_ls
        
    # Start 3: NEH Max
    perm = neh_initial(p, 'max')
    perm_ls, ms_ls = local_search_adjacent(perm, p, max_passes=30)
    if ms_ls < best_overall_ms:
        best_overall_ms = ms_ls
        best_overall_perm = perm_ls
        
    # Start 4: Random
    perm = list(np.arange(n))
    rng.shuffle(perm)
    perm = list(perm)
    perm_ls, ms_ls = local_search_adjacent(perm, p, max_passes=20)
    if ms_ls < best_overall_ms:
        best_overall_ms = ms_ls
        best_overall_perm = perm_ls

    return best_overall_perm
# EVOLVE-BLOCK-END
