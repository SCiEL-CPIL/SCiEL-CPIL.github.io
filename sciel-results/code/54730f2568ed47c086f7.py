import numpy as np
import scipy.sparse as sp
import time
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = range(v)
    t_subs = list(itertools.combinations(points, t))
    num_t_subs = len(t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    k_subs = list(itertools.combinations(points, k))
    num_k_subs = len(k_subs)
    
    rows = []
    cols = []
    block_t_indices = []
    
    for i, k_sub in enumerate(k_subs):
        idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            cols.append(idx)
            rows.append(i)
            idxs.append(idx)
        block_t_indices.append(idxs)
        
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_k_subs, num_t_subs))
    
    best_solution = None
    best_len = float('inf')
    
    def run_greedy_reduction():
        uncovered = np.ones(num_t_subs, dtype=np.int32)
        selected_blocks = []
        
        while np.sum(uncovered) > 0:
            scores = A @ uncovered
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            candidates = np.where(scores == max_score)[0]
            best_idx = np.random.choice(candidates)
            
            selected_blocks.append(best_idx)
            
            idxs = block_t_indices[best_idx]
            uncovered[idxs] = 0
            
        if not selected_blocks:
            return []
            
        cov_counts = np.zeros(num_t_subs, dtype=np.int32)
        for b in selected_blocks:
            idxs = block_t_indices[b]
            cov_counts[idxs] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b = selected_blocks[i]
                idxs = block_t_indices[b]
                if np.any(cov_counts[idxs] == 1):
                    continue
                selected_blocks.pop(i)
                cov_counts[idxs] -= 1
                changed = True
                
        return selected_blocks

    while time.time() < deadline:
        sol = run_greedy_reduction()
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = [list(k_subs[b]) for b in sol]
            if best_len <= 264:
                break

    if best_solution is None:
        # Fallback: return a valid cover, e.g., from the last run
        sol = run_greedy_reduction()
        best_solution = [list(k_subs[b]) for b in sol]

    return best_solution
