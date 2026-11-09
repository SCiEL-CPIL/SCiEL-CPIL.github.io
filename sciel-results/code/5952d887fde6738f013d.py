import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
    # Precompute as numpy arrays for faster iteration
    block_tsubs_np = [np.array(subs, dtype=np.int32) for subs in block_tsubs]
    
    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            subs = block_tsubs_np[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    for bi in tsub_blocks[si]:
                        block_gain[bi] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        
        total_count = np.zeros(N_t, dtype=np.int32)
        for i in range(n_sel):
            for si in block_tsubs_np[selected_indices[i]]:
                total_count[si] += 1
                
        crit_count = np.zeros(n_sel, dtype=np.int32)
        for i in range(n_sel):
            for si in block_tsubs_np[selected_indices[i]]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = list(np.where(crit_count == 0)[0])
        removed = np.zeros(n_sel, dtype=np.int8)
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = 1
            g = selected_indices[i]
            
            for si in block_tsubs_np[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in global_to_local:
                            i_other = global_to_local[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search(selected_indices, max_iter=200):
        """
        Local search: try to remove blocks or perform 1-swaps to improve.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs_np[g]:
                total_count[si] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < DEADLINE - 10:
            improved = False
            iter_count += 1
            # Try removals
            for i in range(len(current)):
                g = current[i]
                removable = True
                for si in block_tsubs_np[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs_np[g]:
                        total_count[si] -= 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 1-swap: remove one non-critical block, add another
            if len(current) > 1:
                # Compute criticality for each block
                block_crit = np.zeros(len(current), dtype=np.int32)
                for i, g in enumerate(current):
                    for si in block_tsubs_np[g]:
                        if total_count[si] == 1:
                            block_crit[i] += 1
                
                # Try removing blocks with 0 criticality
                zero_crit = np.where(block_crit == 0)[0]
                random.shuffle(zero_crit.tolist())
                
                found_swap = False
                for i in zero_crit.tolist():
                    g_old = current[i]
                    
                    # Find valid replacement: must cover all t-subsets that only g_old covers
                    # i.e., for every si in g_old with total_count[si]==1, g_new must contain si
                    candidates = []
                    n_samples = min(80, N_blocks)
                    for _ in range(n_samples):
                        g_new = random.randint(0, N_blocks - 1)
                        if g_new in current_set:
                            continue
                        # Check: all t-subsets in g_old with count==1 must be in g_new
                        valid = True
                        for si in block_tsubs_np[g_old]:
                            if total_count[si] == 1:
                                if si not in block_tsubs_np[g_new]:
                                    valid = False
                                    break
                        if valid:
                            candidates.append(g_new)
                    
                    for g_new in candidates:
                        current[i] = g_new
                        current_set.discard(g_old)
                        current_set.add(g_new)
                        for si in block_tsubs_np[g_old]:
                            total_count[si] -= 1
                        for si in block_tsubs_np[g_new]:
                            total_count[si] += 1
                        improved = True
                        found_swap = True
                        break
                    
                    if found_swap:
                        break
        
        return current

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Multi-start with local search
    while time.time() < DEADLINE - 15.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        cand_idx = local_search(cand_idx, max_iter=100)
        
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
    
    # Final polish on best solution
    if best_sol and time.time() < DEADLINE - 5.0:
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_idx = [block_to_idx[b] for b in best_sol]
        improved = local_search(best_idx, max_iter=500)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol = [all_blocks[i] for i in improved]
    
    return best_sol
