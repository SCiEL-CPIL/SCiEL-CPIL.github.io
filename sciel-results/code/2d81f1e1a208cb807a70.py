import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
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
            
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        local_idx_arr = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            local_idx_arr[g] = i
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        i_other = local_idx_arr[g_other]
                        if i_other >= 0 and not removed[i_other]:
                            crit_count[i_other] += 1
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search_swap(selected_indices, time_budget):
        """
        Local search with incremental coverage tracking and priority-based swap evaluation.
        """
        if not selected_indices:
            return selected_indices
        
        # Build initial coverage
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Compute unique t-subsets for each block
        n_sel = len(selected_indices)
        unique_subs_list = [None] * n_sel
        for i in range(n_sel):
            g = selected_indices[i]
            unique_subs_list[i] = [si for si in block_tsubs[g] if total_count[si] == 1]
        
        # Priority queue: (num_unique_subs, index) - process most removable first
        pq = [(len(unique_subs_list[i]), i) for i in range(n_sel)]
        heapq.heapify(pq)
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            n_sel = len(selected_indices)
            
            # Rebuild priority queue if structure changed significantly
            if not pq or len(pq) != n_sel:
                pq = [(len(unique_subs_list[i]), i) for i in range(n_sel) if i < n_sel]
                heapq.heapify(pq)
            
            # Try to find a removable or swappable block
            while pq and time.time() < time_budget:
                _, i = heapq.heappop(pq)
                
                if i >= n_sel or i < 0:
                    continue
                
                g = selected_indices[i]
                unique_subs = unique_subs_list[i]
                
                if not unique_subs:
                    # Can remove this block
                    selected_indices.pop(i)
                    n_sel -= 1
                    global_to_local.pop(g, None)
                    
                    # Update coverage
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                        if total_count[si] == 1:
                            # This t-subset is now critical for the block that covers it
                            for g_other in tsub_blocks[si]:
                                if g_other in global_to_local:
                                    j = global_to_local[g_other]
                                    if j < n_sel:
                                        # Update unique_subs_list[j]
                                        if si not in unique_subs_list[j]:
                                            unique_subs_list[j].append(si)
                    
                    improved = True
                    break
                
                # Try to find a replacement block
                if len(unique_subs) > 25:
                    continue
                
                # Find candidate blocks that cover all unique_subs
                candidate_blocks = None
                for si in unique_subs:
                    if candidate_blocks is None:
                        candidate_blocks = set(tsub_blocks[si])
                    else:
                        candidate_blocks &= set(tsub_blocks[si])
                
                if candidate_blocks is None or not candidate_blocks:
                    continue
                
                candidate_blocks.discard(g)
                
                # Evaluate candidates
                best_candidate = None
                best_cost = len(block_tsubs[g])
                for g_cand in candidate_blocks:
                    if g_cand in global_to_local:
                        continue
                    new_subs = sum(1 for si in block_tsubs[g_cand] if total_count[si] == 0)
                    cost = new_subs
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = g_cand
                
                if best_candidate is not None and best_cost < len(block_tsubs[g]):
                    # Perform swap
                    # Remove old block
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                        if total_count[si] == 1:
                            for g_other in tsub_blocks[si]:
                                if g_other in global_to_local and g_other != g:
                                    j = global_to_local[g_other]
                                    if j < n_sel and si not in unique_subs_list[j]:
                                        unique_subs_list[j].append(si)
                    
                    # Add new block
                    selected_indices[i] = best_candidate
                    global_to_local.pop(g, None)
                    global_to_local[best_candidate] = i
                    unique_subs_list[i] = [si for si in block_tsubs[best_candidate] if total_count[si] == 1]
                    
                    for si in block_tsubs[best_candidate]:
                        total_count[si] += 1
                        if total_count[si] == 2:
                            # No longer critical
                            for g_other in tsub_blocks[si]:
                                if g_other in global_to_local:
                                    j = global_to_local[g_other]
                                    if j < n_sel and si in unique_subs_list[j]:
                                        unique_subs_list[j].remove(si)
                    
                    improved = True
                    break
        
        # Final cleanup: rebuild and reduce
        selected_indices = reduce_solution_fast(selected_indices)
        return selected_indices

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    sol_idx = local_search_swap(sol_idx, DEADLINE - 45.0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Randomized restarts
    attempts = 0
    while time.time() < DEADLINE - 20.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        
        remaining = DEADLINE - time.time()
        ls_budget = min(remaining * 0.4, 25.0)
        cand_idx = local_search_swap(cand_idx, time.time() + ls_budget)
        cand_idx = reduce_solution_fast(cand_idx)
        
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
        
        attempts += 1
        if best_len < 340 and attempts > 8:
            break
            
    return best_sol
