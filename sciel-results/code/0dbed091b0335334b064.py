import itertools
import time
import numpy as np
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    # Helper: Greedy construction
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []
        
        uncovered_count_per_block = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)
        
        while total_uncovered > 0:
            if time.time() > deadline:
                break
            best_b_idx = np.argmax(uncovered_count_per_block)
            if uncovered_count_per_block[best_b_idx] == 0:
                break
            
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
                        
        return solution_list, solution_set

    # Helper: Local Search (Deletion, Swap, and 2-for-1)
    def local_search(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        improved = True
        max_passes = 20  # Prevent infinite loops
        pass_count = 0
        
        while improved and time.time() < deadline - 1.0 and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # Pass 1: Deletion
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break
                    
            # Pass 2: Swap
            if not improved:
                for i in range(len(solution_list)):
                    b_idx = solution_list[i]
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    if len(critical_subsets) == 1:
                        candidates = tsub_to_blocks[critical_subsets[0]]
                    else:
                        min_idx = min(range(len(critical_subsets)), key=lambda j: len(tsub_to_blocks[critical_subsets[j]]))
                        candidates = set(tsub_to_blocks[critical_subsets[min_idx]])
                        for s in critical_subsets:
                            if s == critical_subsets[min_idx]:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                    
                    if candidates:
                        best_candidate = -1
                        for c in candidates:
                            if c not in solution_set:
                                best_candidate = c
                                break
                                
                        if best_candidate != -1:
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            solution_list[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break
                            
            # Pass 3: 2-for-1 replacement
            if not improved:
                C_k_t = math.comb(k, t)
                for i in range(len(solution_list)):
                    b1 = solution_list[i]
                    crit1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    if not crit1:
                        continue
                    for j in range(i+1, len(solution_list)):
                        b2 = solution_list[j]
                        crit2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                        if not crit2:
                            continue
                        all_crit = set(crit1) | set(crit2)
                        if len(all_crit) > C_k_t:
                            continue
                        min_crit = min(all_crit, key=lambda s: len(tsub_to_blocks[s]))
                        candidates = set(tsub_to_blocks[min_crit])
                        for s in all_crit:
                            if s == min_crit:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                        if candidates:
                            best_cand = -1
                            for c in candidates:
                                if c not in solution_set:
                                    best_cand = c
                                    break
                            if best_cand != -1:
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[best_cand]:
                                    cover_count[s] += 1
                                solution_list.pop(j)
                                solution_list.pop(i)
                                solution_list.append(best_cand)
                                solution_set.remove(b1)
                                solution_set.remove(b2)
                                solution_set.add(best_cand)
                                improved = True
                                break
                    if improved:
                        break
                            
        return solution_list, solution_set

    # Helper: Perturbation and Repair with criticality awareness
    def perturb_and_repair(solution_list, solution_set):
        # Compute cover_count for current solution
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        # Score each block by its "criticality" (number of subsets it uniquely covers)
        criticality = []
        for i, b_idx in enumerate(solution_list):
            crit_count = 0
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    crit_count += 1
            criticality.append(crit_count)
            
        # Remove blocks with lowest criticality first
        n_remove = random.randint(2, 5)
        if len(solution_list) <= n_remove:
            return greedy_init()
            
        # Sort by criticality (ascending) and remove from least critical
        indices_by_crit = sorted(range(len(solution_list)), key=lambda i: criticality[i])
        indices_to_remove = indices_by_crit[:n_remove]
        indices_to_remove_set = set(indices_to_remove)
        
        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)
            
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove_set]
        
        # Calculate uncovered subsets
        cover_count_new = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count_new[s] += 1
                
        uncovered_mask = (cover_count_new == 0)
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set
            
        # Greedy Repair - pick uncovered subset with fewest candidates, then best covering block
        to_cover = set(uncovered_indices.tolist())
        
        while to_cover:
            if time.time() > deadline - 2.0:
                break
            
            # Pick an uncovered subset with fewest candidate blocks
            min_candidates = None
            best_uncovered = None
            for s in to_cover:
                cands = [b for b in tsub_to_blocks[s] if b not in new_solution_set]
                if not cands:
                    # No block can cover this subset - should not happen
                    break
                if min_candidates is None or len(cands) < len(min_candidates):
                    min_candidates = cands
                    best_uncovered = s
            
            if min_candidates is None or not min_candidates:
                break
            
            # Find the block that covers the most uncovered subsets
            best_b_idx = min_candidates[0]
            max_cov = 0
            for b_idx in min_candidates:
                cnt = 0
                for s in block_to_tsubs[b_idx]:
                    if s in to_cover:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b_idx = b_idx
                if cnt == len(to_cover):
                    break
            
            new_solution_set.add(best_b_idx)
            new_solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if s in to_cover:
                    to_cover.discard(s)
                    
        return new_solution_list, new_solution_set

    # Precompute C(k,t) for the 2-for-1 check
    C_k_t = math.comb(k, t)
    
    # Main ILS Loop
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set
    
    random.seed(42)
    
    # Track consecutive non-improving iterations for adaptive perturbation
    stagnant = 0
    
    while time.time() < deadline - 5.0:
        # Adaptive perturbation strength based on stagnation
        if stagnant == 0:
            n_remove_range = (2, 4)
        elif stagnant < 3:
            n_remove_range = (3, 6)
        else:
            n_remove_range = (4, 8)
        
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)
        
        opt_list, opt_set = local_search(pert_list, pert_set)
        
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
            stagnant = 0
        else:
            stagnant += 1
            # Accept if it improves over current or with small probability
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set
            elif random.random() < 0.20:  # Slightly higher acceptance rate for diversity
                current_solution_list = opt_list
                current_solution_set = opt_set
            # If stagnant for too long, restart from best
            if stagnant > 8:  # Shorter stagnation threshold to explore more
                current_solution_list = list(best_solution_list)
                current_solution_set = set(best_solution_set)
                stagnant = 0
            
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
