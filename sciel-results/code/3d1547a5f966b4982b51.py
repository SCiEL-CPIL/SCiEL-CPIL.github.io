import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t) = C(7, 3) = 35
    block_indices = np.empty((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
            
    # Precompute frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks_idx = []
    
    # Greedy Phase
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks_idx.append(best_idx)
        uncovered[block_indices[best_idx]] = False
        
    selected = set(selected_blocks_idx)
    
    # Maintain coverage counts
    coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected:
        coverage_count[block_indices[idx]] += 1
        
    # Precompute block_to_triple_sets for fast lookup
    block_to_triple_sets = [set(block_indices[i]) for i in range(num_blocks)]
    
    # Precompute triple_to_blocks
    triple_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for t_idx in block_indices[i]:
            triple_to_blocks[t_idx].append(i)
            
    # Phase 2: Redundancy Removal
    while True:
        if time.time() > deadline:
            break
        removed_any = False
        for idx in list(selected):
            coverage_count[block_indices[idx]] -= 1
            if np.min(coverage_count) > 0:
                selected.remove(idx)
                removed_any = True
            else:
                coverage_count[block_indices[idx]] += 1
                if time.time() > deadline:
                    break
        if not removed_any:
            break
            
    # Phase 3: 2-for-1 Local Search (FIXED)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        sel_list = list(selected)
        n_sel = len(sel_list)
        
        for i in range(n_sel):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_sel):
                if time.time() > deadline:
                    break
                    
                idx_i = sel_list[i]
                idx_j = sel_list[j]
                
                t_set_i = block_to_triple_sets[idx_i]
                t_set_j = block_to_triple_sets[idx_j]
                t_union = t_set_i.union(t_set_j)
                
                critical_triples = []
                for t in t_union:
                    c = coverage_count[t]
                    n_cover_by_pair = (1 if t in t_set_i else 0) + (1 if t in t_set_j else 0)
                    if c - n_cover_by_pair == 0:
                        critical_triples.append(t)
                        
                if not critical_triples:
                    selected.remove(idx_i)
                    selected.remove(idx_j)
                    coverage_count[block_indices[idx_i]] -= 1
                    coverage_count[block_indices[idx_j]] -= 1
                    improved = True
                    break
                else:
                    first_t = critical_triples[0]
                    candidates = triple_to_blocks[first_t]
                    
                    for k in candidates:
                        if k in selected:
                            continue
                        
                        # Check if k covers all critical_triples
                        k_set = block_to_triple_sets[k]
                        if all(t in k_set for t in critical_triples):
                            # Replace idx_i, idx_j with k
                            selected.remove(idx_i)
                            selected.remove(idx_j)
                            selected.add(k)
                            coverage_count[block_indices[idx_i]] -= 1
                            coverage_count[block_indices[idx_j]] -= 1
                            coverage_count[block_indices[k]] += 1
                            improved = True
                            break
                            
            if improved:
                break
                
    # Phase 4: Simulated Annealing
    if time.time() < deadline - 30:  # Reserve 30s for SA
        current_solution = list(selected)
        current_size = len(current_solution)
        best_solution = current_solution[:]
        best_size = current_size
        
        temp = 10.0
        cooling_rate = 0.995
        min_temp = 0.1
        
        while temp > min_temp and time.time() < deadline - 5:
            if not current_solution:
                break
                
            # Random move: remove one block and add another
            idx_to_remove = random.choice(current_solution)
            
            # Check if removing this block would uncover any triples
            temp_coverage = coverage_count.copy()
            temp_coverage[block_indices[idx_to_remove]] -= 1
            
            if np.min(temp_coverage) <= 0:
                # Can't remove, try different move
                temp *= cooling_rate
                continue
            
            # Find uncovered triples after removal
            uncovered_triples = np.where(temp_coverage == 0)[0]
            
            # Find candidate blocks that cover all uncovered triples
            if len(uncovered_triples) > 0:
                first_uncovered = uncovered_triples[0]
                candidates = [c for c in triple_to_blocks[first_uncovered] if c not in current_solution]
                
                if candidates:
                    # Pick a random candidate
                    idx_to_add = random.choice(candidates)
                    
                    # Verify it covers all uncovered triples
                    add_set = block_to_triple_sets[idx_to_add]
                    if all(t in add_set for t in uncovered_triples):
                        # Make the move
                        current_solution.remove(idx_to_remove)
                        current_solution.append(idx_to_add)
                        coverage_count[block_indices[idx_to_remove]] -= 1
                        coverage_count[block_indices[idx_to_add]] += 1
                        
                        new_size = len(current_solution)
                        
                        # Accept if better or with probability based on temperature
                        if new_size < best_size:
                            best_solution = current_solution[:]
                            best_size = new_size
                        elif new_size > current_size:
                            # Reject (we want to minimize)
                            current_solution.remove(idx_to_add)
                            current_solution.append(idx_to_remove)
                            coverage_count[block_indices[idx_to_remove]] += 1
                            coverage_count[block_indices[idx_to_add]] -= 1
                        else:
                            # Same size, accept with probability exp(-delta/temp)
                            delta = 0  # Same size
                            if random.random() < np.exp(-delta / temp):
                                pass  # Keep the move
                            else:
                                # Revert
                                current_solution.remove(idx_to_add)
                                current_solution.append(idx_to_remove)
                                coverage_count[block_indices[idx_to_remove]] += 1
                                coverage_count[block_indices[idx_to_add]] -= 1
            
            temp *= cooling_rate
        
        # Use best solution found
        selected = set(best_solution)
        # Rebuild coverage_count for final solution
        coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            coverage_count[block_indices[idx]] += 1
        
        # Final redundancy removal
        while True:
            if time.time() > deadline:
                break
            removed_any = False
            for idx in list(selected):
                coverage_count[block_indices[idx]] -= 1
                if np.min(coverage_count) > 0:
                    selected.remove(idx)
                    removed_any = True
                else:
                    coverage_count[block_indices[idx]] += 1
            if not removed_any:
                break
                
    # Convert back to list of blocks
    return [list(blocks[idx]) for idx in selected]
