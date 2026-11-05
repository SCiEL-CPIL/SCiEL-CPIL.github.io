import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Precompute candidate lists as numpy arrays for faster access
    t_subset_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in t_subset_to_blocks]
    
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    while True:
        if time.time() > deadline - 25.0:
            break
            
        random.seed(restart * 7919 + 13)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_list = list(range(num_t_subsets))
        
        # Greedy Phase
        while uncovered_list:
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive K
            frac_uncovered = len(uncovered_list) / num_t_subsets
            if frac_uncovered > 0.5:
                K = min(50, max(10, int(50 * frac_uncovered)))
            elif frac_uncovered > 0.1:
                K = 20
            else:
                K = 10
            
            # Pick K random uncovered subsets
            sampled = random.sample(uncovered_list, min(K, len(uncovered_list)))
            
            # Collect candidate blocks
            candidate_set = set()
            for u_idx in sampled:
                for b_idx in t_subset_to_blocks_arr[u_idx]:
                    candidate_set.add(b_idx)
            
            if not candidate_set:
                break
            
            cand_list = list(candidate_set)
            
            # Vectorized gain calculation
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            
            # Gain = number of currently uncovered t-subsets covered by this block
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            
            # Update uncovered list
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
            
            if not uncovered_list:
                break
        
        # If we broke out of greedy due to time, this covering might be invalid
        if uncovered_list:
            # Try to complete quickly with simple greedy
            while uncovered_list and time.time() < deadline - 2.0:
                u_idx = random.choice(uncovered_list)
                candidates = t_subset_to_blocks_arr[u_idx]
                # Pick candidate with max gain
                best_b = -1
                best_gain = -1
                for b_idx in candidates:
                    gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1: break
                block_indices.append(best_b)
                idxs = block_t_indices[best_b]
                was_uncovered = coverage[idxs] == 0
                coverage[idxs] += 1
                newly_covered = idxs[was_uncovered]
                if len(newly_covered) > 0:
                    newly_covered_set = set(newly_covered.tolist())
                    uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
        
        if uncovered_list:
            continue # Invalid covering, restart
        
        # Reduction Phase: remove non-essential blocks
        current_indices = block_indices[:]
        random.shuffle(current_indices)
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
            
            if time.time() > deadline - 10.0:
                break
        
        # Local Search Phase: swap-based improvement
        # Strategy: For each block, try to find a replacement that covers its unique subsets
        # and allows removal of another block.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Randomize order of blocks to try
            block_order = current_indices[:]
            random.shuffle(block_order)
            
            for i in range(len(current_indices)):
                if time.time() > deadline - 5.0: break
                b_idx = current_indices[i]
                t_idxs = block_t_indices[b_idx]
                
                # Find subsets that would become uncovered if we remove this block
                at_risk_mask = coverage[t_idxs] == 1
                at_risk = t_idxs[at_risk_mask]
                
                if len(at_risk) == 0:
                    # Can remove entirely
                    coverage[t_idxs] -= 1
                    current_indices.pop(i)
                    improved = True
                    break
                
                # If at_risk is not empty, try to find a replacement
                # Only try if at_risk is small enough to search
                if len(at_risk) > 5:
                    continue
                
                # Find candidate replacements that cover all at_risk
                candidates = set(t_subset_to_blocks_arr[at_risk[0]].tolist())
                for r in at_risk[1:]:
                    candidates &= set(t_subset_to_blocks_arr[r].tolist())
                
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                # Evaluate candidates: pick one that maximizes potential for removal
                # Heuristic: prefer blocks that cover many subsets with high coverage
                # (i.e., subsets that are already well-covered, so adding more coverage
                #  doesn't hurt but might make other blocks redundant)
                
                best_repl = -1
                best_score_repl = -1
                
                for rep_b in candidates:
                    rep_t_idxs = block_t_indices[rep_b]
                    
                    # Score: number of subsets in rep_b that have coverage >= 2
                    # (these are "safe" to add extra coverage to)
                    score = np.sum(coverage[rep_t_idxs] >= 2)
                    
                    if score > best_score_repl:
                        best_score_repl = score
                        best_repl = rep_b
                
                if best_repl != -1:
                    # Perform swap
                    coverage[t_idxs] -= 1
                    new_t_idxs = block_t_indices[best_repl]
                    coverage[new_t_idxs] += 1
                    current_indices[i] = best_repl
                    improved = True
                    break
        
        # Final check for redundancy after local search
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
        
        # Update best
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
        if restart > 150:
            break
    
    if best_block_indices is None:
        # Fallback: run one more greedy to ensure we have a valid covering
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_list = list(range(num_t_subsets))
        
        while uncovered_list:
            u_idx = random.choice(uncovered_list)
            candidates = t_subset_to_blocks_arr[u_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            block_indices.append(best_block_idx)
            idxs = block_t_indices[best_block_idx]
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
        
        best_block_indices = block_indices
    
    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_block_indices]
