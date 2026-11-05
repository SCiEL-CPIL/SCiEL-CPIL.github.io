import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    
    # Precompute block coverage indices for fast access
    block_indices = [M[b].indices for b in range(num_k)]
    # Also precompute as sorted tuples for faster set operations
    block_indices_tuple = [tuple(sorted(idx)) for idx in block_indices]
    
    # Precompute for each t-subset, which blocks cover it
    t_subset_to_blocks = [set() for _ in range(num_t)]
    for b_idx in range(num_k):
        for s_idx in block_indices[b_idx]:
            t_subset_to_blocks[s_idx].add(b_idx)

    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_indices[b_idx]
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed, temp=0.1):
        """
        Greedy construction with temperature-based selection.
        """
        random.seed(seed)
        np.random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            mask = (coverage == 0)
            gains = M.dot(mask.astype(np.int8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 1 and temp > 0:
                noise = np.random.random(len(candidates)) * temp
                scores = max_gain + noise
                b_idx = candidates[np.argmax(scores)]
            else:
                b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            idxs = block_indices[b_idx]
            coverage[idxs] += 1
            uncovered_count = np.sum(coverage == 0)
            
        return blocks_idx, coverage, None

    def local_search(blocks_idx, coverage, time_limit):
        """
        Improved local search with faster exposed-subset checking.
        """
        ls_start = time.time()
        improved = True
        blocks_set = set(blocks_idx)
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            order = list(range(len(blocks_idx)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2.0:
                    return blocks_idx, coverage, False
                
                b_remove = blocks_idx[i]
                idxs_remove = block_indices[b_remove]
                
                exposed_mask = (coverage == 1)
                exposed = idxs_remove[exposed_mask[idxs_remove]]
                
                if len(exposed) == 0:
                    coverage[idxs_remove] -= 1
                    blocks_idx.pop(i)
                    blocks_set.discard(b_remove)
                    improved = True
                    blocks_idx = reduce_blocks(blocks_idx, coverage)
                    blocks_set = set(blocks_idx)
                    break
                else:
                    first_exposed = exposed[0]
                    candidate_blocks = t_subset_to_blocks[first_exposed]
                    
                    best_swap = -1
                    best_gain = -1
                    
                    # Precompute exposed as a set for fast membership testing
                    exposed_set = set(exposed)
                    
                    for b_candidate in candidate_blocks:
                        if b_candidate == b_remove or b_candidate in blocks_set:
                            continue
                        
                        # Fast check: does b_candidate cover all exposed?
                        cand_idxs = block_indices[b_candidate]
                        # Use numpy intersection check
                        # Count how many of exposed are in cand_idxs
                        if len(cand_idxs) == 0:
                            continue
                        # Quick filter: if cand_idxs size < exposed size, skip
                        if len(cand_idxs) < len(exposed):
                            continue
                        
                        # Check if all exposed are covered by candidate
                        # Use set intersection for speed
                        cand_set = set(cand_idxs)
                        if not exposed_set.issubset(cand_set):
                            continue
                        
                        # Calculate gain: new subsets covered (coverage == 0)
                        new_covers = np.sum(coverage[cand_idxs] == 0)
                        
                        if new_covers > best_gain:
                            best_gain = new_covers
                            best_swap = b_candidate
                    
                    if best_swap >= 0:
                        coverage[idxs_remove] -= 1
                        blocks_idx.pop(i)
                        blocks_set.discard(b_remove)
                        
                        idxs_add = block_indices[best_swap]
                        coverage[idxs_add] += 1
                        blocks_idx.append(best_swap)
                        blocks_set.add(best_swap)
                        
                        improved = True
                        blocks_idx = reduce_blocks(blocks_idx, coverage)
                        blocks_set = set(blocks_idx)
                        break
        
        return blocks_idx, coverage, improved

    def polish(blocks_idx, coverage, time_limit):
        """
        High-precision polish: try multiple passes of local search with different orderings,
        and attempt 2-exchanges (remove two, add one or two).
        """
        polish_start = time.time()
        blocks_set = set(blocks_idx)
        
        # Multiple passes of local search with shuffled orderings
        while (time.time() - polish_start) < time_limit * 0.6:
            old_len = len(blocks_idx)
            blocks_idx, coverage, improved = local_search(blocks_idx, coverage, time_limit * 0.15)
            if not improved or len(blocks_idx) >= old_len:
                break
        
        # Try 2-exchange: remove two blocks, try to replace with one or two new blocks
        while (time.time() - polish_start) < time_limit:
            improved = False
            n = len(blocks_idx)
            if n < 2:
                break
            
            # Try removing pairs of blocks
            pair_order = list(itertools.combinations(range(n), 2))
            random.shuffle(pair_order)
            
            for (i, j) in pair_order:
                if time.time() > deadline - 2.0 or (time.time() - polish_start) > time_limit:
                    return blocks_idx, coverage, False
                
                if i >= len(blocks_idx) or j >= len(blocks_idx):
                    continue
                if i == j:
                    continue
                
                b1 = blocks_idx[i]
                b2 = blocks_idx[j]
                
                # Find exposed subsets when removing both
                idxs1 = block_indices[b1]
                idxs2 = block_indices[b2]
                
                # Temporarily remove both
                coverage[idxs1] -= 1
                coverage[idxs2] -= 1
                
                exposed_mask = (coverage == 0)
                exposed = np.where(exposed_mask)[0]
                
                if len(exposed) == 0:
                    # Can remove both without adding
                    blocks_idx.pop(max(i, j))
                    blocks_idx.pop(min(i, j))
                    blocks_set.discard(b1)
                    blocks_set.discard(b2)
                    improved = True
                    blocks_idx = reduce_blocks(blocks_idx, coverage)
                    blocks_set = set(blocks_idx)
                    break
                else:
                    # Try to find one block that covers all exposed
                    if len(exposed) > 0:
                        first_exp = exposed[0]
                        candidates = t_subset_to_blocks[first_exp]
                        
                        best_single = -1
                        best_single_gain = -1
                        
                        exposed_set = set(exposed)
                        for b_cand in candidates:
                            if b_cand in blocks_set or b_cand == b1 or b_cand == b2:
                                continue
                            cand_idxs = block_indices[b_cand]
                            if len(cand_idxs) < len(exposed):
                                continue
                            cand_set = set(cand_idxs)
                            if not exposed_set.issubset(cand_set):
                                continue
                            gain = np.sum(coverage[cand_idxs] == 0)
                            if gain > best_single_gain:
                                best_single_gain = gain
                                best_single = b_cand
                        
                        if best_single >= 0:
                            # Add the single block
                            idxs_add = block_indices[best_single]
                            coverage[idxs_add] += 1
                            blocks_idx.append(best_single)
                            blocks_set.add(best_single)
                            improved = True
                            blocks_idx = reduce_blocks(blocks_idx, coverage)
                            blocks_set = set(blocks_idx)
                            break
                        else:
                            # Try two blocks to cover exposed
                            # This is more expensive; only try if exposed is small
                            if len(exposed) <= 30:
                                best_pair = None
                                best_pair_gain = -1
                                
                                # Try pairs of candidates
                                cand_list = list(candidates)
                                if len(cand_list) > 50:
                                    # Too many, skip
                                    coverage[idxs1] += 1
                                    coverage[idxs2] += 1
                                    continue
                                
                                for a in range(len(cand_list)):
                                    if time.time() > deadline - 2.0:
                                        break
                                    b_a = cand_list[a]
                                    if b_a in blocks_set or b_a == b1 or b_a == b2:
                                        continue
                                    idxs_a = block_indices[b_a]
                                    set_a = set(idxs_a)
                                    
                                    for b_b in cand_list[a+1:]:
                                        if b_b in blocks_set or b_b == b1 or b_b == b2:
                                            continue
                                        idxs_b = block_indices[b_b]
                                        set_b = set(idxs_b)
                                        
                                        # Check if union covers all exposed
                                        combined = set_a | set_b
                                        if not exposed_set.issubset(combined):
                                            continue
                                        
                                        # Calculate gain
                                        # Subsets covered by either that are currently uncovered
                                        # Use numpy: create a boolean array
                                        all_idxs = np.union1d(idxs_a, idxs_b)
                                        gain = np.sum(coverage[all_idxs] == 0)
                                        
                                        if gain > best_pair_gain:
                                            best_pair_gain = gain
                                            best_pair = (b_a, b_b)
                                
                                if best_pair is not None:
                                    b_a, b_b = best_pair
                                    idxs_a = block_indices[b_a]
                                    idxs_b = block_indices[b_b]
                                    coverage[idxs_a] += 1
                                    coverage[idxs_b] += 1
                                    blocks_idx.append(b_a)
                                    blocks_idx.append(b_b)
                                    blocks_set.add(b_a)
                                    blocks_set.add(b_b)
                                    improved = True
                                    blocks_idx = reduce_blocks(blocks_idx, coverage)
                                    blocks_set = set(blocks_idx)
                                    break
                            else:
                                coverage[idxs1] += 1
                                coverage[idxs2] += 1
                                continue
                    
                    # Restore coverage if we didn't find a replacement
                    coverage[idxs1] += 1
                    coverage[idxs2] += 1
        
        return blocks_idx, coverage, improved

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 15.0:
        num_restarts += 1
        temp = 0.05 + 0.1 * (num_restarts % 5)
        res = greedy_construct(num_restarts, temp)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            time_left = deadline - time.time()
            # Allocate time for local search and polish
            ls_time = min(10.0, time_left * 0.25)
            polish_time = min(20.0, time_left * 0.4)
            
            if ls_time > 1.0:
                blocks_idx, coverage, _ = local_search(blocks_idx, coverage, ls_time)
                blocks_idx = reduce_blocks(blocks_idx, coverage)
                current_score = len(blocks_idx)
                if current_score < best_score:
                    best_score = current_score
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            if polish_time > 1.0 and (deadline - time.time()) > 5.0:
                blocks_idx, coverage, _ = polish(blocks_idx, coverage, polish_time)
                blocks_idx = reduce_blocks(blocks_idx, coverage)
                current_score = len(blocks_idx)
                if current_score < best_score:
                    best_score = current_score
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
                    
    # Final polish on the best solution found
    if best_blocks is not None and (deadline - time.time()) > 5.0:
        # Reconstruct coverage for best_blocks
        blocks_idx_final = []
        for block in best_blocks:
            # Find the index of this block in all_k_blocks
            # Use a dictionary for fast lookup
            pass
        
        # Simpler: just run polish on the current best
        # Rebuild coverage from best_blocks
        coverage_final = np.zeros(num_t, dtype=np.int32)
        blocks_idx_final = []
        for block in best_blocks:
            # Find block index
            b_tuple = tuple(block)
            # Search for it (could precompute, but this is a one-time cost)
            for idx, bk in enumerate(all_k_blocks):
                if tuple(bk) == b_tuple:
                    blocks_idx_final.append(idx)
                    coverage_final[block_indices[idx]] += 1
                    break
        
        if blocks_idx_final:
            final_polish_time = min(15.0, deadline - time.time() - 2.0)
            if final_polish_time > 1.0:
                blocks_idx_final, coverage_final, _ = polish(blocks_idx_final, coverage_final, final_polish_time)
                blocks_idx_final = reduce_blocks(blocks_idx_final, coverage_final)
                if len(blocks_idx_final) < best_score:
                    best_score = len(blocks_idx_final)
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx_final]
                    
    return best_blocks
