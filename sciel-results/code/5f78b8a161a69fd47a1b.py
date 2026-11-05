import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim for 140s limit.
    deadline = start_time + 140.0
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each block covers
    # block_subsets[i] is a list of subset indices
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute which blocks cover each subset
    # blocks_covering_subset[j] is a list of block indices
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # Build sparse incidence matrix M: M[b, s] = 1 if block b covers t-subset s
    # This allows for fast vectorized gain calculation
    rows = []
    cols = []
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N_blocks, N_subsets))
    
    def time_exceeded():
        return time.time() > deadline - 2.0

    # 2. Greedy Construction (Vectorized)
    # Uses M to compute gains for all blocks at once.
    def run_greedy(randomize_ties=False, alpha=1.0):
        if time_exceeded():
            return None
            
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        selected_blocks = []
        uncovered_mask = np.ones(N_subsets, dtype=np.int8)
        
        while np.sum(uncovered_mask) > 0:
            if time_exceeded():
                return None
                
            # Compute gains: number of uncovered subsets covered by each block
            # gains[b] = sum(M[b, :] * uncovered_mask)
            gains = M.dot(uncovered_mask)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if problem is solvable
                
            # Probabilistic selection (GRASP-like)
            if randomize_ties:
                # Find candidates with high gain
                # Threshold: max_gain * alpha? Or just top few?
                # Let's use a soft-max style or simple threshold
                threshold = max_gain * 0.8
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gains == max_gain)[0]
                
                # Weight by gain squared to prefer higher gains more strongly
                weights = gains[candidates] ** 2
                weights /= np.sum(weights)
                b_idx = np.random.choice(candidates, p=weights)
            else:
                b_idx = np.argmax(gains)
                
            selected_blocks.append(b_idx)
            
            # Update cover_count and uncovered_mask
            idxs = M[b_idx].indices
            old_counts = cover_count[idxs]
            cover_count[idxs] += 1
            
            # Update mask: if old count was 0, it becomes covered (mask becomes 0)
            newly_covered = (old_counts == 0)
            if np.any(newly_covered):
                uncovered_mask[idxs[newly_covered]] = 0
            # Note: if old count > 0, mask is already 0, so no change needed there.
            # If old count was 0, mask was 1, now becomes 0.
            
        return selected_blocks

    # 3. Local Search: Reduction and Swap
    def local_search(selected_blocks):
        if time_exceeded():
            return selected_blocks
            
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            idxs = M[b_idx].indices
            cover_count[idxs] += 1
                
        selected_set = set(selected_blocks)
        
        # Phase 1: Reduction
        changed = True
        while changed and not time_exceeded():
            changed = False
            new_blocks = []
            # Iterate in random order for better reduction results?
            # Or just standard. Let's stick to standard for determinism in reduction,
            # but maybe shuffle in swap phase.
            for b_idx in selected_blocks:
                idxs = M[b_idx].indices
                # If all subsets covered by this block have count > 1, it's redundant
                if np.all(cover_count[idxs] > 1):
                    cover_count[idxs] -= 1
                    selected_set.remove(b_idx)
                    changed = True
                else:
                    new_blocks.append(b_idx)
            selected_blocks = new_blocks
            
        # Phase 2: Swap Search
        # Try to swap blocks to enable further reductions
        for _ in range(20): # Limit swap iterations
            if time_exceeded():
                break
                
            improved = False
            current_list = list(selected_blocks)
            random.shuffle(current_list)
            
            for b_idx in current_list:
                if b_idx not in selected_set:
                    continue
                             
                # Find subsets that become uncovered if we remove b_idx
                idxs = M[b_idx].indices
                uncovered_subs = []
                for s in idxs:
                    if cover_count[s] == 1:
                        uncovered_subs.append(s)
                        
                if not uncovered_subs:
                    # Redundant, remove it (should have been caught in reduction, but just in case)
                    cover_count[idxs] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.remove(b_idx)
                    improved = True
                    break
                    
                # Find a block j that covers all uncovered_subs
                # Intersection of blocks_covering_subset[s] for s in uncovered_subs
                if uncovered_subs:
                    candidate_blocks = set(blocks_covering_subset[uncovered_subs[0]])
                    for s in uncovered_subs[1:]:
                        candidate_blocks &= set(blocks_covering_subset[s])
                    
                    # Filter out blocks already in selected_set
                    candidate_blocks -= selected_set
                    
                    if candidate_blocks:
                        # Pick a candidate
                        # Heuristic: Pick one that covers the most *other* subsets?
                        # Let's pick randomly for diversity
                        j_idx = random.choice(list(candidate_blocks))
                        
                        # Perform swap
                        # Remove b_idx
                        cover_count[idxs] -= 1
                        selected_blocks.remove(b_idx)
                        selected_set.remove(b_idx)
                        
                        # Add j_idx
                        j_idxs = M[j_idx].indices
                        cover_count[j_idxs] += 1
                        selected_blocks.append(j_idx)
                        selected_set.add(j_idx)
                        
                        improved = True
                        break # Restart reduction
                        
            if not improved:
                break
                
            # Re-run reduction
            changed = True
            while changed and not time_exceeded():
                changed = False
                new_blocks = []
                for b_idx in selected_blocks:
                    idxs = M[b_idx].indices
                    if np.all(cover_count[idxs] > 1):
                        cover_count[idxs] -= 1
                        selected_set.remove(b_idx)
                        changed = True
                    else:
                        new_blocks.append(b_idx)
                selected_blocks = new_blocks
                
        return selected_blocks

    # 4. Main Execution
    best_blocks = None
    best_len = float('inf')
    
    # Run deterministic greedy first
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search(res)
        if res:
            best_blocks = res
            best_len = len(res)
            
    # Run randomized restarts (GRASP)
    # We have ~120s left (140 - 20 for precomp/deterministic)
    # Each greedy is fast with vectorization. Local search is the bottleneck.
    # Let's run as many as we can.
    
    for i in range(100): # Cap attempts
        if time.time() > deadline - 5.0:
            break
            
        # Vary alpha slightly? Or just randomize.
        res = run_greedy(randomize_ties=True)
        if res:
            res = local_search(res)
            if res and len(res) < best_len:
                best_len = len(res)
                best_blocks = res
                # If we hit a very good score, we might want to polish more?
                # For now, just keep searching.

    if best_blocks is None:
        return []

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
