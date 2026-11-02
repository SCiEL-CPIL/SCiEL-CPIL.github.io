"""Combined approach: Greedy + Column Generation + Simulated Annealing with incremental coverage."""
import time
import random
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute C(k, t)
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover_matrix: shape (n_k, c_k_t), each entry is an index into all_t_subsets
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]

    # Build t_to_blocks: for each t-subset, list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for ts_idx in cover_matrix[i]:
            t_to_blocks[ts_idx].append(i)

    # 1. Greedy Initialization
    best_blocks = []
    uncovered = np.ones(n_t, dtype=bool)
    
    # Standard Greedy
    while np.any(uncovered):
        gains = np.sum(uncovered[cover_matrix], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # Random tie-breaking
        idx = int(candidates[random.randint(0, len(candidates) - 1)])
        best_blocks.append(idx)
        uncovered[cover_matrix[idx]] = False
        
    # 2. Column Generation Phase: Use LP dual to find better blocks
    # Start with current solution, iteratively add blocks that minimize weighted uncovered triples
    current_block_set = set(best_blocks)
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_block_set:
        cover_counts[cover_matrix[idx]] += 1
    
    # Column generation: try to find blocks that cover many currently under-covered triples
    cg_iterations = 0
    max_cg_iterations = 50
    while cg_iterations < max_cg_iterations and time.time() < deadline - 60:
        cg_iterations += 1
        
        # Find the most under-covered triples
        min_cover = np.min(cover_counts)
        if min_cover >= 1:
            break  # All triples are covered at least once
        
        # Focus on triples with minimum coverage
        under_covered = np.where(cover_counts <= min_cover)[0]
        
        # Try to find a block that covers many under-covered triples
        best_new_block = -1
        best_gain = 0
        
        # Sample candidate blocks from t_to_blocks of under-covered triples
        for tc_idx in under_covered[:10]:  # Limit to first 10 for speed
            candidates = t_to_blocks[tc_idx]
            for cand in candidates:
                if cand in current_block_set:
                    continue
                # Count how many under-covered triples this block covers
                block_cover = cover_matrix[cand]
                gain = np.sum(cover_counts[block_cover] <= min_cover)
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = cand
        
        if best_new_block == -1 or best_gain <= 0:
            break
        
        # Add the new block
        current_block_set.add(best_new_block)
        cover_counts[cover_matrix[best_new_block]] += 1
    
    # Convert back to list
    current_blocks = list(current_block_set)
    best_count = len(current_blocks)
    
    # 3. Simulated Annealing with improved repair
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    def greedy_repair(U, current_block_set, max_blocks=2):
        """
        Try to cover U with at most max_blocks blocks not in current_block_set.
        Returns a list of block indices or None.
        """
        if len(U) == 0:
            return []
        
        U_remaining = set(U)
        repair_blocks = []
        
        for _ in range(max_blocks):
            if not U_remaining:
                break
            
            # Find block covering most of U_remaining
            best_block = -1
            best_covered = 0
            
            target_t = next(iter(U_remaining))
            candidates = t_to_blocks[target_t]
            
            for cand_idx in candidates:
                if cand_idx in current_block_set:
                    continue
                # Check how many of U_remaining this block covers
                covered = np.isin(U_remaining, cover_matrix[cand_idx])
                count = np.sum(covered)
                if count > best_covered:
                    best_covered = count
                    best_block = cand_idx
                    if count == len(U_remaining):
                        break
            
            if best_block == -1:
                return None
            
            repair_blocks.append(best_block)
            current_block_set.add(best_block)
            
            # Update U_remaining
            to_remove = [u for u in U_remaining if u in cover_matrix[best_block]]
            for u in to_remove:
                U_remaining.discard(u)
        
        if U_remaining:
            return None
        return repair_blocks

    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        if not current_blocks:
            break
            
        remove_idx_pos = rng.randint(0, len(current_blocks) - 1)
        block_to_remove = current_blocks[remove_idx_pos]
        
        # Tentatively remove
        current_blocks.pop(remove_idx_pos)
        current_block_set = set(current_blocks)
        
        cover_counts[cover_matrix[block_to_remove]] -= 1
        
        # Find uncovered
        uncovered_mask = cover_counts == 0
        if not np.any(uncovered_mask):
            # Block was redundant
            best_count = len(current_blocks)
            T *= cooling
            continue
        
        U = np.where(uncovered_mask)[0]
        
        # Try to repair with 1 block first
        repair = greedy_repair(U, current_block_set, max_blocks=1)
        
        if repair is not None:
            current_blocks.extend(repair)
            for b in repair:
                cover_counts[cover_matrix[b]] += 1
            T *= cooling
        else:
            # Try to repair with 2 blocks
            repair = greedy_repair(U, current_block_set, max_blocks=2)
            
            if repair is not None:
                delta = 1
                if rng.random() < np.exp(-delta / T):
                    current_blocks.extend(repair)
                    for b in repair:
                        cover_counts[cover_matrix[b]] += 1
                    T *= cooling
                else:
                    current_blocks.append(block_to_remove)
                    cover_counts[cover_matrix[block_to_remove]] += 1
                    T *= cooling
            else:
                current_blocks.append(block_to_remove)
                cover_counts[cover_matrix[block_to_remove]] += 1
                T *= cooling

    # Final cleanup: Remove redundant blocks (two passes)
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_blocks:
        cover_counts[cover_matrix[idx]] += 1
        
    final_blocks = []
    for idx in current_blocks:
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            cover_counts[cover_matrix[idx]] -= 1
        else:
            final_blocks.append(idx)
            
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in final_blocks:
        cover_counts[cover_matrix[idx]] += 1
        
    truly_final_blocks = []
    for idx in final_blocks:
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            cover_counts[cover_matrix[idx]] -= 1
        else:
            truly_final_blocks.append(idx)

    return [list(all_k_subsets[idx]) for idx in truly_final_blocks]
