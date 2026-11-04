"""Covering design C(v,k,t) via greedy set cover with NumPy-vectorized gains,
aggressive multi-pass reduction, and multi-restart local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets and the t-subsets each covers
    all_k_subs = list(itertools.combinations(all_points, k))
    n_k = len(all_k_subs)

    # For each k-subset, list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        idxs = [t_sub_to_idx[ts] for ts in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(np.array(idxs, dtype=np.int32))

    # For each t-subset, list of k-subset indices that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, idxs in enumerate(k_sub_t_idxs):
        for t_idx in idxs:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed_order=None):
        """Greedy set cover. Returns list of block indices."""
        covered = np.zeros(n_t, dtype=np.uint8)
        uncovered_count = n_t
        used_blocks = set()
        blocks = []

        if seed_order is None:
            seed_order = list(range(n_t))
            random.shuffle(seed_order)

        # Maintain a list of uncovered t-subset indices for fast access
        uncovered_list = list(range(n_t))

        while uncovered_count > 0 and time_ok():
            # Find the most constrained uncovered t-subset (fewest candidate blocks)
            best_s_idx = -1
            best_num_candidates = float('inf')
            
            # Sample some uncovered t-subsets to find a constrained one
            # For speed, check up to 50 random uncovered ones
            sample_size = min(50, len(uncovered_list))
            sample = random.sample(uncovered_list, sample_size)
            
            for s_idx in sample:
                if covered[s_idx]:
                    continue
                num_cand = len([b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks])
                if num_cand < best_num_candidates:
                    best_num_candidates = num_cand
                    best_s_idx = s_idx
                    if num_cand <= 1:
                        break

            if best_s_idx == -1:
                # Fallback: pick first uncovered
                for s_idx in uncovered_list:
                    if not covered[s_idx]:
                        best_s_idx = s_idx
                        break
                if best_s_idx == -1:
                    break

            # Among all blocks covering best_s_idx, find the one with max gain
            candidates = t_sub_to_k_idxs[best_s_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Compute gain: number of uncovered t-subsets this block covers
                t_idxs = k_sub_t_idxs[b_idx]
                gain = int(np.sum(covered[t_idxs] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # All candidates already used; this shouldn't happen if logic is correct
                # But if it does, we're stuck. Break.
                break

            blocks.append(best_block_idx)
            used_blocks.add(best_block_idx)

            # Update coverage
            t_idxs = k_sub_t_idxs[best_block_idx]
            newly_covered = t_idxs[covered[t_idxs] == 0]
            if len(newly_covered) > 0:
                covered[newly_covered] = 1
                uncovered_count -= len(newly_covered)
                # Remove newly covered from uncovered_list
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]

        return blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks via multi-pass. Returns new list of block indices."""
        if not block_indices:
            return []

        current_blocks = list(block_indices)
        
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        # Multi-pass reduction
        for _pass in range(10):  # up to 10 passes
            if not time_ok():
                break
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if any t-subset in this block has cover_count == 1
                t_idxs = k_sub_t_idxs[b_idx]
                if np.any(cover_count[t_idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    cover_count[t_idxs] -= 1
            current_blocks = new_blocks
            if not changed:
                break

        return current_blocks

    def local_swap_optimize(block_indices):
        """Try to replace blocks with better ones to enable further reduction."""
        if not block_indices or not time_ok():
            return block_indices

        current_blocks = list(block_indices)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        improved = True
        while improved and time_ok():
            improved = False
            # Try swapping each block with a different block that covers its unique t-subsets
            for i, b_idx in enumerate(current_blocks):
                if not time_ok():
                    break
                t_idxs = k_sub_t_idxs[b_idx]
                # Find t-subsets uniquely covered by this block
                unique_mask = cover_count[t_idxs] == 1
                if not np.any(unique_mask):
                    continue
                unique_t_idxs = t_idxs[unique_mask]
                
                # Try to find a replacement block that covers all unique_t_idxs
                # and is not already in current_blocks
                found_replacement = False
                for cand_b in range(n_k):
                    if cand_b in current_blocks:
                        continue
                    cand_t_idxs = k_sub_t_idxs[cand_b]
                    # Check if cand_b covers all unique_t_idxs
                    if len(set(unique_t_idxs.tolist()).intersection(set(cand_t_idxs.tolist()))) == len(unique_t_idxs):
                        # Also check that cand_b doesn't reduce coverage of other t-subsets
                        # that are only covered by current blocks
                        # Simple check: does replacing b_idx with cand_b keep all covered?
                        # We need: for every t-subset covered by b_idx, it's still covered by cand_b or another block
                        # This is complex; let's just do the swap and check
                        old_b = current_blocks[i]
                        # Tentatively swap
                        cover_count[k_sub_t_idxs[old_b]] -= 1
                        cover_count[cand_t_idxs] += 1
                        if np.all(cover_count > 0):
                            current_blocks[i] = cand_b
                            improved = True
                            found_replacement = True
                        else:
                            # Revert
                            cover_count[cand_t_idxs] -= 1
                            cover_count[k_sub_t_idxs[old_b]] += 1
                        if found_replacement:
                            break

            # After swaps, do another reduction pass
            if improved and time_ok():
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in current_blocks:
                    cover_count[k_sub_t_idxs[b_idx]] += 1
                changed = True
                while changed and time_ok():
                    changed = False
                    new_blocks = []
                    for b_idx in current_blocks:
                        t_idxs = k_sub_t_idxs[b_idx]
                        if np.any(cover_count[t_idxs] == 1):
                            new_blocks.append(b_idx)
                        else:
                            changed = True
                            cover_count[t_idxs] -= 1
                    current_blocks = new_blocks

        return current_blocks

    # Main strategy: multiple restarts with different random seeds, keep best
    best_blocks = None
    best_count = float('inf')

    # Run multiple restarts
    num_restarts = 0
    while time_ok() and num_restarts < 50:
        num_restarts += 1
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        
        # Greedy cover
        block_indices = greedy_cover()
        
        if not block_indices:
            continue

        # Reduce
        block_indices = reduce_blocks(block_indices)
        
        # Local swap optimization
        block_indices = local_swap_optimize(block_indices)
        
        # Final reduction after swaps
        block_indices = reduce_blocks(block_indices)
        
        if len(block_indices) < best_count:
            best_count = len(block_indices)
            best_blocks = list(block_indices)
        
        # If we've reached the known lower bound or very close, stop early
        if best_count <= 270:
            break

    # Convert block indices to actual blocks
    if best_blocks is None:
        # Fallback: single greedy run
        random.seed(42)
        block_indices = greedy_cover()
        block_indices = reduce_blocks(block_indices)
        best_blocks = block_indices

    blocks = [list(all_k_subs[b_idx]) for b_idx in best_blocks]
    
    return blocks
# EVOLVE-BLOCK-END
