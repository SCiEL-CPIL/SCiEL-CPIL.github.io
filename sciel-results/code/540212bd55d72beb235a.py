import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 130.0  # Safety margin from 150s

    # Pre-compute all t-subsets and index them
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)

    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)

    # Build subset_to_blocks mapping
    subset_to_blocks = {i: [] for i in range(n_t_subs)}
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    # Greedy Construction
    uncovered = [1] * n_t_subs
    cover_count = [0] * n_t_subs
    blocks = []
    
    # Track which blocks are still "active" (have uncovered subsets)
    uncovered_count_per_block = [len(subs) for subs in block_t_subs]
    total_uncovered = n_t_subs
    
    # For efficiency, maintain a list of active block indices
    active_blocks = list(range(n_blocks))
    
    while total_uncovered > 0 and (time.time() - start_time) < deadline * 0.4:
        # Find the block with maximum uncovered subsets
        max_count = -1
        best_b_idx = -1
        
        # Sample a subset of active blocks to speed up search
        # Use random sampling if active list is large
        if len(active_blocks) > 5000:
            sample_size = min(5000, len(active_blocks))
            sample = random.sample(active_blocks, sample_size)
        else:
            sample = active_blocks
            
        for i in sample:
            if uncovered_count_per_block[i] > max_count:
                max_count = uncovered_count_per_block[i]
                best_b_idx = i
                if max_count == len(block_t_subs[i]):
                    break
        
        if best_b_idx == -1 or max_count == 0:
            # Fallback: scan all active
            for i in active_blocks:
                if uncovered_count_per_block[i] > max_count:
                    max_count = uncovered_count_per_block[i]
                    best_b_idx = i
                    if max_count == len(block_t_subs[i]):
                        break
        
        if best_b_idx == -1 or max_count == 0:
            break
            
        blocks.append(all_blocks[best_b_idx])
        
        # Update counts
        for s in block_t_subs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                cover_count[s] += 1
                # Update all blocks containing s
                for b_idx in subset_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1
                    if uncovered_count_per_block[b_idx] == 0:
                        # Remove from active list (lazy deletion)
                        pass

    # Local Search Phase: Aggressive Pruning
    # Convert blocks to indices for faster access
    block_set = set(blocks)
    block_indices = []
    for b in blocks:
        idx = all_blocks.index(b)
        block_indices.append(idx)
    
    # Recompute cover_count from scratch
    cover_count = [0] * n_t_subs
    for b_idx in block_indices:
        for s in block_t_subs[b_idx]:
            cover_count[s] += 1

    # Phase 1: Remove redundant blocks iteratively
    improved = True
    while improved and (time.time() - start_time) < deadline * 0.7:
        improved = False
        # Try removing each block
        for i in range(len(block_indices) - 1, -1, -1):
            b_idx = block_indices[i]
            # Check if redundant: all its subsets are covered by other blocks
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break  # Restart as indices changed

    # Phase 2: Swap-based improvement
    # Try to replace blocks with better ones that enable further pruning
    improved = True
    while improved and (time.time() - start_time) < deadline * 0.9:
        improved = False
        for i in range(len(block_indices)):
            b_idx = block_indices[i]
            # Find critical subsets (covered only by this block)
            critical_subsets = [s for s in block_t_subs[b_idx] if cover_count[s] == 1]
            if not critical_subsets:
                # This block is redundant, should have been removed
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break

            # Find candidate replacement blocks that cover all critical subsets
            candidates = set(subset_to_blocks[critical_subsets[0]])
            for s in critical_subsets[1:]:
                candidates.intersection_update(subset_to_blocks[s])
                if not candidates:
                    break

            if not candidates:
                continue

            # Score candidates by how many of their subsets are already well-covered
            best_gain = 0
            best_new_idx = -1
            for new_idx in candidates:
                if new_idx == b_idx or new_idx in block_indices:
                    continue

                # Calculate potential gain:
                # - Loss: subsets in b_idx that become uncovered (cover_count==1 and not in new_idx)
                # - Gain: subsets in new_idx that were uncovered (cover_count==0)
                new_set = set(block_t_subs[new_idx])
                loss = 0
                for s in block_t_subs[b_idx]:
                    if cover_count[s] == 1 and s not in new_set:
                        loss += 1

                newly_covered = 0
                for s in block_t_subs[new_idx]:
                    if cover_count[s] == 0:
                        newly_covered += 1

                gain = newly_covered - loss
                if gain > best_gain:
                    best_gain = gain
                    best_new_idx = new_idx

            if best_gain > 0 and best_new_idx != -1:
                # Perform swap
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                for s in block_t_subs[best_new_idx]:
                    cover_count[s] += 1
                block_indices[i] = best_new_idx
                improved = True
                break

    # Final pruning pass
    improved = True
    while improved and (time.time() - start_time) < deadline * 0.95:
        improved = False
        for i in range(len(block_indices) - 1, -1, -1):
            b_idx = block_indices[i]
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break

    final_blocks = [all_blocks[i] for i in block_indices]
    return final_blocks
