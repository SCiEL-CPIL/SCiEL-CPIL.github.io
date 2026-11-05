import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Fast greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k):
            if not uncovered.any():
                break
            
            # Add noise for randomization if seed > 0
            if seed > 0:
                noise = rng.random() * 0.1
            else:
                noise = 0.0
            
            # Find best block using vectorized argmax
            scores = gains.astype(np.float64).copy()
            if seed > 0:
                scores += rng.random() * 0.01
            
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] < 0.5:
                break
                
            selected.append(best_idx)
            
            # Update gains using vectorized operations
            t_idxs = k_sub_t_idxs[best_idx]
            for t_idx in t_idxs:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_subset
                    gains[t_sub_to_k_idxs[t_idx]] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search_swap(current_blocks, max_time):
        """
        Local search with deletion and swap moves.
        State: List of block indices.
        Moves: 
          1. Delete a block if it's redundant
          2. Swap a block with another that maintains coverage
        """
        current_blocks = list(current_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(42)
        
        while time.time() < max_time:
            # --- Deletion Pass ---
            changed = True
            while changed:
                changed = False
                rng.shuffle(current_blocks)
                for j in range(len(current_blocks)):
                    b_idx = current_blocks[j]
                    is_essential = False
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        # Remove
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(j)
                        changed = True
                        break  # Restart loop as indices changed

            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)

            # --- Swap Pass ---
            # Try to swap blocks to improve structure
            swaps_done = 0
            max_swaps = 30
            
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() >= max_time:
                    break
                if swaps_done >= max_swaps:
                    break
                    
                old_b = current_blocks[i]
                
                # Find unique subsets covered by old_b
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # No unique subsets. We can swap with any other block.
                    new_b = rng.randint(0, n_k - 1)
                    if new_b in current_blocks:
                        continue
                    
                    # Perform swap
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    swaps_done += 1
                    
                else:
                    # We have unique subsets. New block MUST cover all of them.
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                    
                    if not candidates:
                        continue
                    
                    new_b = rng.choice(list(candidates))
                    
                    # Perform swap
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    swaps_done += 1

            # Check for improvement after swaps
            # Do a quick deletion pass
            changed = True
            while changed:
                changed = False
                rng.shuffle(current_blocks)
                for j in range(len(current_blocks)):
                    b_idx = current_blocks[j]
                    is_essential = False
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(j)
                        changed = True
                        break

            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)

        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try multiple randomized greedy restarts
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
        for seed in seeds:
            if time.time() >= deadline - 10.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run local search with swaps on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        ls_result = local_search_swap(best_blocks, deadline - 5.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
