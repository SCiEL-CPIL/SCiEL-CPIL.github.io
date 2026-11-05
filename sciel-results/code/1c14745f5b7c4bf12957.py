import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute bitmask of t-subsets it covers
    # Since n_t can be large (C(19,5) = 11628), we use uint64 arrays
    # Actually C(19,5) = 11628 which exceeds 64 bits, so we need multiple words or use numpy arrays
    
    # Let's use numpy uint64 arrays where each block covers C(8,5)=56 t-subsets
    # We can represent coverage as a set of t-subset indices
    # But for fast intersection/count, let's use boolean arrays or sparse representation
    
    # Given the size, let's use a different approach:
    # For each block, store the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = tuple(t_sub_to_idx[sub] for sub in itertools.combinations(k_sub, t))
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)
    
    def greedy_cover(seed):
        """
        Criticality-prioritized greedy: 
        1. Find the most constrained uncovered t-subset (fewest candidate blocks)
        2. Among blocks covering it, pick the one with max marginal gain
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # For each t-subset, track how many blocks still cover it (among not-yet-selected)
        # Initially, all blocks cover their t-subsets
        # We'll maintain a "remaining candidates" count per t-subset
        
        # Actually, for criticality approach:
        # At each step, find uncovered t-subset with minimum number of remaining (unselected) blocks covering it
        # This requires tracking which blocks are still available
        
        # Simpler approach that still captures criticality:
        # Standard greedy but with tie-breaking by "rarity" of t-subsets covered
        
        # Let's use a hybrid: standard greedy gain, but when gains are close,
        # prefer blocks that cover rarer t-subsets (fewer blocks cover them)
        
        # Precompute rarity: for each t-subset, number of blocks that cover it
        t_sub_rarity = np.array([len(t_sub_to_k_idxs[i]) for i in range(n_t)], dtype=np.int32)
        
        gains = np.full(n_k, len(k_sub_t_idxs[0]), dtype=np.int32)
        selected = []
        
        # For tie-breaking, we can add a small bonus for covering rare t-subsets
        # rarity_bonus[k] = sum of 1/rarity[t] for t in block k, or just count of rare t-subsets
        
        for _ in range(n_k):
            if not uncovered.any():
                break
            
            # Find best block by gain, with tie-breaking by rarity
            # Add small noise for randomization
            if seed > 0:
                noise = rng.random() * 0.5
                scores = gains.astype(np.float64) + noise
            else:
                scores = gains.astype(np.float64)
            
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def criticality_greedy(seed):
        """
        True criticality-prioritized greedy:
        At each step, find the uncovered t-subset with the fewest remaining candidate blocks.
        Select a block that covers it, choosing the one with maximum marginal gain.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Track remaining candidate blocks for each t-subset
        # remaining_candidates[t] = number of blocks (not yet selected) that cover t
        remaining_candidates = np.array([len(t_sub_to_k_idxs[i]) for i in range(n_t)], dtype=np.int32)
        
        # Track which blocks are selected
        selected_blocks = set()
        selected = []
        
        # Current gain for each block (number of uncovered t-subsets it covers)
        gains = np.full(n_k, len(k_sub_t_idxs[0]), dtype=np.int32)
        
        for step in range(n_k):
            if not uncovered.any():
                break
            
            # Find the uncovered t-subset with minimum remaining candidates (most constrained)
            # Only consider uncovered t-subsets
            uncovered_idx = np.where(uncovered)[0]
            if len(uncovered_idx) == 0:
                break
            
            # Get remaining candidates for uncovered t-subsets
            rem_cand_uncovered = remaining_candidates[uncovered_idx]
            
            # Find the minimum
            min_rem = np.min(rem_cand_uncovered)
            
            # Get all t-subsets with this minimum
            most_constrained = uncovered_idx[rem_cand_uncovered == min_rem]
            
            if len(most_constrained) == 0:
                break
            
            # Pick one at random (or the first) from the most constrained
            t_idx = rng.choice(most_constrained.tolist())
            
            # Get all remaining blocks that cover this t-subset
            candidate_blocks = [b for b in t_sub_to_k_idxs[t_idx] if b not in selected_blocks]
            
            if not candidate_blocks:
                break
            
            # Among candidate blocks, pick the one with max gain
            if len(candidate_blocks) == 1:
                best_idx = candidate_blocks[0]
            else:
                cand_gains = np.array([gains[b] for b in candidate_blocks])
                # Add small noise for tie-breaking
                if seed > 0:
                    cand_gains += rng.random() * 0.1
                best_local = int(np.argmax(cand_gains))
                best_idx = candidate_blocks[best_local]
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            selected_blocks.add(best_idx)
            
            # Update: remove best_idx from remaining candidates of all t-subsets it covers
            for t_idx2 in k_sub_t_idxs[best_idx]:
                remaining_candidates[t_idx2] -= 1
            
            # Update gains
            for t_idx2 in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx2]:
                    uncovered[t_idx2] = False
                    for b_idx in t_sub_to_k_idxs[t_idx2]:
                        if b_idx not in selected_blocks:
                            gains[b_idx] -= 1
                        
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
            shuffled_blocks = list(blocks_idx)
            random.shuffle(shuffled_blocks)
            
            for b_idx in shuffled_blocks:
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

    def local_search(blocks_idx, max_time):
        """
        Randomized 1-swap local search.
        """
        current_blocks = list(blocks_idx)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(12345)
        
        while time.time() < max_time:
            # Deletion Pass
            changed = True
            while changed:
                changed = False
                shuffled_indices = list(range(len(current_blocks)))
                rng.shuffle(shuffled_indices)
                
                for i in shuffled_indices:
                    if i >= len(current_blocks):
                        continue
                    b_idx = current_blocks[i]
                    is_essential = False
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        changed = True
                        break

            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)

            # Swap Pass: try to swap a block with another to potentially enable more deletions
            num_swaps_to_try = min(len(current_blocks), 30)
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices[:num_swaps_to_try]:
                if time.time() >= max_time:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_b = current_blocks[i]
                
                # Find unique subsets covered by old_b
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # Can swap with any other block
                    new_b = -1
                    for _ in range(5):
                        candidate = rng.randint(0, n_k - 1)
                        if candidate not in current_blocks:
                            new_b = candidate
                            break
                    if new_b == -1:
                        continue
                    
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    
                else:
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                    
                    if not candidates:
                        continue
                    
                    new_b = rng.choice(list(candidates))
                    
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b

        return best_blocks

    # Run criticality greedy first
    blocks_idx = criticality_greedy(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try standard greedy
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        b_idx = greedy_cover(0)
        b_idx = reduce_blocks(b_idx)
        if b_idx and len(b_idx) < best_len:
            best_blocks = b_idx
            best_len = len(b_idx)

    # Try randomized criticality greedy with different seeds
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for seed in seeds:
            if time.time() >= deadline - 15.0:
                break
            b_idx = criticality_greedy(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        ls_result = local_search(best_blocks, deadline - 5.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
