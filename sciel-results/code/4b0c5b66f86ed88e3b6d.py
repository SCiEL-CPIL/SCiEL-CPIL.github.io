import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin
    
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each k-block
    # shape: (n_ks, C(k,t))
    t_count = 1
    for i in range(t):
        t_count *= (k - i)
        t_count //= (i + 1)
        
    sub_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            sub_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        selected_blocks = []
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Vectorized gain calculation
            # uncovered[sub_indices] is a boolean array of shape (n_ks, t_count)
            # Summing along axis 1 gives the number of newly covered subsets for each block
            gains = uncovered[sub_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if uncovered > 0 and we have valid blocks
            
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            
            selected_blocks.append(best)
            subs = sub_indices[best]
            coverage_count[subs] += 1
            uncovered[subs] = False
            
        return selected_blocks

    def reduce_cover(blocks, seed=0):
        random.seed(seed)
        
        if not blocks:
            return []
            
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            # Calculate min coverage for each block's subsets
            # This helps prioritize blocks that are "redundant"
            # min_cov[b] = min(coverage_count[sub_indices[b]])
            # If min_cov[b] > 1, it is safe to remove.
            
            # We can just iterate and check. To be faster, we can sort.
            # But sorting every iteration is O(N log N). N is ~70. It's fine.
            
            # Calculate safety score: how "redundant" is the block?
            # A block is redundant if all its subsets are covered by others.
            # Let's just try to remove blocks in a random order or sorted by "uniqueness".
            
            # Heuristic: Remove blocks that cover the most "common" subsets first?
            # Or just random shuffle to avoid bias.
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                # Check if all subsets in this block are covered by other blocks
                # i.e., coverage_count[subs] > 1
                if np.all(coverage_count[subs] > 1):
                    # Safe to remove
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        """
        Try to improve the solution by swapping blocks.
        This is a simple hill climbing step.
        """
        random.seed(seed)
        if not blocks:
            return blocks
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            # Try to remove blocks again (in case previous step left some)
            # Then try swaps?
            # Swaps are expensive to check.
            # Let's stick to reduction for now, but maybe a "re-greedy" on the remaining uncovered?
            # No, it's fully covered.
            
            # A better polish: 
            # Pick a random block in the set.
            # Remove it.
            # If the set becomes invalid, re-add blocks to cover the gaps.
            # Then reduce again.
            # This is a "kick" move.
            
            if len(current_blocks) > 0:
                # Pick a random block to kick out
                kick_idx = random.randint(0, len(current_blocks) - 1)
                kick_block = current_blocks[kick_idx]
                
                # Remove it
                new_blocks = [b for i, b in enumerate(current_blocks) if i != kick_idx]
                
                # Update coverage
                # We need to check if it's valid.
                # Instead of updating counts for all, let's just check uncovered.
                # But we need counts to know what's uncovered.
                
                # Let's do this:
                # 1. Remove kick_block from coverage_count
                coverage_count[sub_indices[kick_block]] -= 1
                
                # 2. Find uncovered subsets
                uncovered = coverage_count == 0
                
                # 3. If none uncovered, we improved!
                if not np.any(uncovered):
                    current_blocks = new_blocks
                    improved = True
                    continue
                    
                # 4. If uncovered, we need to add blocks to cover them.
                # Use greedy to cover the newly uncovered subsets.
                # We can reuse the logic from greedy_cover but only for the uncovered ones.
                
                # To keep it simple and fast, we might just revert the kick if it causes too many issues?
                # Or we can try to find a single block that covers the most uncovered ones.
                
                gains = uncovered[sub_indices].sum(axis=1)
                max_gain = np.max(gains)
                
                if max_gain > 0:
                    candidates = np.where(gains == max_gain)[0]
                    # Prefer a block that is NOT the one we kicked?
                    # And NOT already in the set?
                    valid_candidates = [c for c in candidates if c not in new_blocks]
                    
                    if valid_candidates:
                        # Pick one randomly
                        new_block = random.choice(valid_candidates)
                        
                        # Add it
                        new_blocks.append(new_block)
                        coverage_count[sub_indices[new_block]] += 1
                        
                        # Now we have a valid set again (hopefully smaller or same size)
                        # Check size
                        if len(new_blocks) < len(current_blocks):
                            current_blocks = new_blocks
                            improved = True
                        elif len(new_blocks) == len(current_blocks) and random.random() < 0.5:
                            # Swap might help future reductions
                            current_blocks = new_blocks
                            improved = True
                            
                # Revert coverage count for the kick block? 
                # No, we already subtracted it.
                # If we added a new block, we added its count.
                # If we didn't add a block (no valid candidates), we must revert the kick.
                # Wait, if max_gain == 0, that means no block covers the uncovered ones? 
                # That shouldn't happen if the original set was valid.
                
                # If we didn't improve, we need to restore the state.
                # This logic is getting complex. Let's stick to simple reduction and restarts.
                # The "kick" logic is prone to bugs if not careful with state restoration.
                
                # Let's just do simple reduction in the main loop.
                pass
                
        return current_blocks

    # Main search loop
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks = greedy_cover(seed)
        
        if blocks is None:
            break
            
        reduced = reduce_cover(blocks, seed + 100)
        
        if reduced is not None and len(reduced) < best_score:
            best_score = len(reduced)
            best_solution = reduced
            
        restart_count += 1
        
        # If we get a very good score, we might want to spend more time polishing
        # But for now, just restart.

    if best_solution is None:
        # Fallback: run one greedy without time limit check inside (should be fast)
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            
    if best_solution is None:
        return []

    # Convert indices to actual blocks
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
