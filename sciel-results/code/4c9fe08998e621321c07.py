import itertools
import time
import random
from collections import Counter
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Leave a safety margin. The evaluator kills at 150s, but we want to return cleanly.
    # 115s is a safe deadline for internal logic.
    deadline = start_time + 115.0 

    # Precompute all t-subsets (triples)
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block_indices: shape (num_blocks, C(k, t))
    # block_indices[i] contains the indices of the t-subsets covered by block i
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        
        if time.time() > deadline - 5:
            # Fallback if precomputation takes too long (unlikely for 24,7,3)
            # But let's assume it finishes.
            pass

    # Precompute frequency of each t-subset across all blocks
    # freq[j] = number of blocks that cover t-subset j
    # This is useful for weighted greedy (inverse frequency)
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    
    # Inverse frequency for weighting (higher weight for rare subsets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple strategies/restarts
    seed = 0
    # Allocate time for local search later, so we stop greedy earlier
    greedy_deadline = deadline - 40.0
    
    while time.time() < greedy_deadline:
        # Cycle through strategies
        strategy = seed % 3
        
        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        else:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        
        if blocks is None:
            seed += 1
            continue
            
        # Post-processing: Remove redundant blocks
        blocks = _remove_redundant(blocks, block_indices, num_tsubsets)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        # Early stop if we hit a very good number
        if best_len <= 76:
            # We can stop generating new random covers if we are already very good,
            # and focus remaining time on local search.
            # However, sometimes a slightly worse random cover is easier to optimize.
            # So we continue a bit but break if we have plenty of time for LS.
            if time.time() > deadline - 50.0:
                break

    # Phase 2: Local Search on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search_vectorized(best_blocks, all_blocks, block_indices, num_tsubsets, deadline)
        
    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        if best_blocks is None:
            best_blocks = []
             
    return best_blocks

def _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy', inv_freq=None, seed=0):
    """
    Vectorized greedy set cover.
    """
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None  # Signal timeout
            
        if mode == 'greedy':
            # Gain = number of uncovered triples covered by the block
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'weighted':
            # Weighted gain = sum of inv_freq for uncovered triples
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'randomized':
            # Calculate gains, then pick randomly from top candidates
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Get indices of blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            # If too many candidates, sample a few
            rng = random.Random(seed + len(selected_indices))
            if len(candidates) > 10:
                candidates = rng.sample(list(candidates), 10)
            
            best_idx = candidates[rng.randint(0, len(candidates)-1)]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        else:
            raise ValueError("Unknown mode")
            
    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected_indices]

def _remove_redundant(blocks, block_indices, num_tsubsets):
    """
    Remove blocks that do not cover any unique t-subset.
    Uses vectorized counting.
    """
    if not blocks:
        return blocks
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # Map selected blocks to their global indices to use block_indices?
    # We need to find the index in `all_blocks` for each block in `blocks`.
    # Since `blocks` are lists of ints, and `all_blocks` are tuples, we can map.
    block_to_global_idx = {}
    for i, b in enumerate(all_blocks):
        block_to_global_idx[tuple(b)] = i
        
    selected_global_indices = []
    for b in blocks:
        tb = tuple(b)
        if tb in block_to_global_idx:
            selected_global_indices.append(block_to_global_idx[tb])
        else:
            # Should not happen if blocks came from all_blocks
            pass
            
    if not selected_global_indices:
        return blocks
        
    sel_indices = np.array(selected_global_indices, dtype=np.int32)
    
    # Get the t-subsets for these selected blocks
    # shape: (n_selected, t_subsets_per_block)
    sel_block_tsubs = block_indices[sel_indices]
    
    # Count coverage
    cover_count = np.bincount(sel_block_tsubs.flatten(), minlength=num_tsubsets)
    
    # A block is redundant if all its t-subsets have count > 1
    # For each selected block, check if min(cover_count[its_tsubs]) > 1
    # We can do this by iterating, or vectorizing.
    # Since n is small (~80), iterating is fine and clear.
    
    keep_indices = []
    for i in range(len(sel_indices)):
        tsubs = sel_block_tsubs[i]
        counts = cover_count[tsubs]
        if np.min(counts) > 1:
            # Redundant
            pass
        else:
            keep_indices.append(i)
            
    if len(keep_indices) == len(sel_indices):
        return blocks
        
    return [blocks[i] for i in keep_indices]

def _local_search_vectorized(blocks, all_blocks, block_indices, num_tsubsets, deadline):
    """
    Optimized local search using vectorized coverage counting.
    Strategy:
    1. Identify "critical" blocks (blocks that have at least one unique triple).
    2. Try to remove a critical block B.
    3. Find a block A (from all possible blocks, or existing blocks modified) that can cover the unique triples of B.
    4. If such A exists and A is not B, replace B with A (if A is new) or just remove B if A already exists?
       Actually, the standard move is: Remove B. The cover is now invalid for U_B (unique triples of B).
       Find a block A such that U_B subset of T(A).
       If A is already in the set, we just remove B (since A covers U_B, and other triples of B were covered by others).
       If A is not in the set, we replace B with A.
    """
    if not blocks:
        return blocks
        
    # Map current blocks to global indices
    block_to_global_idx = {}
    for i, b in enumerate(all_blocks):
        block_to_global_idx[tuple(b)] = i
        
    current_global_indices = []
    for b in blocks:
        tb = tuple(b)
        if tb in block_to_global_idx:
            current_global_indices.append(block_to_global_idx[tb])
            
    if not current_global_indices:
        return blocks
        
    n = len(current_global_indices)
    
    # Precompute t-subsets for current blocks
    # We need to update this as we modify the set.
    
    improved_blocks = list(blocks)
    improved_indices = list(current_global_indices)
    
    # Helper to get t-subsets for a list of global indices
    def get_tsubs(indices):
        return block_indices[indices]
        
    # Initial coverage count
    cover_count = np.bincount(get_tsubs(improved_indices).flatten(), minlength=num_tsubsets)
    
    no_progress = 0
    max_no_progress = 5
    
    while time.time() < deadline and no_progress < max_no_progress:
        n = len(improved_indices)
        if n == 0:
            break
            
        # Identify blocks with unique triples
        # A block i has unique triples if min(cover_count[tsubs[i]]) == 1
        # We want to remove blocks with *few* unique triples, as they are easier to replace.
        
        unique_counts = []
        for i in range(n):
            tsubs = block_indices[improved_indices[i]]
            counts = cover_count[tsubs]
            # Number of triples that are uniquely covered by this block
            num_unique = np.sum(counts == 1)
            unique_counts.append(num_unique)
            
        # Sort blocks by number of unique triples (ascending)
        # We try to remove blocks with 0 unique triples first (redundant), then 1, etc.
        sorted_indices = sorted(range(n), key=lambda x: unique_counts[x])
        
        move_made = False
        
        for b_idx in sorted_indices:
            if time.time() > deadline:
                break
                
            b_global = improved_indices[b_idx]
            b_tsubs = block_indices[b_global]
            b_counts = cover_count[b_tsubs]
            
            # If no unique triples, it's redundant
            if np.min(b_counts) > 1:
                # Remove it
                cover_count[b_tsubs] -= 1
                del improved_indices[b_idx]
                del improved_blocks[b_idx]
                move_made = True
                break
            
            # If it has unique triples, we need to find a replacement
            unique_mask = (b_counts == 1)
            unique_tsubs = b_tsubs[unique_mask]
            
            if len(unique_tsubs) == 0:
                # Should have been caught by redundant check, but just in case
                continue
                
            # We need to find a block A such that it covers ALL unique_tsubs.
            # Note: A can be any block from `all_blocks`.
            # To speed this up, we can filter candidates.
            # The candidate block A must contain all points in the union of unique_tsubs?
            # No, it just needs to contain the triples.
            # If unique_tsubs are disjoint, A must contain their union.
            # If they share points, A must contain the union of points.
            
            # Calculate the set of points that MUST be in A
            # Actually, if A covers a triple (a,b,c), A must contain a,b,c.
            # So A must contain the union of all points in unique_tsubs.
            required_points = set()
            for sub in unique_tsubs:
                required_points.update(sub)
                
            if len(required_points) > k:
                continue # No single block can cover these unique triples
                
            # Find all blocks in all_blocks that contain required_points
            # We can precompute a mapping from point-set to blocks? Or just filter.
            # Since v=24, k=7, all_blocks is 245k.
            # Filtering 245k blocks every time is too slow.
            
            # Optimization: Precompute for each point p, the list of blocks containing p.
            # Or better, since we are doing this in a loop, we can use the `block_indices` structure?
            # No, block_indices is triple-based.
            
            # Let's use a heuristic: Only consider blocks that are "close" to B?
            # Or, precompute a dictionary: point -> list of block indices containing point.
            # This is expensive to build once, but fast to query.
            
            # Let's build `point_to_blocks` once outside the loop?
            # It's a list of lists.
            # point_to_blocks[p] = [idx for idx, block in enumerate(all_blocks) if p in block]
            
            # Since we don't have it in scope, let's check if we can pass it or build it.
            # Building it takes time.
            # Alternative: Iterate through a random subset of all_blocks?
            # Or, notice that B is a block. The replacement A should be "similar" to B?
            # Not necessarily.
            
            # Let's try a different approach:
            # Instead of searching all_blocks, we can try to modify B itself.
            # Swap a point in B with a point outside B.
            # This is a 1-swap local search.
            
            b_points = list(all_blocks[b_global])
            other_points = [p for p in range(v) if p not in b_points]
            
            found_replacement = False
            # Try swapping each point in B with each point outside B
            for p_out in b_points:
                for p_in in other_points:
                    new_block_points = [p if p != p_out else p_in for p in b_points]
                    new_block_tuple = tuple(sorted(new_block_points))
                    
                    # Check if this new block is valid (distinct points)
                    if len(set(new_block_points)) != k:
                        continue
                        
                    # Get global index of new block
                    if new_block_tuple not in block_to_global_idx:
                        continue
                    a_global = block_to_global_idx[new_block_tuple]
                    
                    # Check if A covers all unique_tsubs of B
                    a_tsubs = block_indices[a_global]
                    a_set = set(a_tsubs.tolist())
                    
                    if all(sub in a_set for sub in unique_tsubs):
                        # A covers the unique parts of B.
                        # Now check if removing B and adding A (if A is new) is valid.
                        
                        # Check if A is already in the set (excluding B)
                        a_in_set = False
                        for i, idx in enumerate(improved_indices):
                            if idx == a_global and i != b_idx:
                                a_in_set = True
                                break
                                
                        if a_in_set:
                            # A is already there. We can just remove B.
                            # But we must ensure B's non-unique parts are still covered.
                            # They are, by definition of "unique".
                            cover_count[b_tsubs] -= 1
                            del improved_indices[b_idx]
                            del improved_blocks[b_idx]
                            move_made = True
                            found_replacement = True
                            break
                        else:
                            # A is not in the set. We replace B with A.
                            # Check if A introduces any NEW uncovered triples?
                            # No, A is a valid block.
                            # But does A cover all of B's triples?
                            # It covers the unique ones. The others were covered by others.
                            # So the cover remains valid.
                            
                            # Update cover count
                            cover_count[b_tsubs] -= 1
                            cover_count[a_tsubs] += 1
                            
                            improved_indices[b_idx] = a_global
                            improved_blocks[b_idx] = list(new_block_points)
                            
                            move_made = True
                            found_replacement = True
                            break
                            
                if found_replacement:
                    break
                    
            if found_replacement:
                break
                
        if not move_made:
            no_progress += 1
        else:
            no_progress = 0
            
    return improved_blocks

import math # Added missing import for math.comb
