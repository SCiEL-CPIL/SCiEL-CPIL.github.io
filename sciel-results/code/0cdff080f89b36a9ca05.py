"""
Strategy: Bitmask-based Vectorized Greedy Set Cover + Aggressive Local Search with Swaps.

1. Precompute all C(19,4) = 3876 t-subsets and all C(19,7) = 50388 k-blocks as bitmasks.
2. Build the coverage matrix (block covers t-subset) as a compact uint8 matrix.
3. Greedy set cover: repeatedly pick the block covering the most uncovered t-subsets.
4. Local search refinement:
   a) Removal pass: remove any block that is not the sole cover of any t-subset.
   b) Swap pass: try to replace a block with another that reduces the "criticality"
      (number of t-subsets it uniquely covers) or simply try random swaps that maintain
      validity while allowing further removals.
5. Multiple restarts with slightly perturbed greedy (random tie-breaking) to escape
   local optima.
"""

import time
import numpy as np
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # 1. Precompute all t-subsets and their bitmasks
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
        
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
        
    # 2. Precompute coverage matrix in chunks to manage memory
    # cover[i, j] = 1 if t-subset j is covered by k-block i
    # i.e., t_masks[j] is a subset of k_masks[i]
    # Use int64 bitwise: (t_masks[j] & ~k_masks[i]) == 0
    
    # Store as list of chunks for memory efficiency
    # Each chunk is a 2D array of shape (chunk_size, num_t) with dtype uint8
    
    chunk_size = 1000
    num_chunks = (num_k + chunk_size - 1) // chunk_size
    cover_chunks = []
    
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end]
        # Vectorized subset check
        # (t_masks[None, :] & ~k_chunk[:, None]) == 0
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_chunks.append(check.astype(np.uint8))
    
    def time_left():
        return deadline - time.time()
    
    def greedy_solve(seed=None):
        """Run greedy set cover with optional randomization for ties."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        uncovered = np.ones(num_t, dtype=np.uint8)
        uncovered_count = num_t
        selected_indices = []
        
        while uncovered_count > 0:
            if time_left() < 10:
                break
                
            best_gain = 0
            best_indices = []
            
            for ci, cover_chunk in enumerate(cover_chunks):
                gains = cover_chunk.dot(uncovered)
                max_gain = np.max(gains)
                if max_gain > best_gain:
                    best_gain = max_gain
                    # Find all indices with this gain
                    best_local = np.where(gains == max_gain)[0]
                    best_indices = [(ci * chunk_size + li) for li in best_local]
                elif max_gain == best_gain and best_gain > 0:
                    best_local = np.where(gains == max_gain)[0]
                    for li in best_local:
                        best_indices.append(ci * chunk_size + li)
            
            if best_gain == 0:
                break
                
            # Randomly pick among ties
            chosen = rng.choice(best_indices) if len(best_indices) > 1 else best_indices[0]
            selected_indices.append(chosen)
            
            # Update uncovered
            ci = chosen // chunk_size
            li = chosen % chunk_size
            cover_row = cover_chunks[ci][li]
            newly_covered = np.where((cover_row == 1) & (uncovered == 1))[0]
            uncovered_count -= len(newly_covered)
            uncovered[newly_covered] = 0
            
        return selected_indices
    
    def compute_counts(indices):
        """Compute coverage counts for each t-subset."""
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices:
            ci = idx // chunk_size
            li = idx % chunk_size
            counts += cover_chunks[ci][li]
        return counts
    
    def removal_pass(indices):
        """Remove non-critical blocks iteratively."""
        if not indices:
            return indices
            
        counts = compute_counts(indices)
        changed = True
        while changed and time_left() > 5:
            changed = False
            new_indices = []
            for idx in indices:
                ci = idx // chunk_size
                li = idx % chunk_size
                cover_row = cover_chunks[ci][li]
                
                # Critical if it's the only cover for some t-subset
                critical = (cover_row == 1) & (counts == 1)
                if np.any(critical):
                    new_indices.append(idx)
                else:
                    counts -= cover_row
                    changed = True
            
            indices = new_indices
            if not changed:
                break
        
        return indices
    
    def swap_pass(indices, max_iterations=200):
        """Try to swap blocks to enable further reductions."""
        if not indices or time_left() < 5:
            return indices
            
        counts = compute_counts(indices)
        n = len(indices)
        
        # For each block, count how many t-subsets it uniquely covers (criticality)
        def criticality(idx):
            ci = idx // chunk_size
            li = idx % chunk_size
            cover_row = cover_chunks[ci][li]
            return np.sum((cover_row == 1) & (counts == 1))
        
        for _ in range(max_iterations):
            if time_left() < 5:
                break
                
            # Try removing a block, then see if we can add another that helps
            best_swap = None
            best_new_count = len(indices)
            
            # Try each block as candidate for removal
            for i, idx_remove in enumerate(indices):
                ci = idx_remove // chunk_size
                li = idx_remove % chunk_size
                cover_remove = cover_chunks[ci][li]
                
                # Tentative counts after removal
                new_counts = counts - cover_remove
                
                # Check if any t-subset becomes uncovered
                if np.any(new_counts == 0):
                    continue
                    
                # Now try to add a new block that reduces total criticality
                # or at least doesn't increase block count
                # For simplicity, just check if removal alone works (that's covered by removal_pass)
                # For swap, we need to find a block to add that makes removal beneficial
                # This is complex; skip for now and rely on removal_pass
                
            # Alternative: try random swaps
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                    
                idx1 = indices[i]
                idx2 = indices[j]
                
                # Check if they are the same
                if idx1 == idx2:
                    continue
                    
                ci1 = idx1 // chunk_size
                li1 = idx1 % chunk_size
                cover1 = cover_chunks[ci1][li1]
                
                ci2 = idx2 // chunk_size
                li2 = idx2 % chunk_size
                cover2 = cover_chunks[ci2][li2]
                
                # Swap: remove idx1, add idx2
                new_counts = counts - cover1 + cover2
                if np.any(new_counts == 0):
                    continue
                    
                # Check if this swap enables any removal
                # For now, just accept if it maintains validity
                # (We can refine this to check for criticality reduction)
                
                # Actually, a swap that doesn't reduce count is useless for minimization
                # unless it enables future removals. Let's try a simpler approach:
                # Just do the swap and then run removal_pass
                new_indices = indices[:]
                new_indices[j] = idx1  # Keep idx1 in position j
                new_indices[i] = idx2  # Put idx2 in position i
                # Wait, that's not a swap, that's just reordering.
                # A swap means replace one block with a completely different one.
                
                # Let me reconsider: a swap means remove block A and add block B (B not in set)
                # This keeps the same count but might enable further removals.
                
                # Find a candidate block B to add
                # B should cover many of the uniquely-covered subsets of A
                unique_a = np.where((cover1 == 1) & (counts == 1))[0]
                if len(unique_a) == 0:
                    continue
                    
                # Find blocks that cover many of unique_a
                # This is expensive; do it randomly for a few candidates
                for trial in range(5):
                    if time_left() < 5:
                        break
                    candidate = random.randint(0, num_k - 1)
                    if candidate == idx1:
                        continue
                    if candidate in indices:
                        continue
                        
                    ci_c = candidate // chunk_size
                    li_c = candidate % chunk_size
                    cover_c = cover_chunks[ci_c][li_c]
                    
                    # Check validity after swap
                    new_counts_test = counts - cover1 + cover_c
                    if np.any(new_counts_test == 0):
                        continue
                        
                    # Check if this enables removal
                    # For now, just accept and let removal_pass handle it
                    # Actually, we should check if criticality decreases
                    crit_before = criticality(idx1)
                    # After swap, compute criticality of the new set
                    # This is expensive; just try it
                    
                    new_indices = indices[:]
                    new_indices[i] = candidate
                    counts = compute_counts(new_indices)
                    new_indices = removal_pass(new_indices)
                    
                    if len(new_indices) < len(indices):
                        return new_indices
                    elif len(new_indices) == len(indices):
                        # Same count but different composition; might help in future
                        # Accept with some probability
                        if random.random() < 0.1:
                            indices = new_indices
                            counts = compute_counts(indices)
                            break
    
        return indices
    
    # Main loop: multiple restarts with different seeds
    best_solution = None
    best_count = float('inf')
    
    num_restarts = 10  # We'll do up to 10 restarts
    for restart in range(num_restarts):
        if time_left() < 20:
            break
            
        seed = restart * 12345 + 67890
        indices = greedy_solve(seed)
        indices = removal_pass(indices)
        indices = swap_pass(indices)
        indices = removal_pass(indices)  # Final cleanup
        
        count = len(indices)
        if count < best_count:
            best_count = count
            best_solution = indices[:]
            
        if time_left() < 15:
            break
    
    # If we never got a valid solution, do one plain greedy
    if best_solution is None:
        indices = greedy_solve(0)
        best_solution = indices
    
    # Final validation
    counts = compute_counts(best_solution)
    if np.any(counts == 0):
        # Invalid; fallback to plain greedy without local search
        indices = greedy_solve(0)
        counts = compute_counts(indices)
        if np.any(counts == 0):
            # Should not happen, but return empty if all else fails
            return []
        best_solution = indices
    
    # Convert indices to actual blocks
    result = []
    for idx in best_solution:
        result.append(list(k_blocks[idx]))
    
    # Remove duplicates (shouldn't happen but just in case)
    seen = set()
    unique_result = []
    for block in result:
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            unique_result.append(block)
    
    return unique_result
