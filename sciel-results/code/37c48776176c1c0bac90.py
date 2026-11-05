import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_k_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(tids)
        
    # Map each triple to the list of block indices that cover it
    # This is needed for efficient gain updates (reverse index)
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    def greedy_construct(seed):
        rng = random.Random(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        # Initialize with the total number of triples in the block (C(k, t))
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 5:
                # If we run out of time during construction, we must return a valid cover.
                # The current selected_blocks might not be a complete cover.
                # We need to finish the cover quickly or return best effort.
                # However, the task requires a valid cover.
                # If we break here, we must complete the cover using a fast greedy that ignores time?
                # Or better, check time BEFORE starting a new iteration.
                # If we are mid-construction and time is up, we are in trouble.
                # Let's assume the loop finishes. If not, we need a fallback.
                pass
            
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true and blocks are valid
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            # For each triple in the selected block:
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Because this triple is no longer "newly covered" by them
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    def reduce_cover(selected_blocks):
        # Convert to list of sets of triple indices for quick check?
        # We maintain cover_count[t]
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_blocks)
        
        # Pass 1: Remove redundant blocks
        # A block is redundant if all its triples have cover_count > 1
        # We iterate and remove.
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                # Check if redundant
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    # Update cover counts
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search(selected_blocks, max_iters=1000):
        # Try to remove a block and replace it with a better one?
        # Or just try removing blocks.
        # Let's try a simple heuristic: for each block, check if it can be removed.
        # If not, try swapping it with another block that covers the unique triples it covers.
        
        # First, compute coverage counts
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        current_size = len(selected_blocks)
        best_size = current_size
        best_solution = list(selected_blocks)
        
        # We will perform random restarts of a local improvement loop
        # Actually, just try to remove blocks if possible.
        # Then try to swap blocks.
        
        # Step 1: Remove redundant blocks
        solution = reduce_cover(selected_blocks)
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = list(solution)
        
        # Step 2: Try to swap blocks
        # For each block in the solution, calculate the unique triples it covers.
        # If a block covers no unique triples, it was removed in Step 1.
        # So every block covers at least one unique triple.
        
        # Heuristic: For each block, try to find a replacement block that covers the same unique triples 
        # and potentially more common triples, allowing removal of another block?
        # This is complex. Let's stick to simple removal and random restarts of greedy.
        
        # Let's try a "kick" approach:
        # Remove a random block, then greedily add blocks to cover the now-uncovered triples.
        # If the new size is smaller, accept.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if len(solution) == 0:
                break
                
            # Pick a random block to remove
            remove_idx = random.randint(0, len(solution) - 1)
            removed_block = solution.pop(remove_idx)
            
            # Recompute coverage
            cover_count = np.zeros(num_triples, dtype=np.int32)
            for b_idx in solution:
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] += 1
            
            # Find uncovered triples
            uncovered = [t_idx for t_idx in range(num_triples) if cover_count[t_idx] == 0]
            
            if not uncovered:
                # This shouldn't happen if the previous solution was valid
                solution.append(removed_block)
                continue
            
            # Greedily add blocks to cover uncovered
            # Gain for a block is number of uncovered triples it covers
            gain = np.zeros(num_blocks, dtype=np.int32)
            for t_idx in uncovered:
                for b_idx in triple_blocks[t_idx]:
                    gain[b_idx] += 1
            
            # Add blocks
            while uncovered:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = random.choice(candidates.tolist())
                
                solution.append(b_idx)
                
                # Update gain and uncovered
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] == 0:
                        cover_count[t_idx] = 1
                        uncovered.remove(t_idx) # O(N) but N is smallish? 2024. Might be slow.
                        for other_b_idx in triple_blocks[t_idx]:
                            gain[other_b_idx] -= 1
                    else:
                        cover_count[t_idx] += 1
                
                # Optimization: Instead of list remove, use a boolean array or set
                # For now, let's assume it's fast enough for a few iterations.
                # Actually, list.remove is O(N). 2024 * 1000 iters * 76 blocks... might be too slow.
                # Let's break if it takes too long.
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = list(solution)
            
            # Restore removed block for next iteration? 
            # No, we want to explore different neighborhoods.
            # But we need a valid starting point. 
            # Let's just restart from best_solution every time?
            # That's basically random restarts of local search.
            
            solution = list(best_solution)

        return best_solution

    best_cover = None
    seed = 0
    
    # Phase 1: Greedy with different seeds
    while time.time() < deadline - 10:
        if time.time() - start_time > deadline:
            break
            
        cover = greedy_construct(seed)
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
        
        seed += 1
        
        # If we have a good cover, spend remaining time on local search
        if best_cover and len(best_cover) <= 76:
            # Try local search on the best cover
            improved = local_search(best_cover, max_iters=50)
            if len(improved) < len(best_cover):
                best_cover = improved

    # Final check
    if best_cover is None:
        cover = greedy_construct(0)
        return [list(all_k_blocks[b_idx]) for b_idx in cover]
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_cover]
