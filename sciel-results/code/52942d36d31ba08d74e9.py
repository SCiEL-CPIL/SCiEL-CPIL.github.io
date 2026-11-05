import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_id = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple IDs it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_id[tup] for tup in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Convert to numpy arrays for vectorized operations
    block_triples_np = [np.array(ts, dtype=np.int32) for ts in block_triples]

    # Map triple ID to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)
    triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]

    # Helper for fast redundancy check
    def is_redundant(b_idx, cover_count):
        # A block is redundant if every triple it covers is covered by at least one other block
        # i.e., cover_count[t] > 1 for all t in block_triples[b_idx]
        return np.min(cover_count[block_triples_np[b_idx]]) > 1

    def greedy_construction(seed=0, time_budget=deadline):
        rng = random.Random(seed)
        # gain[b_idx] is the number of currently uncovered triples covered by block b_idx
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > time_budget:
                break
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random tie-breaking
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            for t_idx in block_triples_np[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    gain[triple_blocks_np[t_idx]] -= 1
                        
        return selected_blocks

    def complete_cover(selected, deadline):
        """Ensure all triples are covered. If greedy stopped early, add more blocks."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
            
        while True:
            if time.time() > deadline:
                break
            uncovered_mask = (cover_count == 0)
            if not np.any(uncovered_mask):
                break
                
            # Find a block that covers the most uncovered triples
            # To speed up, we can sample or just pick the first one that covers a specific uncovered triple
            # Let's pick an uncovered triple and find a block covering it with max gain among those
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break
                
            # Pick a random uncovered triple to anchor the search
            target_tid = unc_idx[random.randint(0, len(unc_idx)-1)]
            
            # Candidates are blocks covering this target
            candidates = triple_blocks_np[target_tid]
            
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                # Gain is number of uncovered triples this block covers
                g = np.sum(uncovered_mask[block_triples_np[b_idx]])
                if g > best_gain:
                    best_gain = g
                    best_b = b_idx
                    
            if best_b == -1:
                break
                
            selected.append(best_b)
            cover_count[block_triples_np[best_b]] += 1
            
        return selected

    def reduce_cover(selected, deadline):
        """Remove redundant blocks iteratively."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
            
        active = [True] * len(selected)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                b_idx = selected[i]
                if is_redundant(b_idx, cover_count):
                    active[i] = False
                    cover_count[block_triples_np[b_idx]] -= 1
                    changed = True
                    
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search(selected, deadline):
        """
        Local search: Try to remove one block and replace it with one or more others
        to reduce total count, or just remove if redundant (handled by reduce_cover mostly,
        but this allows swaps).
        """
        if len(selected) < 2:
            return reduce_cover(selected, deadline)
            
        selected = list(selected)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
            
        rng = random.Random(123)
        improved = True
        # Limit iterations to avoid infinite loops if stuck
        max_iterations = 50
        
        while improved and time.time() < deadline and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try removing each block in random order
            order = list(range(len(selected)))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                
                b_idx = selected[i]
                # Tentatively remove
                cover_count[block_triples_np[b_idx]] -= 1
                uncovered = (cover_count == 0)
                
                if not np.any(uncovered):
                    # Block was redundant, remove it permanently
                    selected.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all currently uncovered triples
                # This is a "1-for-1" swap or "1-for-0" if we found a better one, but here we just want to cover the gap
                unc_idx = np.where(uncovered)[0]
                
                # If the number of uncovered triples is small, we can intersect the candidate sets
                # If it's large, it's unlikely a single block covers them all, so we skip this specific removal
                # Note: max triples in a block is C(7,3)=35. So if len(unc_idx) > 35, no single block can cover them.
                if len(unc_idx) <= 35:
                    # Find intersection of blocks covering all uncovered triples
                    # Start with blocks covering the first uncovered triple
                    candidates = set(triple_blocks_np[unc_idx[0]].tolist())
                    
                    for tid in unc_idx[1:]:
                        if not candidates:
                            break
                        candidates &= set(triple_blocks_np[tid].tolist())
                        
                    # Exclude the block we removed (and others already in selected? No, we can reuse if it helps, but usually we want new ones)
                    # Actually, if we removed b_idx, we can't add it back.
                    candidates.discard(b_idx)
                    
                    if candidates:
                        # Pick one candidate
                        new_b = rng.choice(list(candidates))
                        
                        # Check if this new block is already in selected (at a different index)
                        # If so, adding it might create a duplicate or just increase count.
                        # The task says blocks must be distinct.
                        if new_b in selected:
                            # We can't add a duplicate. Revert.
                            cover_count[block_triples_np[b_idx]] += 1
                        else:
                            # Perform swap: remove b_idx, add new_b
                            selected.pop(i)
                            selected.append(new_b)
                            cover_count[block_triples_np[new_b]] += 1
                            improved = True
                            break
                    else:
                        # Revert
                        cover_count[block_triples_np[b_idx]] += 1
                else:
                    # Too many uncovered to fix with 1 block, revert
                    cover_count[block_triples_np[b_idx]] += 1
            
        return reduce_cover(selected, deadline)

    def local_search_swap(selected, deadline):
        """
        Alternative local search: Try to replace one block with another that allows removal of a second block later.
        Or simply: Remove block A, try to cover gap with block B. If B covers gap AND allows removal of C, do it.
        This is more complex. Let's stick to the simpler 1-for-1 swap that maintains coverage.
        """
        pass

    best_cover = None
    best_size = float('inf')
    
    # Phase 1: Multiple Greedy Restarts + Reduce
    seed = 0
    # Allocate most time to this phase
    phase1_deadline = deadline - 10.0
    
    while time.time() < phase1_deadline:
        # Greedy
        blocks = greedy_construction(seed, phase1_deadline - 1.0)
        
        # Complete if necessary
        blocks = complete_cover(blocks, phase1_deadline - 1.0)
        
        # Reduce
        blocks = reduce_cover(blocks, phase1_deadline - 1.0)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_cover = blocks[:]
            
        # If we found a very good solution, we can stop early?
        # Lower bound is 69. 76 is known.
        if best_size <= 72:
             break
             
        seed += 1
        
        # Occasionally run local search on the current best or current candidate
        if seed % 3 == 0 and best_cover:
             # Run local search on a copy of the best cover
             ls_deadline = min(deadline, time.time() + 5.0)
             improved = local_search(best_cover[:], ls_deadline)
             if len(improved) < best_size:
                 best_size = len(improved)
                 best_cover = improved[:]
                 
    # Phase 2: Final Local Search on the Best Cover
    if best_cover:
        final_ls_deadline = deadline - 2.0
        best_cover = local_search(best_cover, final_ls_deadline)
        best_size = len(best_cover)
        
        # One more reduction pass just in case
        best_cover = reduce_cover(best_cover, deadline - 1.0)
        best_size = len(best_cover)

    # Fallback if something went wrong
    if not best_cover:
        blocks = greedy_construction(0, deadline - 5.0)
        blocks = complete_cover(blocks, deadline - 2.0)
        blocks = reduce_cover(blocks, deadline - 1.0)
        best_cover = blocks

    # Convert to list of lists of integers
    return [list(all_blocks[b_idx]) for b_idx in best_cover]

# Safety check for the specific task parameters
# The evaluator calls construct(v, k, t)
# For C(24, 7, 3), v=24, k=7, t=3
