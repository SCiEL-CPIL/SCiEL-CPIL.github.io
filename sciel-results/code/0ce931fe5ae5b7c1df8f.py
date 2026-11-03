<<<<<<< SEARCH
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    # Precompute all k-blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    all_tsets = list(itertools.combinations(range(v), t))
    
    num_blocks = len(all_blocks)
    num_tsets = len(all_tsets)
    
    # Map tset to index
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Map block to list of tset indices it covers
    block_to_tsets = []
    for block in all_blocks:
        covered = [tset_index[ts] for ts in itertools.combinations(block, t)]
        block_to_tsets.append(covered)
        
    # Map tset to list of block indices that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets in enumerate(block_to_tsets):
        for ts_idx in tsets:
            tset_to_blocks[ts_idx].append(b_idx)

    # 1. Greedy Set Cover
    # Weights for tsets: 1 / count of blocks covering it
    tset_counts = np.zeros(num_tsets, dtype=np.int32)
    for tsets in block_to_tsets:
        for ts_idx in tsets:
            tset_counts[ts_idx] += 1
            
    # Avoid division by zero
    safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
    tset_weights = 1.0 / safe_counts
    
    # Initial gains for each block
    gains = np.zeros(num_blocks, dtype=np.float64)
    for b_idx, tsets in enumerate(block_to_tsets):
        if tsets:
            gains[b_idx] = np.sum(tset_weights[tsets])
            
    uncovered = np.ones(num_tsets, dtype=np.bool_)
    uncovered_count = num_tsets
    selected_blocks = []
    selected_mask = np.zeros(num_blocks, dtype=np.bool_)
    
    # Greedy Loop
    while uncovered_count > 0 and elapsed() < deadline - 10:
        # Find best block
        best_idx = -1
        best_gain = -1.0
        # Optimization: Only check unselected blocks
        # This loop is O(num_blocks) which is 75k. 
        # We can use numpy argmax on a masked array, but updating gains is the bottleneck.
        
        # Efficient update of gains is hard with numpy if we do it per tset.
        # Let's stick to the previous approach but optimize the inner loop.
        
        # Actually, let's use a priority queue or just scan. Scanning 75k floats is fast in numpy.
        # mask_gains = np.where(selected_mask, -np.inf, gains)
        # best_idx = np.argmax(mask_gains)
        
        # But we need to update gains.
        # When we select a block, we subtract its weight from all blocks covering the same tsets.
        
        # Let's use the reverse mapping to update gains.
        
        # Find best
        # To avoid creating a new array every time, we can just scan.
        # 75k iterations in Python is ~0.005s. 1000 iterations is ~5s.
        # We might do ~300-500 selections. 500 * 0.005 = 2.5s. Acceptable.
        
        for i in range(num_blocks):
            if not selected_mask[i] and gains[i] > best_gain:
                best_gain = gains[i]
                best_idx = i
                
        if best_idx == -1 or best_gain <= 0:
            break
            
        selected_blocks.append(best_idx)
        selected_mask[best_idx] = True
        
        # Update uncovered and gains
        for ts_idx in block_to_tsets[best_idx]:
            if uncovered[ts_idx]:
                uncovered[ts_idx] = False
                uncovered_count -= 1
                w = tset_weights[ts_idx]
                # Update gains for all blocks covering this tset
                for b in tset_to_blocks[ts_idx]:
                    if not selected_mask[b]:
                        gains[b] -= w
                        
    # Current solution is valid.
    # Now Local Search to minimize size.
    
    # State: list of selected block indices
    current_solution = list(selected_blocks)
    
    # Helper to check if a solution is valid and get cover counts
    def get_cover_counts(sol):
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b in sol:
            for ts in block_to_tsets[b]:
                counts[ts] += 1
        return counts
        
    # Initial counts
    cover_counts = get_cover_counts(current_solution)
    
    # Local Search Loop
    # We want to minimize len(current_solution)
    # Move 1: Remove a block if it doesn't uncover any tset.
    # Move 2: Swap 1 block for 1 block (neutral) to escape local minima.
    # Move 3: Swap 2 blocks for 1 block (gainful).
    
    best_solution = list(current_solution)
    best_size = len(best_solution)
    
    # To speed up "Remove if safe", maintain cover_counts.
    # To speed up swaps, we need to find blocks covering specific sets.
    
    # Precompute for each tset, the blocks covering it (already have tset_to_blocks)
    
    # For "2-for-1", we need to find a block covering the union of uncovered tsets from 2 blocks.
    # This is expensive. We can approximate or use a limited search.
    
    # Let's focus on "Remove if safe" and "1-for-1 Swap" first.
    
    no_improvement_count = 0
    max_no_improvement = 50 # Restart or change strategy
    
    while elapsed() < deadline - 5:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # Shuffle to avoid bias
        random.shuffle(current_solution)
        
        for i in range(len(current_solution) - 1, -1, -1):
            b = current_solution[i]
            # Check if all tsets in b have count > 1
            can_remove = True
            for ts in block_to_tsets[b]:
                if cover_counts[ts] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove
                for ts in block_to_tsets[b]:
                    cover_counts[ts] -= 1
                current_solution.pop(i)
                improved = True
                if len(current_solution) < best_size:
                    best_size = len(current_solution)
                    best_solution = list(current_solution)
        
        if improved:
            no_improvement_count = 0
            continue
        else:
            no_improvement_count += 1
            
        # Phase 2: 1-for-1 Swap
        # Pick a random block B_out
        # Find tsets U that would be uncovered
        # Find a block B_in that covers U
        # If found, swap
        
        if no_improvement_count > 10:
            # Try a swap
            if current_solution:
                b_out_idx = random.randint(0, len(current_solution) - 1)
                b_out = current_solution[b_out_idx]
                
                # Find tsets that would become uncovered
                uncovered_tsets = []
                for ts in block_to_tsets[b_out]:
                    if cover_counts[ts] == 1:
                        uncovered_tsets.append(ts)
                
                if not uncovered_tsets:
                    # This shouldn't happen if Phase 1 ran fully, but just in case
                    # Actually if Phase 1 ran fully, no block is removable.
                    # So every block has at least one critical tset.
                    pass
                else:
                    # We need a block that covers ALL tsets in uncovered_tsets
                    # Intersect the lists of blocks covering each tset in uncovered_tsets
                    # Exclude b_out
                    
                    candidate_blocks = set(tset_to_blocks[uncovered_tsets[0]])
                    candidate_blocks.discard(b_out)
                    
                    for ts in uncovered_tsets[1:]:
                        candidate_blocks.intersection_update(tset_to_blocks[ts])
                        if not candidate_blocks:
                            break
                    
                    if candidate_blocks:
                        # Pick a random candidate
                        b_in = random.choice(list(candidate_blocks))
                        
                        # Perform swap
                        # Update counts
                        # Decrement for b_out
                        for ts in block_to_tsets[b_out]:
                            cover_counts[ts] -= 1
                        # Increment for b_in
                        for ts in block_to_tsets[b_in]:
                            cover_counts[ts] += 1
                        
                        # Replace in list
                        current_solution[b_out_idx] = b_in
                        
                        # This swap is size-neutral, but might allow future removals
                        # Mark as improved to reset counter? No, size didn't change.
                        # But it's a move.
                        improved = True # To prevent immediate restart, but not to update best_size
                        # Actually, we want to run Phase 1 again after a swap.
                        
        if not improved:
            # Stuck. Try random restart of the local search state?
            # Or just break and try a different greedy initialization?
            # For now, break out of local search for this initialization.
            break
            
    # If we have time, try another greedy initialization with randomization
    # But for now, return best found.
    
    # Final check: ensure best_solution is valid
    # The evaluator checks validity.
    
    # Convert indices to actual blocks
    result = [list(all_blocks[i]) for i in best_solution]
    
    return result
=======
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    # Precompute all k-blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    all_tsets = list(itertools.combinations(range(v), t))
    
    num_blocks = len(all_blocks)
    num_tsets = len(all_tsets)
    
    # Map tset to index
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Map block to list of tset indices it covers
    block_to_tsets = []
    for block in all_blocks:
        covered = [tset_index[ts] for ts in itertools.combinations(block, t)]
        block_to_tsets.append(covered)
        
    # Map tset to list of block indices that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets in enumerate(block_to_tsets):
        for ts_idx in tsets:
            tset_to_blocks[ts_idx].append(b_idx)

    # Precompute tset_counts
    tset_counts = np.zeros(num_tsets, dtype=np.int32)
    for tsets in block_to_tsets:
        for ts_idx in tsets:
            tset_counts[ts_idx] += 1
            
    safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
    tset_weights = 1.0 / safe_counts

    def greedy_set_cover(rng_seed=None, deadline_margin=15.0):
        """Run greedy set cover with optional randomization."""
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
            
        # For randomized greedy, we can perturb weights slightly
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx, tsets in enumerate(block_to_tsets):
            if tsets:
                gains[b_idx] = np.sum(tset_weights[tsets])
        
        # Add small random perturbation to break ties and explore different solutions
        if rng_seed is not None:
            perturbation = rng.uniform(0, 0.01)
            gains = gains * (1.0 + perturbation * (2 * rng.random(num_blocks) - 1))
        
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected_blocks = []
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        # Greedy Loop with numpy argmax
        while uncovered_count > 0 and elapsed() < deadline - deadline_margin:
            # Find best block using numpy
            masked_gains = np.where(selected_mask, -np.inf, gains)
            best_idx = int(np.argmax(masked_gains))
            best_gain = masked_gains[best_idx]
            
            if best_gain <= 0:
                break
            
            selected_blocks.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and gains
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    # Update gains for all blocks covering this tset
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
                            
        return selected_blocks

    def local_search(initial_solution, deadline_margin=5.0):
        """Perform local search to minimize solution size."""
        current_solution = list(initial_solution)
        
        def get_cover_counts(sol):
            counts = np.zeros(num_tsets, dtype=np.int32)
            for b in sol:
                for ts in block_to_tsets[b]:
                    counts[ts] += 1
            return counts
        
        cover_counts = get_cover_counts(current_solution)
        best_solution = list(current_solution)
        best_size = len(best_solution)
        
        no_improvement_count = 0
        
        while elapsed() < deadline - deadline_margin:
            improved = False
            
            # Phase 1: Remove redundant blocks
            random.shuffle(current_solution)
            
            for i in range(len(current_solution) - 1, -1, -1):
                b = current_solution[i]
                can_remove = True
                for ts in block_to_tsets[b]:
                    if cover_counts[ts] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for ts in block_to_tsets[b]:
                        cover_counts[ts] -= 1
                    current_solution.pop(i)
                    improved = True
                    if len(current_solution) < best_size:
                        best_size = len(current_solution)
                        best_solution = list(current_solution)
            
            if improved:
                no_improvement_count = 0
                continue
            else:
                no_improvement_count += 1
                
            # Phase 2: 1-for-1 Swap
            if no_improvement_count > 5 and current_solution:
                b_out_idx = random.randint(0, len(current_solution) - 1)
                b_out = current_solution[b_out_idx]
                
                uncovered_tsets = []
                for ts in block_to_tsets[b_out]:
                    if cover_counts[ts] == 1:
                        uncovered_tsets.append(ts)
                
                if uncovered_tsets:
                    candidate_blocks = set(tset_to_blocks[uncovered_tsets[0]])
                    candidate_blocks.discard(b_out)
                    
                    for ts in uncovered_tsets[1:]:
                        candidate_blocks.intersection_update(tset_to_blocks[ts])
                        if not candidate_blocks:
                            break
                    
                    if candidate_blocks:
                        b_in = random.choice(list(candidate_blocks))
                        
                        for ts in block_to_tsets[b_out]:
                            cover_counts[ts] -= 1
                        for ts in block_to_tsets[b_in]:
                            cover_counts[ts] += 1
                        
                        current_solution[b_out_idx] = b_in
                        improved = True
            
            # Phase 3: 2-for-1 Swap (remove 2 blocks, add 1)
            if no_improvement_count > 15 and len(current_solution) >= 2:
                # Pick two random blocks to try removing
                idx1, idx2 = random.sample(range(len(current_solution)), 2)
                b1 = current_solution[idx1]
                b2 = current_solution[idx2]
                
                # Find tsets that would become uncovered
                # A tset is critical if it's covered only by b1 or b2 (not both)
                # Actually, a tset becomes uncovered if its count drops to 0
                # Count how many of b1, b2 cover each tset
                critical_tsets = []
                for ts in block_to_tsets[b1]:
                    if cover_counts[ts] == 1:
                        critical_tsets.append(ts)
                    elif cover_counts[ts] == 2 and ts in block_to_tsets[b2]:
                        # This tset is covered by both b1 and b2, so removing both would uncover it
                        critical_tsets.append(ts)
                
                # Also check tsets only in b2
                for ts in block_to_tsets[b2]:
                    if cover_counts[ts] == 1 and ts not in block_to_tsets[b1]:
                        critical_tsets.append(ts)
                    elif cover_counts[ts] == 2 and ts in block_to_tsets[b1]:
                        # Already counted above
                        pass
                
                if critical_tsets:
                    # Find a block that covers all critical_tsets
                    # Use the first critical tset to get candidates
                    candidate_blocks = set(tset_to_blocks[critical_tsets[0]])
                    candidate_blocks.discard(b1)
                    candidate_blocks.discard(b2)
                    
                    for ts in critical_tsets[1:]:
                        candidate_blocks.intersection_update(tset_to_blocks[ts])
                        if not candidate_blocks:
                            break
                    
                    if candidate_blocks:
                        b_in = random.choice(list(candidate_blocks))
                        
                        # Perform 2-for-1 swap
                        for ts in block_to_tsets[b1]:
                            cover_counts[ts] -= 1
                        for ts in block_to_tsets[b2]:
                            cover_counts[ts] -= 1
                        for ts in block_to_tsets[b_in]:
                            cover_counts[ts] += 1
                        
                        # Remove b1 and b2 from current_solution
                        # Be careful with indices
                        if idx1 < idx2:
                            current_solution.pop(idx2)
                            current_solution.pop(idx1)
                        else:
                            current_solution.pop(idx1)
                            current_solution.pop(idx2)
                        
                        current_solution.append(b_in)
                        
                        if len(current_solution) < best_size:
                            best_size = len(current_solution)
                            best_solution = list(current_solution)
                        
                        improved = True
            
            if not improved:
                break
                
        return best_solution

    # Run multiple greedy restarts with local search
    best_overall = None
    best_size_overall = float('inf')
    
    # First, run deterministic greedy
    greedy_result = greedy_set_cover(rng_seed=None, deadline_margin=20.0)
    if greedy_result:
        ls_result = local_search(greedy_result, deadline_margin=10.0)
        if len(ls_result) < best_size_overall:
            best_size_overall = len(ls_result)
            best_overall = ls_result
    
    # Then, run randomized greedy restarts if time permits
    max_restarts = 5
    for restart in range(max_restarts):
        if elapsed() > deadline - 25.0:
            break
            
        seed = restart * 12345 + 678
        greedy_result = greedy_set_cover(rng_seed=seed, deadline_margin=20.0)
        if greedy_result:
            ls_result = local_search(greedy_result, deadline_margin=10.0)
            if len(ls_result) < best_size_overall:
                best_size_overall = len(ls_result)
                best_overall = ls_result
    
    if best_overall is None:
        # Fallback: return empty or minimal valid solution
        return []
    
    result = [list(all_blocks[i]) for i in best_overall]
    return result
>>>>>>> REPLACE
