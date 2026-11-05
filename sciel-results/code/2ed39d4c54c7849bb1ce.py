import time
import numpy as np
import random
import itertools

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs) using uint8
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    # Also build reverse map: for each tsub, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_tsubs)]
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1
            t_to_blocks[idx].append(i)
    
    t_to_blocks = [np.array(lst, dtype=np.int32) for lst in t_to_blocks]

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Helper: Greedy construction
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            # Gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            # Update uncovered
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    # Helper: Remove redundant blocks
    def remove_redundant(indices):
        indices_set = set(indices)
        # Calculate counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            # Find blocks that are redundant
            # A block is redundant if for all tsubs it covers, count > 1
            # We iterate and check
            to_remove = []
            for idx in indices_set:
                row = M[idx]
                covered = np.where(row == 1)[0]
                # If all covered tsubs have count > 1, it's redundant
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set)

    # Helper: Try to swap blocks to reduce count
    # This is the core local search step
    def local_search(indices):
        indices_set = set(indices)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        improved = True
        while improved:
            improved = False
            # First pass: remove redundant
            new_indices = remove_redundant(list(indices_set))
            if len(new_indices) < len(indices_set):
                indices_set = set(new_indices)
                # Recalculate counts
                counts = np.zeros(num_tsubs, dtype=np.int32)
                for idx in indices_set:
                    counts += M[idx]
                improved = True
                continue
            
            # Second pass: try swaps
            # For each block, check if it can be swapped
            # To make it faster, we only try a subset or random order
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.monotonic() > deadline - 5.0:
                    return list(indices_set)
                
                # Critical tsubs: those covered ONLY by b_idx
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                critical_mask = counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # Redundant, should have been removed
                    indices_set.remove(b_idx)
                    counts -= row
                    improved = True
                    break
                
                # We need a block that covers all critical_indices
                # And is not in indices_set
                # Check M[:, critical_indices].all(axis=1)
                # This is expensive if done for every block.
                # Optimization: Only check if critical_indices is small?
                # Or use a precomputed list of blocks for each tsub?
                
                # Let's try to find candidates
                # Intersection of sets of blocks covering each critical tsub
                # Since M is dense, we can do:
                # candidates_mask = M[:, critical_indices].all(axis=1)
                # candidates_mask &= ~np.isin(np.arange(num_kblocks), list(indices_set))
                
                # To speed up, we can just pick a random candidate from the union? No, must cover ALL.
                # Let's just do the numpy op, it's vectorized.
                if len(critical_indices) > 50:
                    # Too many constraints, unlikely to find a single block covering all
                    # Skip this block for swap
                    continue
                    
                candidates_mask = M[:, critical_indices].all(axis=1)
                # Exclude current blocks
                # Creating a boolean array of length num_kblocks every time is slow.
                # We can maintain a 'selected' boolean array.
                # But we are in a loop.
                
                # Let's optimize by maintaining selected_mask
                # This requires passing it around.
                
                # For now, let's just check candidates
                candidates = np.where(candidates_mask)[0]
                # Filter out current
                current_list = list(indices_set)
                current_set = set(current_list)
                candidates = [c for c in candidates if c not in current_set]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    c_idx = candidates[rng.randrange(len(candidates))]
                    
                    # Perform swap
                    indices_set.remove(b_idx)
                    counts -= M[b_idx]
                    indices_set.add(c_idx)
                    counts += M[c_idx]
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 1: Initial Greedy
    initial_sol = greedy_construction()
    if initial_sol:
        best_solution = initial_sol
        best_len = len(initial_sol)

    # Phase 2: Simulated Annealing
    # State: set of blocks
    # Moves: swap a random block with a candidate that maximizes marginal coverage gain
    # Accept worse solutions with temperature-dependent probability
    
    current_sol = best_solution[:]
    current_set = set(current_sol)
    current_counts = np.zeros(num_tsubs, dtype=np.int32)
    for idx in current_set:
        current_counts += M[idx]
    
    # Temperature schedule
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    
    sa_iterations = 0
    max_sa_iters = 5000
    
    while T > T_end and sa_iterations < max_sa_iters:
        if time.monotonic() > deadline - 10.0:
            break
        
        sa_iterations += 1
        
        # Pick a random block to remove
        if not current_set:
            break
            
        old_block = list(current_set)[rng.randrange(len(current_set))]
        old_row = M[old_block]
        old_covered = np.where(old_row == 1)[0]
        
        # Find critical tsubs (covered only by old_block)
        critical_mask = current_counts == 1
        critical_indices = old_covered[critical_mask[old_covered]]
        
        # If no critical tsubs, just remove
        if len(critical_indices) == 0:
            current_set.remove(old_block)
            current_counts -= old_row
            if len(current_set) < best_len:
                best_len = len(current_set)
                best_solution = list(current_set)
            T *= cooling_rate
            continue
        
        # Try to find a replacement block that covers all critical tsubs
        # and maximizes marginal gain
        if len(critical_indices) > 30:
            # Too many constraints, unlikely to find a good replacement
            T *= cooling_rate
            continue
            
        # Find candidates that cover all critical tsubs
        candidates_mask = M[:, critical_indices].all(axis=1)
        candidates = np.where(candidates_mask)[0]
        candidates = [c for c in candidates if c not in current_set]
        
        if len(candidates) == 0:
            T *= cooling_rate
            continue
        
        # Pick the candidate with maximum marginal gain
        # Marginal gain = number of tsubs covered by candidate that are currently uncovered or under-covered
        best_candidate = None
        best_gain = -1
        
        # Sample up to 20 candidates for speed
        sample_size = min(20, len(candidates))
        sampled = rng.sample(candidates, sample_size)
        
        for cand in sampled:
            cand_row = M[cand]
            cand_covered = np.where(cand_row == 1)[0]
            
            # Gain: how many of cand's covered tsubs are currently uncovered (count == 0)
            # or under-covered (count == 1, but not critical for old_block)
            # Actually, for a valid swap, we need cand to cover all critical tsubs
            # The gain is the number of tsubs where count increases from 0 to 1
            gain = np.sum(current_counts[cand_covered] == 0)
            
            # Penalty for creating new critical tsubs
            # After swap, some tsubs might become critical
            # This is complex to compute exactly, so use a heuristic
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate is None:
            T *= cooling_rate
            continue
            
        # Compute the energy change
        # Energy = number of blocks (we want to minimize)
        # For a swap, energy change is 0 (same number of blocks)
        # But we want to maximize coverage quality
        # Use a soft energy: -sum of log(count+1) or similar
        # For simplicity, use: energy = -total_coverage + lambda * num_blocks
        # Since num_blocks is constant for swaps, energy change is based on coverage
        
        # Actually, let's use a simpler acceptance criterion:
        # Accept if the new solution is valid (covers all tsubs) and has same or fewer blocks
        # For swaps, block count is same, so accept if coverage doesn't decrease
        
        # Check if the swap maintains coverage
        # After removing old_block, some tsubs might become uncovered
        # After adding best_candidate, they should be covered again
        # Since best_candidate covers all critical tsubs, coverage is maintained
        
        # Accept the swap
        current_set.remove(old_block)
        current_counts -= old_row
        current_set.add(best_candidate)
        current_counts += M[best_candidate]
        
        if len(current_set) < best_len:
            best_len = len(current_set)
            best_solution = list(current_set)
        
        T *= cooling_rate
    
    # Phase 3: Final local search polish
    if time.monotonic() < deadline - 5.0:
        polished_sol = local_search(best_solution)
        if len(polished_sol) < best_len:
            best_len = len(polished_sol)
            best_solution = polished_sol

    # Final reduction
    best_solution = remove_redundant(best_solution)
    
    # Convert to output
    return get_blocks(best_solution)
