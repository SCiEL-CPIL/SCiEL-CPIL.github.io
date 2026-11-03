import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Precompute block coverage as numpy arrays of indices
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(np.array(covered, dtype=np.int32))
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper to get the set of t-subsets covered by a block
    # We use lists for block_cover, but sets for fast intersection in CSP
    # Converting to sets is expensive, so we keep lists and use numpy masks or manual checks
    # For CSP, we need "uncovered" set operations.
    
    # 2. Initial Solution via Greedy (Fast)
    def fast_greedy(rng):
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        while num_uncovered > 0:
            # Pick random uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            ti_target = rng.choice(uncovered_indices.tolist())
            
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    def prune(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return []
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    # 3. Probabilistic Annealing Strategy
    # Instead of exact CSP backtracking which explodes, we use Simulated Annealing.
    # State: Set of selected blocks (represented by indices).
    # Energy: lambda * (num_blocks) + penalty * (num_uncovered_t_subsets)
    # We want to minimize this. Since we need a valid cover, we heavily penalize uncovered sets initially,
    # then reduce the penalty to allow "invalid" intermediate states that might lead to a better valid state.
    
    # However, a simpler and often more effective approach for set cover is:
    # Start with a valid cover.
    # Move: Swap a block in the cover with a block not in the cover.
    #   - If the new cover is valid and has fewer blocks (or same size but better structure), accept.
    #   - If the new cover is invalid, accept with probability exp(-delta_E / T).
    #   - If the new cover is valid but same size, accept with probability exp(-delta_E / T) where delta_E is 0? 
    #     Actually, we want to minimize block count. 
    #     Let's define Energy = num_blocks + M * num_uncovered.
    #     M is a large penalty.
    #     If we swap block A (in) with B (out):
    #       num_blocks stays same.
    #       num_uncovered changes by (new_uncovered - old_uncovered).
    #       delta_E = M * (new_uncovered - old_uncovered).
    #     If new_uncovered < old_uncovered, delta_E < 0, always accept.
    #     If new_uncovered > old_uncovered, delta_E > 0, accept with prob exp(-delta_E/T).
    #     If new_uncovered == old_uncovered, delta_E = 0, accept with prob 1 (random walk to escape local minima).
    
    # But we also want to reduce num_blocks.
    # Move 2: Remove a block.
    #   If the resulting cover is still valid, num_blocks decreases by 1. Always accept.
    #   If invalid, num_uncovered increases. delta_E = -1 + M * (new_uncovered - 0).
    #   Accept with prob exp(-delta_E/T).
    
    # Move 3: Add a block.
    #   num_blocks increases by 1.
    #   num_uncovered decreases by gain.
    #   delta_E = 1 - M * gain.
    #   If gain is high enough, delta_E < 0, always accept.
    
    # This formulation allows the algorithm to explore invalid states (with some uncovered sets)
    # if it helps in finding a configuration with fewer blocks later.
    
    def simulated_annealing(initial_blocks, time_limit, rng):
        current_blocks = list(initial_blocks)
        current_counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                current_counts[ti] += 1
        
        current_uncovered = np.sum(current_counts == 0)
        current_blocks_set = set(current_blocks)
        
        # Initial Temperature
        # M should be large enough to prioritize validity, but not so large that we can't escape.
        # Let's say M = 1000.
        M = 1000.0
        T_start = M * 10.0
        T_end = 0.1
        cooling_rate = 0.995 # Tune this
        
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        best_uncovered = current_uncovered
        
        start_time = time.time()
        
        while time.time() < time_limit:
            # Generate Move
            move_type = rng.random()
            if move_type < 0.4:
                # Swap
                if not current_blocks:
                    continue
                bi_old = rng.choice(current_blocks)
                # Pick a random block not in current set?
                # Or pick a block that covers some uncovered sets?
                # To be efficient, pick a block that shares at least one t-subset with bi_old?
                # Or just random from all blocks?
                # Random from all blocks is slow.
                # Let's pick a block that covers at least one of the t-subsets covered by bi_old.
                # This creates a "local" move.
                
                # Get t-subsets covered by bi_old
                # We want a block bi_new such that bi_new != bi_old
                # And ideally bi_new covers some of the unique t-subsets of bi_old?
                # No, if bi_new covers unique t-subsets of bi_old, it might not be a good swap if bi_old covers others.
                
                # Let's just pick a random block from the whole space? 
                # N_b is ~75582. That's small.
                bi_new = rng.randrange(N_b)
                if bi_new in current_blocks_set:
                    continue
                
                # Calculate delta
                # Remove bi_old, Add bi_new
                # Delta counts:
                # For ti in block_cover[bi_old]: counts[ti] -= 1
                # For ti in block_cover[bi_new]: counts[ti] += 1
                
                # We need to know how many t-subsets become uncovered or covered.
                # Let's calculate new_uncovered without actually modifying the array, or do it and revert.
                # Since N_t is 11628, and block size is 56, it's fast to modify and revert.
                
                # Save old counts
                # Actually, let's just compute delta_uncovered directly.
                
                # t-subsets that are covered ONLY by bi_old (among current blocks) will become uncovered.
                # t-subsets that are NOT covered by current blocks (except maybe bi_new) will become covered.
                
                # This is complex to compute analytically without modifying counts.
                # Let's modify, check, revert if rejected.
                
                # Apply
                for ti in block_cover[bi_old]:
                    current_counts[ti] -= 1
                for ti in block_cover[bi_new]:
                    current_counts[ti] += 1
                    
                new_uncovered = np.sum(current_counts == 0)
                
                # Delta Energy
                # Energy = len(blocks) + M * uncovered
                # len(blocks) is same.
                delta_E = M * (new_uncovered - current_uncovered)
                
                accept = False
                if delta_E < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta_E / T)
                    if T > 0:
                        prob = np.exp(-delta_E / T)
                        if rng.random() < prob:
                            accept = True
                    else:
                        accept = False # At T=0, only accept improvements
                
                if accept:
                    current_blocks[current_blocks.index(bi_old)] = bi_new
                    current_blocks_set.remove(bi_old)
                    current_blocks_set.add(bi_new)
                    current_uncovered = new_uncovered
                    
                    # Update best
                    if current_uncovered == 0:
                        if len(current_blocks) < best_size:
                            best_size = len(current_blocks)
                            best_blocks = list(current_blocks)
                else:
                    # Revert
                    for ti in block_cover[bi_old]:
                        current_counts[ti] += 1
                    for ti in block_cover[bi_new]:
                        current_counts[ti] -= 1
                        
            elif move_type < 0.7:
                # Remove
                if not current_blocks:
                    continue
                idx = rng.randrange(len(current_blocks))
                bi_old = current_blocks[idx]
                
                # Apply removal
                for ti in block_cover[bi_old]:
                    current_counts[ti] -= 1
                    
                new_uncovered = np.sum(current_counts == 0)
                
                # Delta Energy
                # len(blocks) decreases by 1.
                # uncovered increases by (new_uncovered - current_uncovered)
                delta_E = -1.0 + M * (new_uncovered - current_uncovered)
                
                accept = False
                if delta_E < 0:
                    accept = True
                else:
                    if T > 0:
                        prob = np.exp(-delta_E / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    current_blocks.pop(idx)
                    current_blocks_set.remove(bi_old)
                    current_uncovered = new_uncovered
                    
                    if current_uncovered == 0:
                        if len(current_blocks) < best_size:
                            best_size = len(current_blocks)
                            best_blocks = list(current_blocks)
                else:
                    # Revert
                    for ti in block_cover[bi_old]:
                        current_counts[ti] += 1
                        
            else:
                # Add
                # Pick a random block not in set
                bi_new = rng.randrange(N_b)
                if bi_new in current_blocks_set:
                    continue
                
                # Apply addition
                for ti in block_cover[bi_new]:
                    current_counts[ti] += 1
                    
                new_uncovered = np.sum(current_counts == 0)
                
                # Delta Energy
                # len(blocks) increases by 1.
                # uncovered decreases by (current_uncovered - new_uncovered)
                delta_E = 1.0 - M * (current_uncovered - new_uncovered)
                
                accept = False
                if delta_E < 0:
                    accept = True
                else:
                    if T > 0:
                        prob = np.exp(-delta_E / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    current_blocks.append(bi_new)
                    current_blocks_set.add(bi_new)
                    current_uncovered = new_uncovered
                    
                    # We don't update best here because we added a block, so size increased.
                    # Unless it fixed validity? But best is tracked on valid states.
                    
                else:
                    # Revert
                    for ti in block_cover[bi_new]:
                        current_counts[ti] -= 1
            
            # Cooling
            T *= cooling_rate
            
        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multi-start Greedy to get a good initial upper bound
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        rng = random.Random(123 + i)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(blocks)
        best_len = len(best_solution)

    # Phase 2: Simulated Annealing
    # Allocate remaining time
    time_remaining = deadline - 5.0 - time.time()
    if time_remaining > 1.0 and best_solution:
        # Run SA with the best solution found
        rng = random.Random(42)
        # We might run multiple SA restarts if time permits
        # But SA is stochastic, so one long run is often better than many short ones.
        # Let's give it most of the time.
        
        # To improve SA, we can use a "restart" strategy if it gets stuck.
        # But for simplicity, one long run.
        
        # We need to be careful with the "best" tracking in SA.
        # The SA function returns the best valid solution found during the run.
        
        sa_solution = simulated_annealing(best_solution, deadline - 5.0, rng)
        
        if sa_solution and len(sa_solution) < best_len:
            best_len = len(sa_solution)
            best_solution = sa_solution

    # Phase 3: Final Pruning (SA might leave redundant blocks if it added them and didn't remove all)
    if best_solution:
        best_solution = prune(best_solution)
        best_len = len(best_solution)

    return [list(all_blocks[bi]) for bi in best_solution]
