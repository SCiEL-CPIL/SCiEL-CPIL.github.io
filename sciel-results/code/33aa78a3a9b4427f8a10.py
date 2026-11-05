import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    # The evaluator kills after 150s. We use a safety margin.
    deadline = start_time + 140.0

    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute coverage: for each block, list of t-subset indices it covers
    # This is expensive to store as lists of lists if we do it naively, 
    # but C(19,7) = 50388 blocks and C(7,4) = 35 subsets per block.
    # Total entries ~ 1.7M. This is manageable in memory as a list of arrays.
    block_coverage = []
    for block in all_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in subs], dtype=np.int32)
        block_coverage.append(indices)
    
    # Helper to check if a set of blocks covers all t-subsets
    def is_valid_cover(block_indices):
        if not block_indices:
            return False
        covered = np.zeros(num_t_subsets, dtype=np.bool_)
        for b_idx in block_indices:
            covered[block_coverage[b_idx]] = True
        return np.all(covered)
    
    # Helper to remove redundant blocks
    # Returns a list of non-redundant blocks
    def reduce_cover(block_indices):
        if not block_indices:
            return []
        
        # Count coverage for each t-subset
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in block_indices:
            cover_counts[block_coverage[b_idx]] += 1
            
        # Iteratively remove blocks that don't cover any unique t-subset
        # A block is redundant if for all t-subsets it covers, count > 1
        changed = True
        current_indices = list(block_indices)
        while changed:
            changed = False
            to_remove = []
            for i, b_idx in enumerate(current_indices):
                cov = block_coverage[b_idx]
                # If any t-subset covered by this block has count 1, it's essential
                # Otherwise, it's redundant
                if np.any(cover_counts[cov] == 1):
                    continue
                # It is redundant
                to_remove.append(i)
            
            if to_remove:
                # Remove them
                remove_set = set(to_remove)
                for i in to_remove:
                    b_idx = current_indices[i]
                    cover_counts[block_coverage[b_idx]] -= 1
                current_indices = [b for i, b in enumerate(current_indices) if i not in remove_set]
                changed = True
                
        return current_indices

    # Phase 1: Greedy Set Cover (Deterministic baseline)
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 60.0:
            break
            
        # Calculate gain for each block
        # Gain = number of uncovered t-subsets covered by this block
        # We can approximate this or compute exactly.
        # Computing exactly:
        gains = np.zeros(num_blocks, dtype=np.int32)
        for b_idx in range(num_blocks):
            cov = block_coverage[b_idx]
            gains[b_idx] = np.sum(active[cov])
            
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = np.argmax(gains)
        selected_block_indices.append(b_idx)
        
        covered_indices = block_coverage[b_idx]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Reduce the greedy solution
    greedy_solution = reduce_cover(selected_block_indices)
    best_solution = greedy_solution
    best_size = len(greedy_solution)
    
    # Phase 2: Probabilistic Method with Alteration
    # Try several random probabilities p
    # Expected number of blocks = C(v,k) * p
    # Probability a specific t-subset is uncovered = (1-p)^{C(v-t, k-t)}
    # For v=19, k=7, t=4: C(15, 3) = 455.
    # If p=0.01, expected blocks ~ 504. Uncovered prob ~ (0.99)^455 ~ 0.01.
    # If p=0.005, expected blocks ~ 252. Uncovered prob ~ (0.995)^455 ~ 0.10.
    # We want to start with a cover that is likely valid or close to valid.
    
    p_values = [0.01, 0.015, 0.02, 0.03]
    
    for p in p_values:
        if time.time() > deadline - 20.0:
            break
            
        # Run multiple trials for this p
        num_trials = 5
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
                
            # Randomly select blocks with probability p
            # Use numpy random for speed
            rand_vals = np.random.random(num_blocks)
            selected = np.where(rand_vals < p)[0].tolist()
            
            # Check if it's a valid cover
            if is_valid_cover(selected):
                # Reduce it
                reduced = reduce_cover(selected)
                if len(reduced) < best_size:
                    best_solution = reduced
                    best_size = len(best_solution)
            else:
                # If not valid, we can try to repair it by adding blocks
                # This is similar to greedy but starting from a random seed
                # However, the directive says "selecting a random subset... then systematically removing redundant blocks".
                # If the random subset is not a cover, we can't just remove blocks.
                # We need to add blocks to make it a cover first.
                # Let's do a "Greedy Repair" on the random set.
                
                # Start with the random set
                current_cover = set(selected)
                cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                for b_idx in current_cover:
                    cover_counts[block_coverage[b_idx]] += 1
                
                # Find uncovered t-subsets
                uncovered = np.where(cover_counts == 0)[0]
                
                # Add blocks to cover uncovered subsets
                # We can use a simple greedy approach on the uncovered sets
                # Or just keep adding random blocks until covered?
                # Greedy repair is better.
                
                while len(uncovered) > 0:
                    if time.time() > deadline - 10.0:
                        break
                    
                    # For each block, count how many uncovered t-subsets it covers
                    gains = np.zeros(num_blocks, dtype=np.int32)
                    for b_idx in range(num_blocks):
                        if b_idx in current_cover:
                            continue
                        cov = block_coverage[b_idx]
                        # Count intersection with uncovered
                        # This is slow if done in Python loop.
                        # Optimization: 
                        # We can just pick a random block that covers at least one uncovered subset.
                        # Or, to be more efficient, we can precompute which blocks cover which t-subsets.
                        # But let's stick to a simple heuristic:
                        # Pick a random uncovered t-subset, and pick a random block that covers it.
                        
                        # Find a random uncovered t-subset
                        u_idx = uncovered[random.randint(0, len(uncovered)-1)]
                        
                        # Find all blocks that cover u_idx
                        # We need a reverse index: t_subset -> list of blocks
                        # This is expensive to build.
                        # Alternative: Just iterate through all blocks and find one that covers u_idx and is not in current_cover.
                        # Since num_blocks is 50k, this is fast enough for a few iterations.
                        
                        candidate = -1
                        for b_idx in range(num_blocks):
                            if b_idx in current_cover:
                                continue
                            if u_idx in block_coverage[b_idx]:
                                candidate = b_idx
                                break
                                
                        if candidate == -1:
                            # Should not happen if cover is possible
                            break
                            
                        # Add candidate
                        current_cover.add(candidate)
                        cov = block_coverage[candidate]
                        cover_counts[cov] += 1
                        
                        # Update uncovered
                        # Any t-subset in cov that was 0 is now 1
                        # Any t-subset in cov that was 1 is now 2 (still covered)
                        # We just need to update the counts and re-evaluate uncovered
                        # Actually, we can just update cover_counts and then find new uncovered
                        # This is O(num_t_subsets) per addition.
                        # num_t_subsets is 3876.
                        
                        # More efficient:
                        # For each t-subset in cov:
                        #   if cover_counts[t_sub] was 0, it is now 1.
                        #   if it was 1, it is now 2.
                        # We need to know which ones became covered.
                        
                        # Let's just recalculate uncovered if the set is small, or use a set.
                        # For simplicity and speed, let's just recalculate if we add few blocks.
                        # Or, better:
                        # We can maintain a set of uncovered indices.
                        
                        pass # Placeholder for efficient update, will use full recalc for simplicity in this draft
                        # Actually, let's just use the full recalc method for correctness in the repair loop,
                        # but limit the number of repair steps.
                        
                        # Recalculate uncovered
                        uncovered = np.where(cover_counts == 0)[0]
                        
                # After repair, we have a valid cover
                if len(uncovered) == 0:
                    # Reduce it
                    reduced = reduce_cover(list(current_cover))
                    if len(reduced) < best_size:
                        best_solution = reduced
                        best_size = len(reduced)

    # Phase 3: Local Search Refinement on the best solution found
    # Try 1-for-1 swaps to reduce size
    if time.time() < deadline - 5.0:
        best_solution_set = set(best_solution)
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in best_solution:
            cover_counts[block_coverage[b_idx]] += 1
            
        current_solution = list(best_solution)
        current_size = len(current_solution)
        
        # We will try to improve current_solution
        # If we find a smaller one, update best_solution
        
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not current_solution:
                break
                
            # Pick a random block to remove
            i = random.randint(0, len(current_solution) - 1)
            b_remove = current_solution[i]
            
            cov_remove = block_coverage[b_remove]
            
            # Find critical t-subsets (only covered by b_remove)
            critical = []
            for c in cov_remove:
                if cover_counts[c] == 1:
                    critical.append(c)
            
            if not critical:
                # Redundant, remove
                current_solution.pop(i)
                best_solution_set.remove(b_remove)
                cover_counts[cov_remove] -= 1
                current_size -= 1
                if current_size < best_size:
                    best_solution = list(current_solution)
                    best_size = current_size
                continue
            
            # Try to find a block that covers all critical t-subsets
            # Sample a few random blocks
            found_swap = False
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                b_add = random.randint(0, num_blocks - 1)
                if b_add in best_solution_set:
                    continue
                
                cov_add = block_coverage[b_add]
                # Check if b_add covers all critical
                # Critical is a list of indices
                # We can use numpy: np.all(np.isin(critical, cov_add))
                if np.all(np.isin(critical, cov_add)):
                    # Perform swap
                    current_solution[i] = b_add
                    best_solution_set.remove(b_remove)
                    best_solution_set.add(b_add)
                    
                    cover_counts[cov_remove] -= 1
                    cover_counts[cov_add] += 1
                    
                    found_swap = True
                    break
            
            if not found_swap:
                # Perturbation: Add a random block
                if current_size < best_size + 2:
                    b_add = random.randint(0, num_blocks - 1)
                    if b_add not in best_solution_set:
                        current_solution.append(b_add)
                        best_solution_set.add(b_add)
                        cover_counts[block_coverage[b_add]] += 1
                        current_size += 1

    # Final reduction
    if best_solution:
        best_solution = reduce_cover(best_solution)
        
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    return blocks
