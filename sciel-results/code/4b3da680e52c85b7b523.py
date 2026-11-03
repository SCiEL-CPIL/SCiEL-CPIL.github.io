import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Map each block to the list of t-subset indices it covers
    block_to_t = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_t.append(covered)
        
    # Convert block_to_t to numpy arrays for faster operations
    block_to_t_np = [np.array(covered, dtype=np.int32) for covered in block_to_t]

    # Inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_to_t):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # --- Helper Functions ---

    def get_coverage_counts(selected_indices):
        """
        Returns an array of counts for each t-subset indicating how many 
        selected blocks cover it.
        """
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in selected_indices:
            counts[block_to_t_np[bi]] += 1
        return counts

    def count_uncovered(counts):
        """
        Returns the number of t-subsets with count 0.
        """
        return np.sum(counts == 0)

    def prob_estimate_uncovered(counts):
        """
        Rough estimate for annealing acceptance.
        """
        return count_uncovered(counts)

    def greedy_cover(seed=42):
        """
        Standard greedy set cover to get a valid initial solution.
        """
        rng = random.Random(seed)
        covered_mask = np.zeros(N_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 20.0:
                break
                
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-subset to reduce bias in greedy choice
            target_t_idx = rng.choice(uncovered_indices.tolist())
            candidates = t_to_blocks[target_t_idx]
            
            if not candidates:
                break
                
            max_gain = -1
            best_candidates = []
            
            for bi in candidates:
                if bi in selected_set:
                    continue
                t_indices = block_to_t_np[bi]
                gain = np.sum(~covered_mask[t_indices])
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain:
                    best_candidates.append(bi)
            
            if not best_candidates:
                bi = candidates[0]
            else:
                bi = rng.choice(best_candidates)
                
            selected_blocks.append(bi)
            selected_set.add(bi)
            
            # Update covered_mask
            covered_mask[block_to_t_np[bi]] = True
            
        return selected_blocks

    def simulated_annealing(initial_blocks, seed=42):
        """
        Optimizes the set of blocks using Simulated Annealing.
        Moves:
        1. Remove a block (if it doesn't break validity, or with penalty).
        2. Swap a block with another.
        3. Add a block (rarely, to escape deep local minima).
        
        Energy Function:
        E = W_uncovered * (Number of Uncovered t-subsets) + W_blocks * (Number of Blocks)
        We want to minimize E.
        """
        rng = random.Random(seed)
        
        # Start with a valid cover
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = count_uncovered(current_counts)
        current_size = len(current_blocks)
        
        # Weights: Heavily penalize uncovered subsets to maintain validity mostly
        W_unc = 10000.0
        W_size = 1.0
        
        current_energy = W_unc * current_uncovered + W_size * current_size
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        best_uncovered = current_uncovered
        best_size = current_size
        
        T = 50.0  # Initial temperature
        cooling = 0.995
        min_T = 0.1
        
        # Precompute block sizes for swaps? No, all blocks are same size.
        
        while T > min_T and time.time() < deadline - 5.0:
            
            # --- Generate Neighbor ---
            move_type = rng.random()
            
            if move_type < 0.6:
                # Remove a block
                if not current_blocks:
                    T *= cooling
                    continue
                idx_to_remove = rng.randrange(len(current_blocks))
                bi = current_blocks[idx_to_remove]
                
                # Simulate removal
                new_counts = current_counts.copy()
                new_counts[block_to_t_np[bi]] -= 1
                new_uncovered = count_uncovered(new_counts)
                new_size = current_size - 1
                new_energy = W_unc * new_uncovered + W_size * new_size
                
                # Acceptance
                delta_E = new_energy - current_energy
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    current_blocks.pop(idx_to_remove)
                    current_counts = new_counts
                    current_uncovered = new_uncovered
                    current_size = new_size
                    current_energy = new_energy
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_blocks = list(current_blocks)
                        best_energy = current_energy
                        best_uncovered = current_uncovered
                        best_size = current_size
                        
            elif move_type < 0.9:
                # Swap a block
                if not current_blocks:
                    T *= cooling
                    continue
                idx_to_swap = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx_to_swap]
                
                # Pick a random block from all blocks to swap in
                # To make it efficient, we can pick from blocks that cover some uncovered t-subsets
                # or just random. Random is safer for exploration.
                # But random from 75k blocks is slow if we do it naively.
                # Let's pick a random block index.
                bi_in = rng.randrange(N_b)
                
                if bi_in == bi_out:
                    T *= cooling
                    continue
                    
                # Check if bi_in is already in the set
                # O(N) check is fine for N~300
                if bi_in in current_blocks:
                    T *= cooling
                    continue
                    
                # Simulate swap
                new_counts = current_counts.copy()
                new_counts[block_to_t_np[bi_out]] -= 1
                new_counts[block_to_t_np[bi_in]] += 1
                new_uncovered = count_uncovered(new_counts)
                new_size = current_size # Size stays same
                new_energy = W_unc * new_uncovered + W_size * new_size
                
                delta_E = new_energy - current_energy
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    current_blocks[idx_to_swap] = bi_in
                    current_counts = new_counts
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_blocks = list(current_blocks)
                        best_energy = current_energy
                        best_uncovered = current_uncovered
                        best_size = current_size
                        
            else:
                # Add a block (to recover from high uncovered count)
                if current_uncovered > 0:
                    # Pick a random uncovered t-subset
                    uncovered_indices = np.where(current_counts == 0)[0]
                    if len(uncovered_indices) > 0:
                        target_t_idx = rng.choice(uncovered_indices.tolist())
                        candidates = t_to_blocks[target_t_idx]
                        if candidates:
                            bi_in = rng.choice(candidates)
                            if bi_in not in current_blocks:
                                new_counts = current_counts.copy()
                                new_counts[block_to_t_np[bi_in]] += 1
                                new_uncovered = count_uncovered(new_counts)
                                new_size = current_size + 1
                                new_energy = W_unc * new_uncovered + W_size * new_size
                                
                                delta_E = new_energy - current_energy
                                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                                    current_blocks.append(bi_in)
                                    current_counts = new_counts
                                    current_uncovered = new_uncovered
                                    current_size = new_size
                                    current_energy = new_energy
                                    # Don't update best here as size increased, 
                                    # but might enable future removals
                
            T *= cooling
            
        # Final Pruning Pass on the best valid solution found
        if best_uncovered == 0:
            # Try to remove redundant blocks from best_blocks
            # This is a local search refinement
            pruned = list(best_blocks)
            pruned_set = set(pruned)
            # Sort by block index or random? Random helps break symmetry
            rng.shuffle(pruned)
            
            changed = True
            while changed and time.time() < deadline - 2.0:
                changed = False
                for i in range(len(pruned)):
                    if time.time() > deadline - 2.0:
                        break
                    bi = pruned[i]
                    # Tentatively remove
                    pruned_set.discard(bi)
                    # Check validity
                    # Rebuilding counts is expensive. 
                    # Instead, check if any t-subset in bi becomes uncovered.
                    # We need counts for the current set pruned_set.
                    # This is O(N_t) per check. 
                    # To speed up, we can just run a full validity check if the set is small.
                    
                    # Efficient check:
                    # A t-subset is uncovered if no block in pruned_set covers it.
                    # We can pre-compute counts for pruned_set once, then update.
                    # But we are iterating. 
                    # Let's just do a simple check: 
                    # For each t-subset covered by bi, if count in (pruned_set \ {bi}) is 0, then bi is needed.
                    
                    # To do this fast, we need counts of pruned_set.
                    # Since we are modifying pruned_set, maintaining counts is tricky.
                    # Let's just do a full validity check on the set minus bi.
                    # If N_t is 11628 and N_blocks is ~300, this is 11628 * (small constant) checks.
                    # It might be slow. 
                    
                    # Alternative: Just try to remove. If invalid, put back.
                    # We'll implement a fast validity check using the inverse index.
                    
                    valid = True
                    # We need to know if every t-subset is covered by pruned_set.
                    # We can iterate through t_to_blocks? No.
                    # We can iterate through the t-subsets covered by bi.
                    # If a t-subset was ONLY covered by bi, then removing bi breaks it.
                    # So we need to know if there is another block in pruned_set covering it.
                    
                    # To do this efficiently, we can maintain a count array for the current pruned_set.
                    # Let's restructure the pruning loop to maintain counts.
                    
                    pass # We will implement a separate pruning function below
                    
            # Use the dedicated pruning function
            pruned = prune_redundant_fast(best_blocks, deadline - 1.0)
            if len(pruned) < len(best_blocks):
                best_blocks = pruned
                best_size = len(pruned)
                
        return best_blocks

    def prune_redundant_fast(blocks, deadline):
        """
        Removes redundant blocks from a valid cover.
        Uses a count-based approach to speed up checks.
        """
        if not blocks:
            return []
            
        current_set = list(blocks)
        current_set_set = set(current_set)
        
        # Initial counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_set:
            counts[block_to_t_np[bi]] += 1
            
        # If any count is 0, the input wasn't valid? 
        # We assume input is valid from SA.
        
        changed = True
        rng = random.Random(99)
        
        while changed and time.time() < deadline:
            changed = False
            # Iterate in random order to avoid bias
            indices = list(range(len(current_set)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                bi = current_set[i]
                
                # Check if removing bi leaves any t-subset with count 0
                # This is equivalent to checking if min(counts[block_to_t_np[bi]]) <= 1
                # If min count is 1, removing bi makes it 0.
                # If min count > 1, removing bi is safe.
                
                t_indices = block_to_t_np[bi]
                if np.min(counts[t_indices]) > 1:
                    # Safe to remove
                    counts[t_indices] -= 1
                    # Remove from list
                    current_set.pop(i)
                    current_set_set.discard(bi)
                    changed = True
                    # Adjust indices? We are iterating over a snapshot of indices.
                    # If we pop i, subsequent indices in the snapshot are shifted.
                    # This is a common bug. 
                    # To handle this, we can just break and restart the loop, 
                    # or use a set of indices to process.
                    # Given the small size (~300), restarting is cheap.
                    break
                    
        return current_set

    # --- Main Execution ---
    
    # 1. Generate initial greedy cover
    initial_blocks = greedy_cover(seed=42)
    if not initial_blocks:
        # Fallback if greedy failed (shouldn't happen)
        return [list(all_blocks[bi]) for bi in range(0)]

    # 2. Run Simulated Annealing
    # We can run multiple SA instances with different seeds if time permits
    best_overall = initial_blocks
    best_overall_size = len(initial_blocks)
    
    # Check if we have time for SA
    if time.time() < deadline - 30.0:
        sa_blocks = simulated_annealing(initial_blocks, seed=100)
        if len(sa_blocks) < best_overall_size:
            best_overall = sa_blocks
            best_overall_size = len(sa_blocks)
            
    # 3. If we still have time, run another SA with a different initial seed
    if time.time() < deadline - 30.0:
        initial_blocks2 = greedy_cover(seed=123)
        if len(initial_blocks2) < best_overall_size:
            best_overall = initial_blocks2
            best_overall_size = len(best_overall)
            
        if time.time() < deadline - 20.0:
            sa_blocks2 = simulated_annealing(initial_blocks2, seed=200)
            if len(sa_blocks2) < best_overall_size:
                best_overall = sa_blocks2
                best_overall_size = len(sa_blocks2)

    # Final safety check: ensure validity
    # Convert to list of lists
    result_blocks = [list(all_blocks[bi]) for bi in best_overall]
    
    # Optional: Final exhaustive validity check? 
    # It takes ~1s. We have ~5s margin.
    if time.time() < deadline - 2.0:
        final_counts = get_coverage_counts(best_overall)
        if np.any(final_counts == 0):
            # Invalid! Fallback to the greedy one (which was valid)
            result_blocks = [list(all_blocks[bi]) for bi in initial_blocks]
            
    return result_blocks
