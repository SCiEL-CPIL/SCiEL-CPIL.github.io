import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute which blocks cover each t-subset
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    block_to_t = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_t.append(covered)
    
    # Inverse: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_to_t):
        for ti in covered:
            t_to_blocks[ti].append(bi)
    
    # Helper: compute coverage counts for a given set of blocks (represented as indices)
    def get_coverage_counts(block_indices):
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in block_indices:
            for ti in block_to_t[bi]:
                counts[ti] += 1
        return counts
    
    # Helper: count uncovered t-subsets
    def count_uncovered(counts):
        return np.sum(counts == 0)
    
    # Helper: probabilistic estimate of uncovered count using random sampling
    # This is the "noisy objective" for SA
    def prob_estimate_uncovered(block_indices, sample_size=500, rng=None):
        if rng is None:
            rng = random.Random(42)
        
        # Sample random t-subsets
        samples = [rng.randrange(N_t) for _ in range(sample_size)]
        
        # Check coverage for each sample
        uncovered_count = 0
        for ti in samples:
            # Check if any block in block_indices covers ti
            # Optimization: use precomputed t_to_blocks
            if not any(bi in block_indices_set for bi in t_to_blocks[ti]):
                uncovered_count += 1
        
        # Scale up to estimate total uncovered
        return (uncovered_count / sample_size) * N_t
    
    # Initial solution: Greedy set cover to get a valid starting point
    def greedy_cover(rng):
        covered_mask = np.zeros(N_t, dtype=bool)
        selected_blocks = []
        remaining_blocks = list(range(N_b))
        
        while not np.all(covered_mask):
            if time.time() > deadline - 5.0:
                break
            
            # Find uncovered
            uncovered_idx = np.where(~covered_mask)[0]
            if len(uncovered_idx) == 0:
                break
            
            # Score blocks by number of newly covered t-subsets
            best_block = -1
            best_gain = -1
            
            # Sample candidate blocks from those covering random uncovered subsets
            sample_size = min(10, len(uncovered_idx))
            sampled_ts = rng.sample(uncovered_idx.tolist(), sample_size)
            
            candidates = set()
            for ti in sampled_ts:
                candidates.update(t_to_blocks[ti])
            
            for bi in candidates:
                if bi in selected_blocks:
                    continue
                gain = 0
                for ti in block_to_t[bi]:
                    if not covered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block == -1:
                # Fallback
                ti = rng.choice(uncovered_idx.tolist())
                best_block = rng.choice(t_to_blocks[ti])
            
            selected_blocks.append(best_block)
            for ti in block_to_t[best_block]:
                covered_mask[ti] = True
        
        return selected_blocks
    
    # Simulated Annealing
    def simulated_annealing(initial_blocks, max_iterations=50000):
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = count_uncovered(current_counts)
        
        if current_uncovered == 0:
            # Already valid, try to minimize block count
            pass
        
        best_blocks = list(current_blocks)
        best_uncovered = current_uncovered
        best_len = len(current_blocks)
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
        T = T_start
        
        rng = random.Random(123)
        
        for it in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Move: try to swap, add, or remove a block
            move_type = rng.random()
            
            if move_type < 0.6:
                # Swap: replace one block with another
                if not current_blocks:
                    continue
                old_idx = rng.randrange(len(current_blocks))
                old_block = current_blocks[old_idx]
                
                # Find a replacement block that covers at least some of the same t-subsets
                # Sample from blocks covering a random t-subset covered by old_block
                if block_to_t[old_block]:
                    ti_sample = rng.choice(block_to_t[old_block])
                    candidates = t_to_blocks[ti_sample]
                    if candidates:
                        new_block = rng.choice(candidates)
                        if new_block != old_block:
                            # Calculate delta in uncovered count
                            # This is expensive, so we approximate or do it carefully
                            # For now, let's do a quick check: does new_block cover any unique t of old_block?
                            # If yes, it might be good. If no, it might be bad.
                            
                            # Let's just perform the swap and calculate the actual delta
                            # But that's O(k) per swap. With 50k swaps, it might be slow.
                            # Let's try a cheaper heuristic: accept if new_block covers more "rare" t-subsets.
                            
                            # Actually, let's just do the swap and recompute uncovered count for affected parts?
                            # No, that's complex. Let's just recompute fully if we accept.
                            # But we need a delta for the SA acceptance criterion.
                            
                            # Alternative: Use the probabilistic estimate as the objective?
                            # The prompt suggests: "minimize this expected value ... treating the coverage status as a noisy objective"
                            
                            # Let's try: Objective = len(blocks) + lambda * prob_estimate_uncovered
                            # This is a composite objective.
                            
                            # Let's stick to a simpler SA: 
                            # State: set of blocks
                            # Move: add, remove, or swap
                            # Cost: if valid, cost = len(blocks). If invalid, cost = len(blocks) + penalty * uncovered_count
                            
                            # Let's use a penalty method.
                            # Cost = len(blocks) + 100 * uncovered_count
                            
                            # For the swap:
                            new_blocks = current_blocks.copy()
                            new_blocks[old_idx] = new_block
                            
                            # Calculate new uncovered count
                            # We can update counts incrementally
                            new_counts = current_counts.copy()
                            for ti in block_to_t[old_block]:
                                new_counts[ti] -= 1
                            for ti in block_to_t[new_block]:
                                new_counts[ti] += 1
                            
                            new_uncovered = count_uncovered(new_counts)
                            new_len = len(new_blocks)
                            
                            old_cost = len(current_blocks) + 100 * current_uncovered
                            new_cost = new_len + 100 * new_uncovered
                            
                            delta_cost = new_cost - old_cost
                            
                            # SA acceptance
                            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                                current_blocks = new_blocks
                                current_counts = new_counts
                                current_uncovered = new_uncovered
                                
                                if new_uncovered == 0 and new_len < best_len:
                                    best_len = new_len
                                    best_blocks = list(current_blocks)
                                    best_uncovered = 0
                            # If rejected, do nothing (current state remains)
                            
                            T *= cooling_rate
                            continue
            
            elif move_type < 0.8:
                # Add a block
                # Pick a random block
                new_block = rng.randrange(N_b)
                if new_block in current_blocks:
                    continue
                
                new_blocks = current_blocks + [new_block]
                new_counts = current_counts.copy()
                for ti in block_to_t[new_block]:
                    new_counts[ti] += 1
                
                new_uncovered = count_uncovered(new_counts)
                new_len = len(new_blocks)
                
                old_cost = len(current_blocks) + 100 * current_uncovered
                new_cost = new_len + 100 * new_uncovered
                
                delta_cost = new_cost - old_cost
                
                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                    current_blocks = new_blocks
                    current_counts = new_counts
                    current_uncovered = new_uncovered
                    
                    if new_uncovered == 0 and new_len < best_len:
                        best_len = new_len
                        best_blocks = list(current_blocks)
                        best_uncovered = 0
                # If rejected, do nothing
                
                T *= cooling_rate
                continue
            
            else:
                # Remove a block
                if not current_blocks:
                    continue
                old_idx = rng.randrange(len(current_blocks))
                old_block = current_blocks[old_idx]
                
                new_blocks = current_blocks.copy()
                new_blocks.pop(old_idx)
                
                new_counts = current_counts.copy()
                for ti in block_to_t[old_block]:
                    new_counts[ti] -= 1
                
                new_uncovered = count_uncovered(new_counts)
                new_len = len(new_blocks)
                
                old_cost = len(current_blocks) + 100 * current_uncovered
                new_cost = new_len + 100 * new_uncovered
                
                delta_cost = new_cost - old_cost
                
                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                    current_blocks = new_blocks
                    current_counts = new_counts
                    current_uncovered = new_uncovered
                    
                    if new_uncovered == 0 and new_len < best_len:
                        best_len = new_len
                        best_blocks = list(current_blocks)
                        best_uncovered = 0
                # If rejected, do nothing
                
                T *= cooling_rate
                continue
        
        return best_blocks
    
    # Main loop: Multi-start SA
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Get a valid solution via greedy
    rng = random.Random(42)
    initial_blocks = greedy_cover(rng)
    initial_counts = get_coverage_counts(initial_blocks)
    initial_uncovered = count_uncovered(initial_counts)
    
    if initial_uncovered > 0:
        # If greedy didn't finish, try to fix it? 
        # For now, assume greedy is valid or close.
        # If not valid, SA will try to fix it.
        pass
    
    # Phase 2: Run SA
    optimized_blocks = simulated_annealing(initial_blocks, max_iterations=20000)
    
    if len(optimized_blocks) < best_len:
        best_len = len(optimized_blocks)
        best_solution = optimized_blocks
    
    # Phase 3: Try a few more restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        local_rng = random.Random(42 + restart * 1000)
        new_initial = greedy_cover(local_rng)
        new_optimized = simulated_annealing(new_initial, max_iterations=10000)
        
        if len(new_optimized) < best_len:
            best_len = len(new_optimized)
            best_solution = new_optimized
    
    # Final check: ensure validity
    if best_solution is None:
        best_solution = initial_blocks
    
    # Verify and return
    final_counts = get_coverage_counts(best_solution)
    if count_uncovered(final_counts) > 0:
        # If invalid, fall back to greedy (which should be valid)
        final_counts = get_coverage_counts(initial_blocks)
        if count_uncovered(final_counts) == 0:
            best_solution = initial_blocks
        else:
            # Last resort: add blocks until valid
            # This should not happen if greedy is correct
            pass
    
    return [list(all_blocks[bi]) for bi in best_solution]
