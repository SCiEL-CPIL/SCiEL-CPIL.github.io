"""Simulated annealing approach for C(19,7,4) covering design."""
import itertools
import random
import numpy as np


def construct(v, k, t):
    """
    Construct a covering design using simulated annealing.
    
    State: set of blocks (each block is a k-subset of {0,...,v-1})
    Objective: minimize number of blocks while covering all t-subsets.
    
    Moves:
    1. Add a random block
    2. Remove a block if its unique t-subsets are covered by others
    3. Swap a block for another that covers its unique t-subsets
    4. Perturb: replace some points in a block
    
    Accept worse solutions probabilistically to escape local minima.
    """
    random.seed(42)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def block_t_subsets(block):
        """Return list of indices of t-subsets covered by this block."""
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def build_coverage(blocks):
        """Build coverage array: for each t-subset, how many blocks cover it."""
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for block in blocks:
            for idx in block_t_subsets(block):
                coverage[idx] += 1
        return coverage
    
    def is_covered(coverage):
        """Check if all t-subsets are covered."""
        return np.all(coverage >= 1)
    
    def uncovered_count(coverage):
        """Count number of uncovered t-subsets."""
        return int(np.sum(coverage == 0))
    
    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        """Greedy set cover initialization."""
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Find block that covers most uncovered t-subsets
            best_gain = -1
            best_block = None
            
            # Try random candidate blocks
            candidates = set()
            for _ in range(200):
                candidates.add(tuple(sorted(random.sample(range(v), k))))
            
            # Also try blocks based on uncovered t-subsets
            uncovered_indices = np.where(uncovered)[0]
            for idx in random.sample(uncovered_indices, min(50, len(uncovered_indices))):
                ts = all_t_subsets[idx]
                # Complete this t-subset to a k-subset
                remaining = list(set(range(v)) - set(ts))
                for _ in range(5):
                    extra = random.sample(remaining, k - t)
                    candidates.add(tuple(sorted(ts + extra)))
            
            for cand in candidates:
                gain = 0
                for t_idx in block_t_subsets(list(cand)):
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(cand)
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            for t_idx in block_t_subsets(best_block):
                uncovered[t_idx] = False
        
        return blocks
    
    blocks = greedy_init()
    
    # Ensure valid coverage
    coverage = build_coverage(blocks)
    if not is_covered(coverage):
        # Add blocks for any remaining uncovered t-subsets
        while not is_covered(coverage):
            uncovered_indices = np.where(coverage == 0)[0]
            idx = random.choice(uncovered_indices)
            ts = all_t_subsets[idx]
            remaining = list(set(range(v)) - set(ts))
            extra = random.sample(remaining, k - t)
            new_block = tuple(sorted(ts + extra))
            blocks.append(list(new_block))
            for t_idx in block_t_subsets(list(new_block)):
                coverage[t_idx] += 1
    
    initial_blocks = len(blocks)
    best_blocks = list(blocks)
    best_coverage = build_coverage(best_blocks)
    
    # --- Phase 2: Simulated annealing ---
    def unique_t_subsets(block, blocks_list, coverage):
        """Return t-subset indices that are only covered by this block."""
        unique = []
        for t_idx in block_t_subsets(block):
            if coverage[t_idx] == 1:
                unique.append(t_idx)
        return unique
    
    def try_remove_block(block_idx, blocks_list, coverage):
        """Try to remove block at block_idx. Returns True if successful."""
        block = blocks_list[block_idx]
        
        # Check if any t-subset would become uncovered
        for t_idx in block_t_subsets(block):
            if coverage[t_idx] == 1:
                return False
        
        # Safe to remove
        for t_idx in block_t_subsets(block):
            coverage[t_idx] -= 1
        blocks_list.pop(block_idx)
        return True
    
    def try_swap_block(block_idx, blocks_list, coverage):
        """Try to swap a block for a different one that covers its unique t-subsets."""
        block = blocks_list[block_idx]
        unique = unique_t_subsets(block, blocks_list, coverage)
        
        if not unique:
            # No unique coverage, can try removing
            return try_remove_block(block_idx, blocks_list, coverage)
        
        # Generate candidate replacement blocks
        candidates = set()
        
        # Candidates based on unique t-subsets
        for t_idx in unique:
            ts = all_t_subsets[t_idx]
            remaining = list(set(range(v)) - set(ts))
            for _ in range(3):
                extra = random.sample(remaining, k - t)
                candidates.add(tuple(sorted(ts + extra)))
        
        # Random candidates
        for _ in range(20):
            candidates.add(tuple(sorted(random.sample(range(v), k))))
        
        for cand in candidates:
            cand_list = list(cand)
            # Check if candidate covers all unique t-subsets
            cand_t_set = set(block_t_subsets(cand_list))
            if all(t_idx in cand_t_set for t_idx in unique):
                # Remove old block
                for t_idx in block_t_subsets(block):
                    coverage[t_idx] -= 1
                # Add new block
                for t_idx in block_t_subsets(cand_list):
                    coverage[t_idx] += 1
                blocks_list[block_idx] = cand_list
                return True
        
        return False
    
    def perturb_block(block_idx, blocks_list, coverage, num_changes=1):
        """Perturb a block by changing some of its points."""
        block = list(blocks_list[block_idx])
        new_block = list(block)
        
        for _ in range(num_changes):
            pos = random.randint(0, k - 1)
            old_point = new_block[pos]
            available = [x for x in range(v) if x not in new_block]
            if available:
                new_point = random.choice(available)
                new_block[pos] = new_point
        
        new_block.sort()
        
        # Check if this is a valid perturbation (maintains coverage)
        # First, see what unique t-subsets the old block covers
        unique_old = unique_t_subsets(block, blocks_list, coverage)
        
        # Check if new block covers all unique t-subsets
        new_t_set = set(block_t_subsets(new_block))
        if all(t_idx in new_t_set for t_idx in unique_old):
            # Apply perturbation
            for t_idx in block_t_subsets(block):
                coverage[t_idx] -= 1
            for t_idx in block_t_subsets(new_block):
                coverage[t_idx] += 1
            blocks_list[block_idx] = new_block
            return True
        
        return False
    
    # Simulated annealing parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    max_iterations = 50000
    
    current_blocks = list(blocks)
    current_coverage = build_coverage(current_blocks)
    current_cost = len(current_blocks)
    
    best_cost = current_cost
    best_blocks = list(current_blocks)
    
    for iteration in range(max_iterations):
        if T < min_T:
            break
        
        # Choose a move
        move_type = random.choices([1, 2, 3, 4], weights=[10, 30, 30, 30])[0]
        
        old_cost = len(current_blocks)
        old_coverage_state = current_coverage.copy()
        old_blocks = [b[:] for b in current_blocks]
        
        if move_type == 1:  # Add a random block
            new_block = tuple(sorted(random.sample(range(v), k)))
            current_blocks.append(list(new_block))
            for t_idx in block_t_subsets(list(new_block)):
                current_coverage[t_idx] += 1
        
        elif move_type == 2:  # Remove a block
            if len(current_blocks) <= 1:
                # Revert
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
            block_idx = random.randint(0, len(current_blocks) - 1)
            if not try_remove_block(block_idx, current_blocks, current_coverage):
                # Revert
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
        
        elif move_type == 3:  # Swap a block
            if len(current_blocks) <= 1:
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
            block_idx = random.randint(0, len(current_blocks) - 1)
            if not try_swap_block(block_idx, current_blocks, current_coverage):
                # Revert
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
        
        else:  # Perturb a block
            if len(current_blocks) <= 1:
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
            block_idx = random.randint(0, len(current_blocks) - 1)
            if not perturb_block(block_idx, current_blocks, current_coverage):
                # Revert
                current_blocks = old_blocks
                current_coverage = old_coverage_state
                T *= cooling_rate
                continue
        
        # Check validity
        if not is_covered(current_coverage):
            # Invalid, revert
            current_blocks = old_blocks
            current_coverage = old_coverage_state
            T *= cooling_rate
            continue
        
        new_cost = len(current_blocks)
        delta_cost = new_cost - old_cost
        
        # Accept or reject
        if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
            if new_cost < best_cost:
                best_cost = new_cost
                best_blocks = [b[:] for b in current_blocks]
        else:
            # Revert
            current_blocks = old_blocks
            current_coverage = old_coverage_state
        
        T *= cooling_rate
    
    # Return the best solution found
    # Ensure validity
    final_coverage = build_coverage(best_blocks)
    if not is_covered(final_coverage):
        # Fallback to greedy
        return greedy_init()
    
    # Remove duplicate blocks if any
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
