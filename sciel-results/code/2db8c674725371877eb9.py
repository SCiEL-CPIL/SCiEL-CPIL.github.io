"""Improved covering design solver using randomized greedy with local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

# Use a fixed seed for reproducibility
random.seed(42)


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with local search.
    
    Returns a list of blocks (each block is a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Create a mapping from t-subset to index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Track uncovered t-subsets using a boolean array
    uncovered = [True] * num_t_subsets
    
    def count_uncovered(block):
        """Count how many t-subsets of the block are still uncovered."""
        count = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if uncovered[idx]:
                count += 1
        return count
    
    def mark_covered(block):
        """Mark all t-subsets of the block as covered."""
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            uncovered[idx] = False
    
    def unmark_covered(block):
        """Unmark all t-subsets of the block as uncovered (for removal)."""
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            # Only set to True if no other block covers it
            # For simplicity in local search, we'll handle this differently
            uncovered[idx] = True
    
    def is_fully_covered():
        """Check if all t-subsets are covered."""
        return not any(uncovered)
    
    # Initial greedy construction
    def greedy_construct():
        """Standard greedy: pick uncovered t-subset, build block around it."""
        blocks = []
        uncovered_local = uncovered[:]
        
        while any(uncovered_local):
            # Find first uncovered t-subset
            first_uncovered_idx = -1
            for i in range(num_t_subsets):
                if uncovered_local[i]:
                    first_uncovered_idx = i
                    break
            if first_uncovered_idx == -1:
                break
            
            seed_block = list(all_t_subsets[first_uncovered_idx])
            
            # Greedily add points to maximize coverage
            block = list(seed_block)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: number of currently uncovered t-subsets that would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx[ts]
                        if uncovered_local[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                uncovered_local[idx] = False
        
        return blocks
    
    # Run initial greedy
    blocks = greedy_construct()
    
    # Local search: try to improve by removing redundant blocks or swapping
    def local_search(blocks):
        """Try to remove blocks that are redundant, or swap points to reduce total blocks."""
        if not blocks:
            return blocks
        
        current_blocks = [b[:] for b in blocks]
        
        # Try removing each block if it's redundant
        improved = True
        while improved:
            improved = False
            for i in range(len(current_blocks)):
                # Temporarily remove block i
                remaining = [b for j, b in enumerate(current_blocks) if j != i]
                
                # Check if all t-subsets are still covered
                covered = [False] * num_t_subsets
                for b in remaining:
                    for ts in itertools.combinations(b, t):
                        idx = t_subset_to_idx[ts]
                        covered[idx] = True
                
                if all(covered):
                    # Block i is redundant, remove it
                    current_blocks.pop(i)
                    improved = True
                    break
        
        return current_blocks
    
    # Apply local search
    blocks = local_search(blocks)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return blocks
    
    # Try randomized restarts with different seeds for initial blocks
    best_blocks = blocks[:]
    best_count = len(best_blocks)
    
    restart_count = 0
    while time.time() - start_time < deadline - 5.0:
        restart_count += 1
        
        # Randomized greedy: pick random uncovered t-subsets
        def randomized_greedy():
            blocks_local = []
            uncovered_local = uncovered[:]
            
            while any(uncovered_local):
                # Find all uncovered t-subsets
                uncovered_indices = [i for i in range(num_t_subsets) if uncovered_local[i]]
                if not uncovered_indices:
                    break
                
                # Pick a random uncovered t-subset
                chosen_idx = random.choice(uncovered_indices)
                seed_block = list(all_t_subsets[chosen_idx])
                
                # Greedily add points
                block = list(seed_block)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_to_idx[ts]
                            if uncovered_local[idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point is None:
                        break
                    block.append(best_point)
                
                block.sort()
                blocks_local.append(block)
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    uncovered_local[idx] = False
            
            return blocks_local
        
        candidate = randomized_greedy()
        candidate = local_search(candidate)
        
        if len(candidate) < best_count:
            best_count = len(candidate)
            best_blocks = candidate[:]
        
        # Check time
        if time.time() - start_time > deadline - 2.0:
            break
    
    return best_blocks
# EVOLVE-BLOCK-END
