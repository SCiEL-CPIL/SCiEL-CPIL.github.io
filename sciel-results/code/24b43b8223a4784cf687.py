"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    # For v=19, k=8, C(19,8) = 75582. This is small enough to precompute.
    all_blocks = list(itertools.combinations(range(v), k))
    block_cover_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_set_index[ts])
        block_cover_indices.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Precompute: for each t-set index, which blocks cover it?
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, block in enumerate(all_blocks):
        for ts_idx in block_cover_indices[b_idx]:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # 3. Greedy Set Cover with Randomized Tie-Breaking
    best_solution = None
    best_size = float('inf')
    
    # Standard Greedy
    solution = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, t_set_to_blocks, rng_seed=42)
    if solution:
        best_size = len(solution)
        best_solution = solution
    
    # Try a few randomized restarts if time permits
    if time.time() < deadline - 10.0:
        for seed in range(1, 10):
            if time.time() > deadline - 5.0:
                break
            sol = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, t_set_to_blocks, rng_seed=seed + 100, randomize=True)
            if sol and len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    if best_solution is None:
        return []
    
    # 4. Backtracking polish: try to reduce block count by replacing subsets
    if time.time() < deadline - 5.0:
        polished = _backtrack_polish(best_solution, all_blocks, block_cover_indices, num_t_sets, t_set_to_blocks, deadline)
        if polished and len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished
    
    return best_solution

def _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, t_set_to_blocks, rng_seed=42, randomize=False):
    rng = random.Random(rng_seed)
    
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline - 2.0:
            return None  # Timeout
        
        target_tset_idx = rng.choice(list(uncovered)) if randomize else min(uncovered)
        
        candidate_block_indices = t_set_to_blocks[target_tset_idx]
        
        best_block_idx = -1
        max_gain = 0
        
        for b_idx in candidate_block_indices:
            gain = 0
            for ts_idx in block_cover_indices[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
        
        selected_blocks.append(all_blocks[best_block_idx])
        
        for ts_idx in block_cover_indices[best_block_idx]:
            if ts_idx in uncovered:
                uncovered.remove(ts_idx)
        
        if not uncovered:
            break
    
    return selected_blocks

def _backtrack_polish(solution, all_blocks, block_cover_indices, num_t_sets, t_set_to_blocks, deadline):
    """
    Try to reduce the number of blocks by identifying blocks that can be removed
    or replaced with fewer blocks. Uses a limited backtracking search.
    """
    n = len(solution)
    if n < 2:
        return solution
    
    # First pass: redundancy removal (remove blocks that are not needed)
    # A block is redundant if all t-sets it covers are also covered by other blocks.
    # Build coverage count for each t-set
    coverage_count = [0] * num_t_sets
    for block in solution:
        for ts_idx in block_cover_indices[all_blocks.index(block)]:
            coverage_count[ts_idx] += 1
    
    # Actually, let's work with block indices for efficiency
    solution_block_indices = []
    for block in solution:
        # Find the index
        found = False
        for i, b in enumerate(all_blocks):
            if b == block:
                solution_block_indices.append(i)
                found = True
                break
        if not found:
            # Should not happen
            return solution
    
    # Rebuild coverage count using indices
    coverage_count = [0] * num_t_sets
    for b_idx in solution_block_indices:
        for ts_idx in block_cover_indices[b_idx]:
            coverage_count[ts_idx] += 1
    
    # Try to remove redundant blocks
    removed = set()
    for i, b_idx in enumerate(solution_block_indices):
        if b_idx in removed:
            continue
        # Check if this block is redundant
        redundant = True
        for ts_idx in block_cover_indices[b_idx]:
            if coverage_count[ts_idx] == 1:
                redundant = False
                break
        if redundant:
            removed.add(b_idx)
            for ts_idx in block_cover_indices[b_idx]:
                coverage_count[ts_idx] -= 1
    
    if removed:
        new_solution = [all_blocks[b_idx] for b_idx in solution_block_indices if b_idx not in removed]
        return new_solution
    
    # Backtracking: try to replace pairs of blocks with one block
    # For each pair of blocks, check if there's a single block that covers
    # all t-sets that were uniquely covered by these two blocks.
    # This is expensive, so we limit the search.
    
    if time.time() > deadline - 3.0:
        return solution
    
    # Try single-block replacement: for each block, see if removing it and adding
    # one other block still covers everything.
    # This is equivalent to checking if the block is "critical" and finding a replacement.
    
    # More practical: try to find blocks that can be swapped for better ones.
    # For each block in the solution, look for an alternative block that covers
    # the same or more uncovered t-sets (after removing the original).
    
    # Due to time constraints, we'll do a limited local search:
    # Try removing each block and see if the rest still covers everything.
    # If not, try to find a replacement.
    
    current_solution = list(solution)
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 3.0:
        improved = False
        iteration += 1
        
        # Recompute coverage
        coverage_count = [0] * num_t_sets
        for block in current_solution:
            # Find block index
            b_idx = None
            for i, b in enumerate(all_blocks):
                if b == block:
                    b_idx = i
                    break
            if b_idx is not None:
                for ts_idx in block_cover_indices[b_idx]:
                    coverage_count[ts_idx] += 1
        
        # Try to remove each block
        for i in range(len(current_solution)):
            if time.time() > deadline - 3.0:
                break
            
            block_to_remove = current_solution[i]
            b_idx = None
            for j, b in enumerate(all_blocks):
                if b == block_to_remove:
                    b_idx = j
                    break
            
            if b_idx is None:
                continue
            
            # Temporarily remove
            for ts_idx in block_cover_indices[b_idx]:
                coverage_count[ts_idx] -= 1
            
            # Check if all t-sets are still covered
            all_covered = True
            for ts_idx in block_cover_indices[b_idx]:
                if coverage_count[ts_idx] == 0:
                    all_covered = False
                    break
            
            if all_covered:
                # Block is redundant, remove it
                current_solution.pop(i)
                improved = True
                break
            else:
                # Restore
                for ts_idx in block_cover_indices[b_idx]:
                    coverage_count[ts_idx] += 1
    
    return current_solution

# EVOLVE-BLOCK-END
