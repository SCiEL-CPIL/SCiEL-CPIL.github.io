import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    M = M.tocsc() # CSC is often faster for column slicing if we were doing that, but we do row access in greedy. CSR is good for row access. Let's stick to CSR for M[b_idx] access.
    M = M.tocsr()
    
    # Precompute indices for each block for fast coverage updates
    block_indices = [M[i].indices for i in range(num_k)]
    
    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            # Shuffle to avoid bias? Or just iterate. Iterating is fine.
            for b_idx in blocks_idx:
                idxs = block_indices[b_idx]
                # Check if coverage is > 1 for all these idxs
                if np.all(coverage[idxs] > 1):
                    # Safe to remove
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed):
        random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            # Compute gain for all blocks
            mask = (coverage == 0).astype(np.int8)
            gains = M.dot(mask)
            
            # Find block with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            # Pick one of the blocks with max gain (random tie-breaking)
            candidates = np.where(gains == max_gain)[0]
            b_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks_idx.append(b_idx)
            
            # Update coverage
            idxs = block_indices[b_idx]
            coverage[idxs] += 1
            
            # Update uncovered_count
            uncovered_count = np.sum(coverage == 0)
            
        return blocks_idx, coverage, None

    def simulated_annealing(blocks_idx, coverage, time_budget=5.0):
        """
        Improves the solution using simulated annealing.
        State: set of blocks.
        Energy: len(blocks) + P * uncovered_count.
        Move: Add a block, Remove a block, or Swap a block.
        """
        blocks_idx = list(blocks_idx)
        coverage = coverage.copy()
        current_energy = len(blocks_idx) # Since it's valid, uncovered is 0
        
        T = 10.0
        T_min = 0.1
        alpha = 0.95
        start_t = time.time()
        
        best_local_blocks = list(blocks_idx)
        best_local_energy = current_energy
        
        while T > T_min and time.time() < start_t + time_budget:
            if time.time() > deadline - 2.0:
                break
                
            # Generate a move
            move_type = random.random()
            old_blocks = list(blocks_idx)
            old_coverage = coverage.copy()
            
            # Calculate change in energy
            delta_blocks = 0
            delta_uncovered = 0
            
            if move_type < 0.4: # Add a block
                if len(blocks_idx) > 0:
                    # Pick a block not in current set
                    # Simple: pick random block index, if in set, try another
                    b_new = random.randint(0, num_k - 1)
                    if b_new not in blocks_idx:
                        blocks_idx.append(b_new)
                        idxs = block_indices[b_new]
                        # Count how many newly covered
                        newly_covered = np.sum(coverage[idxs] == 0)
                        coverage[idxs] += 1
                        delta_blocks = 1
                        delta_uncovered = -newly_covered
                    else:
                        continue
                else:
                    b_new = random.randint(0, num_k - 1)
                    blocks_idx.append(b_new)
                    idxs = block_indices[b_new]
                    newly_covered = np.sum(coverage[idxs] == 0)
                    coverage[idxs] += 1
                    delta_blocks = 1
                    delta_uncovered = -newly_covered

            elif move_type < 0.8: # Remove a block
                if len(blocks_idx) > 0:
                    b_old_idx = random.randint(0, len(blocks_idx) - 1)
                    b_old = blocks_idx[b_old_idx]
                    idxs = block_indices[b_old]
                    # Check if removal causes uncovered
                    newly_uncovered = np.sum(coverage[idxs] == 1)
                    if newly_uncovered == 0:
                        # Safe removal
                        coverage[idxs] -= 1
                        blocks_idx.pop(b_old_idx)
                        delta_blocks = -1
                        delta_uncovered = 0
                    else:
                        # Unsafe removal, high penalty
                        coverage[idxs] -= 1
                        blocks_idx.pop(b_old_idx)
                        delta_blocks = -1
                        delta_uncovered = newly_uncovered
                else:
                    continue

            else: # Swap: Remove one, Add one
                if len(blocks_idx) > 0:
                    b_old_idx = random.randint(0, len(blocks_idx) - 1)
                    b_old = blocks_idx[b_old_idx]
                    b_new = random.randint(0, num_k - 1)
                    
                    if b_new == b_old:
                        continue
                        
                    # Remove old
                    idxs_old = block_indices[b_old]
                    newly_uncovered_by_removal = np.sum(coverage[idxs_old] == 1)
                    coverage[idxs_old] -= 1
                    blocks_idx.pop(b_old_idx)
                    
                    # Add new
                    idxs_new = block_indices[b_new]
                    newly_covered_by_addition = np.sum(coverage[idxs_new] == 0)
                    coverage[idxs_new] += 1
                    blocks_idx.append(b_new)
                    
                    delta_blocks = 0
                    # Net change in uncovered
                    # This is tricky to compute exactly from just sums because of overlaps.
                    # Let's just recompute uncovered count for safety in energy calc if we accept.
                    # For now, approximate delta_uncovered = newly_uncovered_by_removal - newly_covered_by_addition
                    # This is not exact if the new block covers some of the ones uncovered by removal.
                    # Let's calculate exact uncovered count for the new state.
                    current_uncovered = np.sum(coverage == 0)
                    old_uncovered = np.sum(old_coverage == 0)
                    delta_uncovered = current_uncovered - old_uncovered

            # Calculate Energy
            # E = len(blocks) + 100 * uncovered_count
            # We need current uncovered count.
            # For Add/Remove, we tracked delta. For Swap, we calculated exact.
            # Let's standardize: calculate current_uncovered for all moves to be safe, 
            # but only if we are considering acceptance.
            
            current_uncovered = np.sum(coverage == 0)
            old_uncovered = np.sum(old_coverage == 0)
            
            current_energy = len(blocks_idx) + 100 * current_uncovered
            old_energy = len(old_blocks) + 100 * old_uncovered
            
            delta_E = current_energy - old_energy
            
            # Acceptance probability
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                # Update best if improved
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_blocks = list(blocks_idx)
                    # We don't update global best here to keep SA fast, 
                    # we return the best local found
            else:
                # Reject, revert
                blocks_idx = old_blocks
                coverage = old_coverage
            
            T *= alpha
            
        return best_local_blocks

    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        res = greedy_construct(i)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            # Local Search with Simulated Annealing
            # Allocate time based on remaining budget
            remaining = deadline - time.time()
            if remaining > 15:
                sa_time = min(5.0, remaining * 0.2)
                improved_blocks = simulated_annealing(blocks_idx, coverage, sa_time)
                
                # Verify and reduce the SA result
                # SA might return an invalid set if we accepted bad moves and didn't revert correctly?
                # No, we revert if rejected. If accepted, it's a valid state in the SA walk.
                # But SA minimizes E = blocks + penalty. So it might end with uncovered > 0?
                # If T is low, it should converge to valid.
                # Let's force reduction and check validity.
                
                # Reconstruct coverage for improved_blocks
                # Actually, simulated_annealing returns best_local_blocks which was tracked.
                # We need to ensure it's valid.
                # Let's do a quick check.
                check_coverage = np.zeros(num_t, dtype=np.int32)
                for b in improved_blocks:
                    check_coverage[block_indices[b]] += 1
                
                if np.all(check_coverage > 0):
                    # Valid
                    improved_blocks = reduce_blocks(improved_blocks, check_coverage)
                    if len(improved_blocks) < best_score:
                        best_score = len(improved_blocks)
                        best_blocks = [list(all_k_blocks[b]) for b in improved_blocks]

    return best_blocks
