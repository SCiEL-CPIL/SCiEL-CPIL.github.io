import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block for faster access later
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Phase 1: Greedy Set Cover
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 80.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = np.argmax(gains)
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Phase 2: Local Search Deletion
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    current_best = len(selected_block_indices)
    
    # Phase 3: Simulated Annealing
    # State: set of selected blocks
    # Energy: number of uncovered t-subsets (should be 0 for valid solutions)
    # We want to minimize block count while maintaining validity
    
    if time.time() < deadline - 30.0:
        import random
        random.seed(42)
        
        # Precompute for each block: which t-subsets it covers
        # Already have A matrix
        
        # Current solution
        current_solution = list(selected_block_indices)
        current_size = len(current_solution)
        
        # Build cover counts for current solution
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_solution:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] += 1
        
        # Verify all covered
        uncovered = np.sum(cover_counts < 1)
        
        # Temperature schedule
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.995
        
        best_solution = list(current_solution)
        best_size = current_size
        
        sa_start = time.time()
        sa_deadline = deadline - 25.0
        
        while T > T_min and time.time() < sa_deadline:
            # Try a move: swap one block with another
            if not current_solution:
                break
                
            # Pick a random block to potentially remove
            remove_pos = random.randint(0, len(current_solution) - 1)
            remove_block = current_solution[remove_pos]
            
            # Compute what would be uncovered if we remove this block
            covered_by_remove = A.indices[A.indptr[remove_block]:A.indptr[remove_block+1]]
            
            # Count how many t-subsets would become uncovered
            would_become_uncovered = np.sum(cover_counts[covered_by_remove] == 1)
            
            # If removing this block doesn't uncover anything, it's redundant
            if would_become_uncovered == 0:
                # Just remove it - this is a pure improvement
                for c in covered_by_remove:
                    cover_counts[c] -= 1
                current_solution.pop(remove_pos)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_solution)
                
                # Continue with same temperature
                continue
            
            # We need to find a replacement block that covers the newly uncovered t-subsets
            # The newly uncovered t-subsets are those with cover_counts == 1
            newly_uncovered = covered_by_remove[cover_counts[covered_by_remove] == 1]
            
            if len(newly_uncovered) == 0:
                # Shouldn't happen if would_become_uncovered > 0
                T *= cooling_rate
                continue
            
            # Find candidate blocks that cover at least one of the newly uncovered t-subsets
            # and compute their marginal gain
            candidate_blocks = []
            for b_idx in range(num_blocks):
                if b_idx in current_solution:
                    continue
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                # Count how many newly_uncovered this block covers
                overlap = np.sum(cover_counts[covered] == 0)  # blocks that would newly cover
                # Actually we want blocks that cover the newly_uncovered subsets
                # Let's compute differently
                pass
            
            # More efficient: for each newly uncovered t-subset, find blocks that cover it
            # But that's expensive. Instead, let's just try random candidate blocks
            
            # Pick a random candidate block not in current solution
            candidate_block = random.randint(0, num_blocks - 1)
            if candidate_block in current_solution:
                T *= cooling_rate
                continue
            
            # Compute the energy change:
            # Remove remove_block, add candidate_block
            # Energy = -number_of_blocks (we want to minimize, so maximize coverage)
            # Actually, let's use: energy = number_of_uncovered_t_subsets * 1000 + number_of_blocks
            # This way we prioritize validity, then minimize blocks
            
            # Compute coverage if we remove remove_block and add candidate_block
            covered_remove = A.indices[A.indptr[remove_block]:A.indptr[remove_block+1]]
            covered_add = A.indices[A.indptr[candidate_block]:A.indptr[candidate_block+1]]
            
            # Simulate the change
            # First, what happens to cover_counts if we remove remove_block?
            # cover_counts[covered_remove] would decrease by 1
            # Then if we add candidate_block, cover_counts[covered_add] would increase by 1
            
            # Uncovered after change:
            # A t-subset is uncovered if its final cover_count is 0
            # Final cover_count for subset s:
            #   if s in covered_remove: cover_counts[s] - 1 + (1 if s in covered_add else 0)
            #   if s not in covered_remove: cover_counts[s] + (1 if s in covered_add else 0)
            
            # Compute number of newly uncovered subsets
            # Subsets that were covered only by remove_block and not by candidate_block
            only_by_remove = np.sum((cover_counts[covered_remove] == 1) & 
                                    (np.isin(covered_remove, covered_add) == False))
            # Wait, this isn't right. Let me think more carefully.
            
            # Let's just compute the actual change in uncovered count
            # Current uncovered: 0 (assuming valid)
            # After removing remove_block: subsets with cover_count == 1 in covered_remove become uncovered
            # After adding candidate_block: subsets in covered_add become covered again
            
            # Uncovered after both changes:
            # A subset s is uncovered if:
            #   (s in covered_remove and cover_counts[s] == 1 and s not in covered_add)
            #   OR (s not in covered_remove and cover_counts[s] == 0 and s not in covered_add)
            
            # Since we start with a valid solution, cover_counts[s] >= 1 for all s
            # So the second condition is never true initially
            
            # Uncovered after change = subsets s where:
            #   s in covered_remove, cover_counts[s] == 1, and s not in covered_add
            
            # Compute this efficiently
            in_both = np.intersect1d(covered_remove, covered_add)
            only_in_remove = np.setdiff1d(covered_remove, covered_add)
            
            newly_uncovered_after_change = np.sum(cover_counts[only_in_remove] == 1)
            
            # Energy change:
            # Delta in block count: 0 (we're swapping)
            # Delta in uncovered: newly_uncovered_after_change
            # So delta_energy = newly_uncovered_after_change * 1000
            
            delta_energy = newly_uncovered_after_change * 1000
            
            # Acceptance probability
            if delta_energy < 0:
                # Improvement - always accept
                accept = True
            else:
                # Worse - accept with probability exp(-delta_energy / T)
                accept_prob = np.exp(-delta_energy / T)
                accept = random.random() < accept_prob
            
            if accept:
                # Apply the swap
                for c in covered_remove:
                    cover_counts[c] -= 1
                for c in covered_add:
                    cover_counts[c] += 1
                
                current_solution[remove_pos] = candidate_block
                
                # Check if we improved the best solution
                if newly_uncovered_after_change == 0:
                    # Still valid
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = list(current_solution)
            
            # Cool down
            T *= cooling_rate
        
        # Use the best solution found during SA
        if best_size < current_best:
            selected_block_indices = best_solution
            current_best = best_size
    
    # Phase 4: Try ILP refinement on the current solution
    if time.time() < deadline - 20.0:
        try:
            # Objective: minimize sum(x)
            c = np.ones(num_blocks)
            
            # Constraints: A^T x >= 1
            # A is (num_blocks, num_t_subsets), so A^T is (num_t_subsets, num_blocks)
            A_T = A.T.tocsr()
            
            constraint = LinearConstraint(A_T, lb=1, ub=np.inf)
            
            # Bounds: x in [0, 1]
            lb = np.zeros(num_blocks)
            ub = np.ones(num_blocks)
            bounds = Bounds(lb, ub)
            
            # Integrality: all variables are binary
            integrality = np.ones(num_blocks)
            
            # Run MILP with a time limit
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                       options={'time_limit': 10.0})
            
            if res.status == 0 and res.x is not None:
                ilp_solution = res.x.astype(int)
                ilp_blocks = np.where(ilp_solution == 1)[0]
                
                # Verify validity
                if len(ilp_blocks) > 0:
                    cover_check = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in ilp_blocks:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_check[covered] += 1
                    
                    if np.all(cover_check >= 1) and len(ilp_blocks) < current_best:
                        selected_block_indices = list(ilp_blocks)
                        current_best = len(selected_block_indices)
                        
        except Exception:
            pass
            
    # Phase 5: Final Deletion Pass on the best solution found
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
