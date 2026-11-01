import itertools
import random
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    
    # Map t-subset to index
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # Precompute block_bitsets and block_t_indices
    block_bitsets = []
    block_t_indices = []
    for b in k_blocks:
        bits = 0
        t_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            bits |= (1 << idx)
            t_indices.append(idx)
        block_bitsets.append(bits)
        block_t_indices.append(t_indices)
    
    # Helper to convert index list to actual blocks
    def to_blocks(indices):
        return [list(k_blocks[i]) for i in indices]
    
    def is_valid(solution_indices):
        """Check if solution covers all t-subsets."""
        if not solution_indices:
            return num_t_subsets == 0
        covered = 0
        for idx in solution_indices:
            covered |= block_bitsets[idx]
        return (covered & ((1 << num_t_subsets) - 1)) == ((1 << num_t_subsets) - 1)
        
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << num_t_subsets) - 1
        solution = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            
            indices = list(range(num_k_blocks))
            random.shuffle(indices)
            
            for b_idx in indices:
                bits = block_bitsets[b_idx]
                gain = (bits & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if best_gain == uncovered.bit_count():
                        break
            
            if best_block_idx == -1:
                break
                
            solution.append(best_block_idx)
            uncovered &= ~block_bitsets[best_block_idx]
            
            if time.time() > deadline:
                return None
        
        return solution

    def reduce_solution(solution):
        """Remove redundant blocks."""
        if not solution:
            return []
        
        counts = [0] * num_t_subsets
        for b_idx in solution:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
                
        new_solution = []
        for b_idx in solution:
            removable = True
            for t_idx in block_t_indices[b_idx]:
                if counts[t_idx] == 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_t_indices[b_idx]:
                    counts[t_idx] -= 1
            else:
                new_solution.append(b_idx)
                
        return new_solution

    def complement_packing_improve(solution, max_iters=50):
        """Use complement packing with bipartite matching to improve solution."""
        if not solution:
            return solution
        
        current_size = len(solution)
        sol_set = set(solution)
        best_solution = solution[:]
        best_size = current_size
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current coverage
            covered = 0
            for b_idx in sol_set:
                covered |= block_bitsets[b_idx]
            
            # Find uncovered t-subsets
            uncovered_mask = (~covered) & ((1 << num_t_subsets) - 1)
            if uncovered_mask == 0:
                break
            
            # Get list of uncovered t-subset indices
            uncovered_indices = []
            mask = uncovered_mask
            while mask:
                # Find lowest set bit
                low_bit = mask & (-mask)
                idx = low_bit.bit_length() - 1
                uncovered_indices.append(idx)
                mask ^= low_bit
            
            if not uncovered_indices:
                break
            
            # For each uncovered t-subset, find blocks that cover it
            # Build bipartite graph: left = uncovered t-subsets, right = candidate blocks
            # Edge exists if block covers the t-subset
            
            # Precompute: for each t-subset index, which blocks cover it
            # This is expensive, so we'll do it on-the-fly but cache if possible
            
            # Strategy: Find a set of blocks that covers many uncovered t-subsets
            # using a greedy + matching hybrid
            
            # First, identify "promising" blocks: those that cover at least one uncovered t-subset
            candidate_blocks = set()
            for t_idx in uncovered_indices:
                # Find blocks covering this t-subset
                # We need to search through all blocks - this is the bottleneck
                # Optimization: Precompute t_to_blocks mapping
                pass
            
            # Alternative approach: Use random sampling of blocks and evaluate their "complement packing" value
            # A block's value = number of uncovered t-subsets it covers
            
            # Sample a subset of blocks to keep computation tractable
            sample_size = min(num_k_blocks, 5000)
            sample_indices = random.sample(range(num_k_blocks), sample_size)
            
            # Score each sampled block by how many uncovered t-subsets it covers
            scored_blocks = []
            for b_idx in sample_indices:
                bits = block_bitsets[b_idx]
                gain = (bits & uncovered_mask).bit_count()
                if gain > 0:
                    scored_blocks.append((gain, b_idx))
            
            if not scored_blocks:
                # No block covers any uncovered t-subset - shouldn't happen if we have all blocks
                break
            
            # Sort by gain descending
            scored_blocks.sort(reverse=True)
            
            # Take top candidates
            top_candidates = [b_idx for _, b_idx in scored_blocks[:100]]
            
            # Now use bipartite matching to find a good set of blocks
            # Left side: uncovered t-subsets (or a subset)
            # Right side: top candidate blocks
            
            # To keep matching tractable, limit the number of t-subsets
            max_t_subsets = min(len(uncovered_indices), 200)
            selected_t_subsets = uncovered_indices[:max_t_subsets]
            
            # Build adjacency: for each selected t-subset, which candidate blocks cover it
            # We'll use a sparse matrix for matching
            
            row_indices = []
            col_indices = []
            
            for row_idx, t_idx in enumerate(selected_t_subsets):
                t_set = t_subsets[t_idx]
                for col_idx, b_idx in enumerate(top_candidates):
                    # Check if block b_idx covers t_set
                    b = k_blocks[b_idx]
                    if all(x in b for x in t_set):
                        row_indices.append(row_idx)
                        col_indices.append(col_idx)
            
            if not row_indices:
                break
            
            # Build sparse matrix
            n_rows = len(selected_t_subsets)
            n_cols = len(top_candidates)
            data = np.ones(len(row_indices))
            adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))
            
            # Find maximum bipartite matching
            try:
                matching = maximum_bipartite_matching(adj_matrix)
                # matching[i] = j means row i matched to col j
                matched_cols = set()
                for i, j in enumerate(matching):
                    if j >= 0:
                        matched_cols.add(j)
            except:
                matched_cols = set()
            
            # If matching found some blocks, try adding them
            if matched_cols:
                new_sol = list(sol_set)
                for col_idx in matched_cols:
                    b_idx = top_candidates[col_idx]
                    if b_idx not in sol_set:
                        new_sol.append(b_idx)
                
                # Reduce the new solution
                reduced = reduce_solution(new_sol)
                
                if len(reduced) < best_size:
                    best_solution = reduced
                    best_size = len(reduced)
                    sol_set = set(reduced)
                    current_size = best_size
            
            # Also try a simpler heuristic: just add the single best block
            if scored_blocks:
                best_gain, best_b_idx = scored_blocks[0]
                if best_b_idx not in sol_set:
                    new_sol = list(sol_set) + [best_b_idx]
                    reduced = reduce_solution(new_sol)
                    if len(reduced) < best_size:
                        best_solution = reduced
                        best_size = len(reduced)
                        sol_set = set(reduced)
                        current_size = best_size
        
        return best_solution

    def local_search_improve(solution, max_iters=30):
        """Try to swap blocks to enable further reductions."""
        if not solution:
            return solution
        
        current_size = len(solution)
        sol_set = set(solution)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            candidate = random.randint(0, num_k_blocks - 1)
            if candidate in sol_set:
                continue
                
            new_sol = solution + [candidate]
            reduced = reduce_solution(new_sol)
            
            if len(reduced) < current_size:
                solution = reduced
                sol_set = set(solution)
                current_size = len(solution)
            
        return solution

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts with complement packing
    for seed in range(150):
        if time.time() > deadline - 15:  # Stop 15s early to ensure return
            break
            
        sol = greedy_construct(seed)
        if sol is None:
            continue
            
        if not is_valid(sol):
            continue
            
        sol = reduce_solution(sol)
        
        # Apply complement packing improvement
        if time.time() < deadline - 10:
            sol = complement_packing_improve(sol, max_iters=30)
            sol = reduce_solution(sol)
        
        # Apply local search
        if time.time() < deadline - 5:
            sol = local_search_improve(sol, max_iters=15)
            sol = reduce_solution(sol)
            
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
        if best_size <= 140:
            break

    if best_solution is None:
        return [list(b) for b in k_blocks]
        
    return to_blocks(best_solution)
