import numpy as np
import itertools
import random

def construct(v, k, t):
    # 1. Generate all triples and blocks
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Build Incidence Matrix M (num_blocks x num_triples)
    # M[b, t_idx] = 1 if block b covers triple t_idx
    M = np.zeros((num_blocks, num_triples), dtype=np.uint8)
    
    # Map triple to index for faster lookup
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            M[b_idx, t_idx] = 1

    best_solution = None
    best_count = float('inf')
    
    # Run multiple greedy instances with random tie-breaking
    for run in range(20):
        random.seed(run)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            # Calculate coverage count for each block
            # M @ uncovered gives sum of 1s in row b where uncovered is 1
            coverage_counts = M @ uncovered
            
            # Find max coverage
            max_cov = np.max(coverage_counts)
            if max_cov == 0:
                # Should not happen if problem is solvable
                break
            
            # Get all blocks with max coverage
            candidates = np.where(coverage_counts == max_cov)[0]
            
            # Randomly pick one to break symmetry and explore different paths
            chosen_b = random.choice(candidates.tolist())
            
            selected_indices.append(chosen_b)
            
            # Update uncovered
            uncovered[M[chosen_b] == 1] = 0
            
        # Local Search: Remove redundant blocks
        # We work on a list of indices
        S = selected_indices[:]
        
        # Repeat removal until stable
        changed = True
        while changed:
            changed = False
            new_S = []
            for i, b_idx in enumerate(S):
                # Check if block b_idx is redundant
                # Get columns where this block has 1
                cols = np.where(M[b_idx] == 1)[0]
                if len(cols) == 0:
                    continue # Empty block? (Impossible for k>=t)
                
                # Sum coverage of these columns by all OTHER blocks in S
                # Create a mask for other blocks
                other_indices = [j for j in range(len(S)) if j != i]
                if not other_indices:
                    # If it's the only block, we can't remove it unless it covers nothing (impossible)
                    new_S.append(b_idx)
                    continue
                
                # Get submatrix of other blocks
                # To be efficient, we can just sum M[other_indices][:, cols]
                # But slicing a list of indices creates a copy. 
                # For small S (~70), this is fine.
                M_others = M[S[other_indices], :][:, cols]
                counts = M_others.sum(axis=0)
                
                if np.all(counts > 0):
                    # Redundant, do not add to new_S
                    changed = True
                else:
                    new_S.append(b_idx)
            
            S = new_S
            
        # Check validity and count
        if len(S) < best_count:
            # Verify validity (safety check)
            # Covered triples = OR of M[S]
            if len(S) > 0:
                cover_mask = np.any(M[S], axis=0)
                if np.all(cover_mask):
                    best_count = len(S)
                    best_solution = [list(all_blocks[i]) for i in S]
            else:
                # No blocks selected? Only valid if no triples (v<t)
                if num_triples == 0:
                    best_count = 0
                    best_solution = []
                    
    if best_solution is None:
        # Fallback: return a valid but potentially suboptimal solution
        # Just take the first greedy result
        uncovered = np.ones(num_triples, dtype=np.uint8)
        selected_indices = []
        while np.any(uncovered):
            coverage_counts = M @ uncovered
            max_cov = np.max(coverage_counts)
            candidates = np.where(coverage_counts == max_cov)[0]
            chosen_b = candidates[0]
            selected_indices.append(chosen_b)
            uncovered[M[chosen_b] == 1] = 0
        best_solution = [list(all_blocks[i]) for i in selected_indices]
        
    return best_solution
