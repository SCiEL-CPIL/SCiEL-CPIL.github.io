"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # 1. Precompute comparability matrix M
    # M[i, j] = 1 if i and j are comparable (strict subset/superset)
    # i != j is required.
    
    # Vectorized construction
    # Create column vector of all masks and row vector of all masks
    i_masks = np.arange(total_subsets, dtype=np.int32)[:, None]
    j_masks = np.arange(total_subsets, dtype=np.int32)[None, :]
    
    # Check subset relations
    # i is subset of j if (i & j) == i
    # j is subset of i if (i & j) == j
    is_subset_ij = (i_masks & j_masks) == i_masks
    is_subset_ji = (i_masks & j_masks) == j_masks
    
    # Comparable if either is true, but not if i == j (handled by diagonal)
    # Actually if i==j, both are true. We want strict.
    # So M[i, j] = 1 if (is_subset_ij OR is_subset_ji) AND i != j
    comparable = is_subset_ij | is_subset_ji
    
    # Zero out the diagonal
    np.fill_diagonal(comparable, False)
    
    M = comparable.astype(np.int8)
    
    # 2. Iterative Minimum Degree Removal
    active = np.ones(total_subsets, dtype=bool)
    # Initial degrees
    degrees = M.sum(axis=1).astype(np.int32)
    
    # We need to remove (total_subsets - m) nodes
    removals = total_subsets - m
    
    for _ in range(removals):
        # Find active node with minimum degree
        # Set inactive nodes to infinity so they are not picked
        temp_degrees = degrees.copy()
        temp_degrees[~active] = np.iinfo(np.int32).max
        
        u = int(np.argmin(temp_degrees))
        
        # Remove u
        active[u] = False
        
        # Update degrees of all remaining active nodes
        # degree[v] -= M[v, u]
        # We can do this vectorized:
        # M[:, u] is a column vector of length total_subsets
        # We only update active nodes
        degrees[active] -= M[active, u]
        
    # 3. Extract the 40 remaining masks
    remaining_masks = np.where(active)[0]
    
    # 4. Convert to lists of integers
    result = []
    for mask in remaining_masks:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
