"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 using Leech lattice E8⊕A2⊕A1 decomposition.
    
    The Leech lattice can be constructed as E8 ⊕ A2 ⊕ A1 where:
    - E8: 8-dimensional lattice with minimal norm 2 (vectors with two ±1's or one ±2)
    - A2: 2-dimensional hexagonal lattice projected to integer coordinates
    - A1: 1-dimensional lattice
    
    For integer vectors in R^11 with norm^2 = 4, we can generate candidates from
    the structure of these component lattices and use a vectorized greedy algorithm
    to find a maximal set where all pairwise distances >= max norm.
    
    Key constraint: For vectors with norm^2 = 4, we need ||u-v||^2 >= 4 for all pairs,
    which means <u,v> <= 2.
    """
    # Generate all integer vectors with squared norm 4
    # Type A: one coordinate is ±2, rest 0 (11*2 = 22 vectors)
    # Type B: four coordinates are ±1, rest 0 (C(11,4)*16 = 5280 vectors)
    # Type C: two coordinates are ±1, one is ±... wait, norm^2=4 means sum of squares = 4
    #   - 2^2 = 4: one ±2
    #   - 1^2+1^2+1^2+1^2 = 4: four ±1's
    #   - 1^2+1^2 = 2, not 4
    #   - So only Type A and Type B
    
    candidates = []
    
    # Type A: one ±2, rest 0
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type B: four ±1's, rest 0
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # All have norm^2 = 4, so max_norm_sq = 4
    # Constraint: for all i != j, ||c_i - c_j||^2 >= 4
    # ||c_i - c_j||^2 = 4 + 4 - 2*c_i·c_j = 8 - 2*c_i·c_j >= 4
    # => c_i·c_j <= 2
    
    # Compute all pairwise dot products
    # dot[i,j] = c_i · c_j
    dot = candidates @ candidates.T
    
    # Conflict: two vectors conflict if their dot product > 2
    # We want to select a set where no two selected vectors conflict
    # i.e., for all selected i,j: dot[i,j] <= 2
    
    # Greedy selection with vectorization
    # Start with all candidates, iteratively remove conflicts
    np.random.seed(42)
    
    # Randomize order for better results
    order = np.random.permutation(n_candidates)
    
    selected_mask = np.zeros(n_candidates, dtype=bool)
    selected_count = 0
    
    # For efficiency, process in batches
    # A vector is compatible with all selected if its dot product with all selected <= 2
    
    # Pre-compute: for each candidate, which other candidates conflict with it
    # conflict[i] = set of j where dot[i,j] > 2
    # But this is O(n^2) memory, let's use a different approach
    
    # Vectorized greedy: maintain a "compatible" array
    # compatible[i] = True if candidate i is compatible with all currently selected
    
    compatible = np.ones(n_candidates, dtype=bool)
    
    # Process candidates in randomized order
    for idx in order:
        if not compatible[idx]:
            continue
        
        # Select this candidate
        selected_mask[idx] = True
        selected_count += 1
        
        # Update compatible: candidates that conflict with idx are no longer compatible
        # Conflict if dot[idx, :] > 2
        conflicts = dot[idx, :] > 2
        compatible[conflicts] = False
        # Also mark idx as no longer compatible (already selected)
        compatible[idx] = False
    
    result = candidates[selected_mask]
    return result
# EVOLVE-BLOCK-END
