"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    """
    Construct kissing configuration from D11 lattice roots.
    D11 roots are vectors with exactly two non-zero entries of ±1.
    We use a symmetry-preserving greedy selection that maximizes pairwise angular separation.
    """
    
    # Generate all D11 roots: vectors with exactly two non-zero entries ±1
    # For each pair (i,j) with i<j, and each sign combination (±1, ±1)
    candidates = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # All candidates have norm^2 = 2
    norm_sq = 2
    
    # Greedy selection with symmetry-preserving ordering
    # Sort candidates by a deterministic criterion that respects D11 symmetry
    # We'll use the index of the first non-zero entry, then second, then signs
    # This creates a natural ordering that we can use for greedy selection
    
    selected = []
    selected_indices = set()
    
    # Precompute dot products for efficiency
    # dot[i,j] = candidates[i] · candidates[j]
    # For D11 roots, dot product is in {-2, -1, 0, 1, 2}
    # dist_sq = |a-b|^2 = |a|^2 + |b|^2 - 2*a·b = 2 + 2 - 2*a·b = 4 - 2*a·b
    # We need dist_sq >= norm_sq = 2
    # So 4 - 2*a·b >= 2 => a·b <= 1
    
    # For D11 roots:
    # a·b = 2 if a = b
    # a·b = -2 if a = -b
    # a·b = 1 if they share one coordinate with same sign, one with opposite
    # a·b = -1 if they share one coordinate with opposite sign, one with same
    # a·b = 0 if they share no coordinates or share one with one same and one opposite... wait
    # Let me think more carefully.
    
    # Two D11 roots a and b:
    # If they share both coordinates: a·b = ±1*±1 + ±1*±1 = 2, 0, or -2
    #   - If same signs on both: 1+1=2
    #   - If opposite on one: 1-1=0 or -1+1=0
    #   - If opposite on both: -1-1=-2
    # If they share exactly one coordinate: a·b = ±1*±1 = ±1
    # If they share no coordinates: a·b = 0
    
    # So a·b ∈ {-2, -1, 0, 1, 2}
    # dist_sq = 4 - 2*a·b
    # We need dist_sq >= 2, so a·b <= 1
    # This means we cannot have a·b = 2 (i.e., a = b)
    # But we CAN have a·b = 1, 0, -1, -2
    
    # Wait, but if a·b = -2, that means a = -b, and dist_sq = 4-2*(-2)=8 >= 2, OK
    # If a·b = 1, dist_sq = 4-2=2 >= 2, OK
    # If a·b = 0, dist_sq = 4 >= 2, OK
    # If a·b = -1, dist_sq = 4+2=6 >= 2, OK
    
    # So the only constraint is that we can't have duplicate vectors (a·b = 2)
    # But wait, we also need to check all pairs, not just against one.
    
    # Actually, the constraint is that for ALL pairs in the selected set, dist_sq >= 2
    # Which means for all pairs, a·b <= 1
    
    # Let's do greedy: start with empty set, add vectors one by one if they satisfy
    # the constraint with all already-selected vectors
    
    # For efficiency, let's precompute the dot product matrix
    # n = C(11,2) * 4 = 55 * 4 = 220
    # 220x220 matrix is manageable
    
    dot_matrix = candidates @ candidates.T
    
    # Now greedy selection
    # Order: we want to maximize the count. Let's try different orderings.
    # A good strategy: process vectors in an order that groups similar vectors together
    # so that conflicts are detected early.
    
    # Actually, for maximum packing, we should be careful about the order.
    # Let's try: group by the pair (i,j), and within each group, process all 4 sign combos.
    # This way, when we pick one from a group, we know the others in that group will conflict
    # (since they share both coordinates).
    
    # For a given pair (i,j), the 4 vectors are:
    # (1,1), (1,-1), (-1,1), (-1,-1)
    # Dot products between them:
    # (1,1)·(1,1)=2, (1,1)·(1,-1)=0, (1,1)·(-1,1)=0, (1,1)·(-1,-1)=-2
    # (1,-1)·(1,-1)=2, (1,-1)·(-1,1)=-2, (1,-1)·(-1,-1)=0
    # (-1,1)·(-1,1)=2, (-1,1)·(-1,-1)=0
    # (-1,-1)·(-1,-1)=2
    
    # So within a group of 4, we can pick at most 2 (since any two with dot=2 conflict,
    # but (1,1) and (1,-1) have dot=0, so they're compatible)
    # Actually (1,1) and (-1,-1) have dot=-2, compatible
    # (1,1) and (1,-1) have dot=0, compatible
    # So we can pick up to 2 from each group of 4? Let's check:
    # Can we pick 3? (1,1), (1,-1), (-1,1): 
    #   (1,1)·(1,-1)=0 OK, (1,1)·(-1,1)=0 OK, (1,-1)·(-1,1)=-2 OK
    # So yes, we can pick 3 from a group!
    # Can we pick all 4? (1,1), (1,-1), (-1,1), (-1,-1):
    #   (1,1)·(-1,-1)=-2 OK, (1,-1)·(-1,1)=-2 OK, (1,-1)·(-1,-1)=0 OK, (-1,1)·(-1,-1)=0 OK
    # Wait, but (1,1)·(1,1)=2 is not a pair, it's the same vector.
    # All distinct pairs: 
    #   (1,1)·(1,-1)=0, (1,1)·(-1,1)=0, (1,1)·(-1,-1)=-2
    #   (1,-1)·(-1,1)=-2, (1,-1)·(-1,-1)=0
    #   (-1,1)·(-1,-1)=0
    # All <= 1, so all 4 are mutually compatible!
    
    # So within each pair (i,j), we can pick all 4 vectors.
    # But vectors from different pairs may conflict.
    
    # Let's just do the greedy with the dot matrix.
    
    # Order the candidates by their index in the original list
    # which is: for each (i,j) in combinations, for si in (1,-1), for sj in (1,-1)
    
    # Try a few different orderings and pick the best
    
    best_selected = []
    best_count = 0
    
    # Ordering 1: natural order
    order1 = list(range(n))
    
    # Ordering 2: reverse
    order2 = list(range(n-1, -1, -1))
    
    # Ordering 3: group by first coordinate i
    # Reorder so that all vectors with the same first non-zero coordinate are together
    order3 = []
    for i in range(D):
        for idx in range(n):
            # Find first non-zero position
            nz = np.nonzero(candidates[idx])[0]
            if len(nz) >= 1 and nz[0] == i:
                order3.append(idx)
    
    for order in [order1, order2, order3]:
        selected_idx = []
        for idx in order:
            if idx in selected_indices:
                continue
            # Check if candidates[idx] is compatible with all selected
            compatible = True
            for sel_idx in selected_idx:
                if dot_matrix[idx, sel_idx] > 1:
                    compatible = False
                    break
            if compatible:
                selected_idx.append(idx)
        
        if len(selected_idx) > best_count:
            best_count = len(selected_idx)
            best_selected = selected_idx
    
    if best_count > 0:
        result = candidates[best_selected]
        return result
    else:
        # Fallback: return all candidates (they should be valid)
        return candidates
# EVOLVE-BLOCK-END
