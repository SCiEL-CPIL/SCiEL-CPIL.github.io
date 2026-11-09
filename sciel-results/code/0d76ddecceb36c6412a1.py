"""Construct a minimum maximal no-isosceles set on an N x N grid."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random

def _precompute_conflicts(n):
    """
    Precompute conflict pairs.
    conflict_pair[i][j] is a bitmask of points k such that {i, j, k} is isosceles.
    Note: i, j, k are distinct.
    """
    n2 = n * n
    points = [(x, y) for x in range(n) for y in range(n)]
    
    # dist2[i] = squared distance from point i to all others
    # Actually, we just need distances between pairs.
    dist2 = np.zeros((n2, n2), dtype=np.int32)
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            dist2[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2
            
    # conflict_pair[i][j] : bitmask of k
    # Using Python integers for bitmasks is efficient for n2 <= 256 (fits in 256 bits, Python handles big ints).
    conflict_pair = [[0] * n2 for _ in range(n2)]
    
    for i in range(n2):
        for j in range(i + 1, n2):
            mask = 0
            di_j = dist2[i, j]
            for k in range(n2):
                if k == i or k == j:
                    continue
                di_k = dist2[i, k]
                dj_k = dist2[j, k]
                if di_j == di_k or di_j == dj_k or di_k == dj_k:
                    mask |= (1 << k)
            conflict_pair[i][j] = mask
            conflict_pair[j][i] = mask
            
    return points, conflict_pair

def construct():
    n = N
    n2 = n * n
    full_mask = (1 << n2) - 1
    points, conflict_pair = _precompute_conflicts(n)
    
    best_set = None
    best_size = n2 + 1
    
    # Helper to check validity of a set represented by a bitmask and a list of indices
    def is_valid_no_isosceles(mask, indices):
        """Check if the set defined by 'indices' (and mask) has no isosceles triples."""
        for i in range(len(indices)):
            a = indices[i]
            for j in range(i + 1, len(indices)):
                b = indices[j]
                # Check if any k in the set is in conflict_pair[a][b]
                if conflict_pair[a][b] & mask:
                    return False
        return True

    def is_maximal(mask, indices):
        """Check if the set is maximal: every point not in mask is covered by some pair in indices."""
        # Calculate coverage: union of conflict_pair[a][b] for all pairs in indices
        coverage = 0
        for i in range(len(indices)):
            a = indices[i]
            for j in range(i + 1, len(indices)):
                b = indices[j]
                coverage |= conflict_pair[a][b]
        
        # All points not in mask must be in coverage
        # i.e., coverage | mask == full_mask
        if (coverage | mask) != full_mask:
            return False
        return True

    def greedy_construct(order):
        S = []
        mask = 0
        for p in order:
            # Check if adding p keeps S no-isosceles
            valid = True
            for a in S:
                if conflict_pair[a][p] & mask:
                    valid = False
                    break
            if valid:
                S.append(p)
                mask |= (1 << p)
        return S, mask

    # Multiple random restarts
    for seed in range(500):
        rng = random.Random(seed)
        order = list(range(n2))
        rng.shuffle(order)
        
        S, mask = greedy_construct(order)
        
        # Local search: try to remove points
        # We want to minimize size while maintaining validity and maximality
        improved = True
        while improved:
            improved = False
            # Shuffle indices to try removals in random order? Or fixed?
            # Fixed order is deterministic per state. Let's iterate through indices.
            for i in range(len(S)):
                p = S[i]
                new_indices = S[:i] + S[i+1:]
                new_mask = mask ^ (1 << p)
                
                # Check no-isosceles
                if not is_valid_no_isosceles(new_mask, new_indices):
                    continue
                    
                # Check maximal
                if not is_maximal(new_mask, new_indices):
                    continue
                
                # Accept
                S = new_indices
                mask = new_mask
                improved = True
                break # Restart loop from beginning as indices changed
                
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
        # Additional local search pass with random perturbations? 
        # Maybe try swapping? 
        # For now, simple removal is the primary driver for minimization.
        
    if best_set is None:
        # Fallback: return all points (definitely maximal, not minimal, but valid)
        best_set = list(range(n2))
        
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
