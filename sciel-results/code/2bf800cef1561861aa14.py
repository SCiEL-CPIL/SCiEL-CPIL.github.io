"""
Kissing number in dimension 11: find integer vectors in R^11 whose minimum pairwise
distance is at least the maximum vector norm.

The seed program generates the 220 weight-2 vectors (D11 minimal vectors).
We can do better by considering other lattice structures.

Key insight: The kissing number of R^11 is known to be 432. The Leech lattice
projection or related constructions can yield more vectors.

However, a simpler approach: consider vectors with entries in {-2, -1, 0, 1, 2}
and find a large set where min pairwise distance >= max norm.

Actually, let me think about this more carefully.

If all vectors have the same norm r, then we need pairwise distances >= r.
For integer vectors, the norm squared is an integer. If we use vectors with
squared norm = 4 (i.e., weight-2 vectors with entries ±1, ±1), that's the seed.

What about squared norm = 2? Those are weight-1 vectors (±e_i), only 22 of them,
and pairwise distance between e_i and e_j is sqrt(2) < sqrt(2)... wait, distance
is sqrt(2) and max norm is sqrt(2), so distance = max norm. That works! But only 22.

What about mixing? If we use vectors of different norms, the constraint is:
min_{i != j} ||v_i - v_j|| >= max_k ||v_k||

So if we have vectors of norm r_max, all pairwise distances must be >= r_max.

For the D11 construction: 220 vectors of norm 2, pairwise distances are either
2 (when they share one coordinate) or 2*sqrt(2) (when disjoint). Min distance = 2,
max norm = 2. So it works.

Can we add more vectors? The kissing number of R^11 is 432, but that's for
real vectors. For integer vectors, we're restricted to Z^11.

Let me try a greedy/randomized approach to find a larger set.

Strategy:
1. Start with the 220 D11 vectors.
2. Try to add vectors of norm 2 or 4 that maintain the constraint.
3. Use randomized local search to improve.

Actually, let me think about what vectors of norm 2 exist in Z^11:
- Weight 2: entries ±1, ±1 in two positions -> 220 vectors (the D11 ones)
- Weight 1 with entry ±2: ±2*e_i -> 22 vectors

Check: distance between (2,0,...,0) and (1,1,0,...,0):
||v - w||^2 = (2-1)^2 + (0-1)^2 = 1 + 1 = 2, so distance = sqrt(2) < 2 = max norm.
Doesn't work.

Check: distance between (2,0,...,0) and (1,-1,0,...,0):
||v - w||^2 = (2-1)^2 + (0-(-1))^2 = 1 + 1 = 2, same issue.

So we can't mix norm-squared-4 vectors with norm-squared-2 vectors easily.

What about all norm-squared-4 vectors? Weight 4 vectors with ±1, or weight 1 with ±2, etc.
This gets complex. Let me try a computational approach.

Given the time constraint (150s), I'll implement:
1. The seed 220 vectors as baseline.
2. A randomized greedy algorithm that tries to add more vectors.
3. Local search / simulated annealing to improve.

But first, let me check: is 220 actually optimal for integer vectors?

The kissing number problem for Z^n is different from R^n. For Z^11, the answer
might be higher than 220.

Let me try a different approach: use a greedy algorithm that builds the set
incrementally, checking the constraint at each step.

For efficiency, I'll precompute norms and use numpy for distance calculations.
"""

import time
import numpy as np
import itertools
from collections import defaultdict

D = 11
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Generate all candidate vectors with small entries
    # Focus on vectors with norm squared <= some bound
    # Start with the known good set: D11 minimal vectors (weight-2, norm^2 = 2)
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Phase 1: Seed with D11 vectors (220 vectors, norm^2 = 2)
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    
    best_set = seed_vectors.copy()
    best_norm_sq = 2  # All have norm^2 = 2
    best_min_dist_sq = 2  # Min pairwise distance squared
    
    # Phase 2: Try to add more vectors via greedy search
    # Generate additional candidates
    # Try vectors with entries in {-2, -1, 0, 1, 2}
    
    # For efficiency, generate candidates with limited weight
    candidates = []
    
    # Weight 1: ±e_i, ±2e_i
    for i in range(D):
        for val in (-2, -1, 1, 2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
    
    # Weight 2: various combinations
    for i, j in itertools.combinations(range(D), 2):
        for vi in (-2, -1, 1, 2):
            for vj in (-2, -1, 1, 2):
                if vi == 0 or vj == 0:
                    continue
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = vi, vj
                candidates.append(v)
    
    # Weight 3: 
    for i, j, k in itertools.combinations(range(D), 3):
        for vi in (-1, 1):
            for vj in (-1, 1):
                for vk in (-1, 1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = vi, vj, vk
                    candidates.append(v)
    
    # Remove duplicates
    candidate_set = set()
    for v in candidates:
        candidate_set.add(tuple(v))
    candidates = [np.array(v, dtype=np.int64) for v in candidate_set]
    
    # Filter: only keep vectors that could potentially be added
    # A vector v can be added if min_{u in best_set} ||v - u|| >= sqrt(max(norm^2(v), best_norm_sq))
    
    def can_add(v, current_set, current_max_norm_sq):
        """Check if v can be added to current_set maintaining the constraint."""
        v_norm_sq = int(np.dot(v, v))
        target_dist_sq = max(v_norm_sq, current_max_norm_sq)
        
        for u in current_set:
            diff = v - u
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < target_dist_sq:
                return False
        return True
    
    # Greedy: try to add candidates one by one
    for v in candidates:
        if not time_ok():
            break
        v_norm_sq = int(np.dot(v, v))
        
        # Skip if this vector is already in our set
        if any(np.array_equal(v, u) for u in best_set):
            continue
        
        # Try adding this vector
        if can_add(v, best_set, best_norm_sq):
            best_set.append(v)
            if v_norm_sq > best_norm_sq:
                best_norm_sq = v_norm_sq
                # After changing max norm, need to revalidate all pairs
                # For now, this is an approximation; do a full check later
    
    # Phase 3: Full validation and cleanup
    # Remove vectors that violate the constraint
    def validate_and_cleanup(vector_list, max_norm_sq):
        """Remove vectors that violate the constraint."""
        if len(vector_list) <= 1:
            return vector_list
        
        # Compute all pairwise distances
        n = len(vector_list)
        arr = np.array(vector_list)
        
        # Distance matrix
        # d(i,j)^2 = ||vi||^2 + ||vj||^2 - 2 vi·vj
        norms_sq = np.sum(arr * arr, axis=1)
        dots = arr @ arr.T
        
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        
        if min_dist_sq < max_norm_sq:
            # Need to remove some vectors
            # Greedy removal: remove the vector involved in the closest pair
            # This is approximate; for a proper solution, we'd need a more careful approach
            return vector_list  # Keep as is for now; the greedy addition should have been careful
        
        return vector_list
    
    # Do a final greedy pass with better ordering
    # Sort candidates by norm squared (prefer smaller norms)
    candidates.sort(key=lambda v: int(np.dot(v, v)))
    
    final_set = []
    final_max_norm_sq = 0
    
    for v in candidates:
        if not time_ok():
            break
        
        v_norm_sq = int(np.dot(v, v))
        
        # Skip if already in set
        if any(np.array_equal(v, u) for u in final_set):
            continue
        
        # Try adding
        if can_add(v, final_set, final_max_norm_sq):
            final_set.append(v)
            if v_norm_sq > final_max_norm_sq:
                final_max_norm_sq = v_norm_sq
    
    # Also try the original seed set
    if len(seed_vectors) > len(final_set):
        final_set = seed_vectors
        final_max_norm_sq = 2
    
    # Final validation
    if len(final_set) > 1:
        arr = np.array(final_set)
        norms_sq = np.sum(arr * arr, axis=1)
        dots = arr @ arr.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = int(np.min(dist_sq))
        max_norm_sq = int(np.max(norms_sq))
        
        if min_dist_sq < max_norm_sq:
            # Fallback to seed
            final_set = seed_vectors
    
    result = np.array(final_set, dtype=np.int64)
    
    if not time_ok():
        # Return the seed if we're running out of time
        return np.array(seed_vectors, dtype=np.int64)
    
    return result

