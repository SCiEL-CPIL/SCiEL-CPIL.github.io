"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools
from itertools import combinations, product

def _make_candidates(max_norm_sq=4, seed=42):
    """Generate all integer vectors in R^D with 0 < ||v||^2 <= max_norm_sq."""
    candidates = []
    
    # Norm^2 = 1: unit vectors and their negatives
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(tuple(v))
    
    # Norm^2 = 2: pairs
    for i, j in combinations(range(D), 2):
        for si, sj in product((1,-1), (1,-1)):
            v = [0]*D
            v[i], v[j] = si, sj
            candidates.append(tuple(v))
    
    # Norm^2 = 3: triples
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1,-1),(1,-1),(1,-1)):
            v = [0]*D
            v[i], v[j], v[k] = si, sj, sk
            candidates.append(tuple(v))
    
    # Norm^2 = 4: quadruples (weight 4 vectors)
    for indices in combinations(range(D), 4):
        for signs in product((1,-1), repeat=4):
            v = [0]*D
            for idx, val in zip(indices, signs):
                v[idx] = val
            candidates.append(tuple(v))
    
    return candidates

def _norm_sq(v):
    return sum(x*x for x in v)

def _dist_sq(u, v):
    return sum((a-b)*(a-b) for a, b in zip(u, v))

def _greedy_max_independent(candidates, deadline=145.0, n_trials=50):
    """
    Find a large subset of candidates where min pairwise distance >= max norm.
    The constraint: for all pairs u,v in S, ||u-v|| >= ||max|| = max_{w in S} ||w||
    
    We use a greedy approach with multiple random orderings.
    """
    start_time = time.time()
    best_set = []
    best_size = 0
    
    candidates = list(candidates)
    n = len(candidates)
    
    # Precompute norms
    norms_sq = [_norm_sq(v) for v in candidates]
    
    # Precompute pairwise distance squared matrix for efficiency
    # This is O(n^2 * D) which might be expensive, so we do it lazily
    # Actually, let's compute it in chunks or use a different approach
    
    # For each trial, we try a different ordering
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            # First trial: sort by norm ascending (prefer smaller norms)
            order = sorted(range(n), key=lambda i: norms_sq[i])
        elif trial == 1:
            # Second trial: sort by norm descending
            order = sorted(range(n), key=lambda i: -norms_sq[i])
        else:
            # Random shuffles
            rng = random.Random(42 + trial)
            order = list(range(n))
            rng.shuffle(order)
        
        selected = []
        selected_norm_sq_max = 0
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            # Check if adding v would violate the constraint
            # New max norm would be max(selected_norm_sq_max, v_norm_sq)
            new_max_norm_sq = max(selected_norm_sq_max, v_norm_sq)
            
            # Need ||u - v||^2 >= new_max_norm_sq for all u in selected
            compatible = True
            for u_idx in selected:
                u = candidates[u_idx]
                if _dist_sq(u, v) < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_norm_sq_max = new_max_norm_sq
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    return [candidates[i] for i in best_set]

def _greedy_with_conflict_matrix(candidates, deadline=145.0, n_trials=100):
    """
    More efficient greedy using precomputed conflict matrix.
    Conflict[i,j] = True if vectors i and j cannot both be in the set.
    Two vectors conflict if ||vi - vj||^2 < max(||vi||^2, ||vj||^2)
    But the actual constraint is ||vi - vj|| >= max_norm_in_set, which is global.
    
    However, a sufficient condition for compatibility is:
    ||vi - vj||^2 >= max(||vi||^2, ||vj||^2)
    This ensures they can coexist regardless of other vectors in the set.
    """
    start_time = time.time()
    n = len(candidates)
    
    norms_sq = np.array([_norm_sq(v) for v in candidates], dtype=np.int64)
    cand_array = np.array(candidates, dtype=np.int64)
    
    # Compute dot product matrix
    dot = cand_array @ cand_array.T
    
    # ||vi - vj||^2 = ||vi||^2 + ||vj||^2 - 2*vi.vj
    # We need ||vi - vj||^2 >= max(||vi||^2, ||vj||^2)
    # i.e., ||vi||^2 + ||vj||^2 - 2*vi.vj >= max(||vi||^2, ||vj||^2)
    # i.e., min(||vi||^2, ||vj||^2) - 2*vi.vj >= 0
    # i.e., 2*vi.vj <= min(||vi||^2, ||vj||^2)
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflict = (2 * dot > min_norms_sq)
    np.fill_diagonal(conflict, False)
    
    # Greedy maximum independent set
    best_set = []
    best_size = 0
    
    # Try different orderings
    degrees = conflict.sum(axis=1)
    
    # Ordering 1: Ascending degree (fewest conflicts first)
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            order = np.argsort(degrees)
        elif trial == 1:
            order = np.argsort(-degrees)
        else:
            rng = np.random.default_rng(42 + trial)
            order = rng.permutation(n)
        
        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            if not np.any(conflict[idx, selected_mask]):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    return [candidates[i] for i in best_set]

def _verify_solution(vectors):
    """Verify that the solution satisfies the constraint."""
    if len(vectors) < 2:
        return True
    
    max_norm_sq = max(_norm_sq(v) for v in vectors)
    for i, j in combinations(range(len(vectors)), 2):
        if _dist_sq(vectors[i], vectors[j]) < max_norm_sq:
            return False
    return True

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Return as int64 array of shape (k, d).
    """
    deadline = 145.0  # 5s safety margin before 150s kill
    start_time = time.time()
    
    # Generate candidates with norm^2 up to 4
    candidates = _make_candidates(max_norm_sq=4, seed=42)
    
    # Use greedy with conflict matrix for better performance
    best_set = _greedy_with_conflict_matrix(candidates, deadline=deadline, n_trials=100)
    
    if not best_set:
        # Fallback: seed solution (norm^2=2 pairs only)
        vectors = []
        for i, j in combinations(range(D), 2):
            for si, sj in product((1,-1), (1,-1)):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
        best_set = vectors
    
    # Verify the result satisfies the constraint
    if not _verify_solution(best_set):
        # Fall back to seed solution
        vectors = []
        for i, j in combinations(range(D), 2):
            for si, sj in product((1,-1), (1,-1)):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
        best_set = vectors
    
    return np.array(best_set, dtype=np.int64)
# EVOLVE-BLOCK-END
