"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations, product

def _make_candidates(max_norm_sq=3, seed=42):
    """Generate all integer vectors in R^D with 0 < ||v||^2 <= max_norm_sq."""
    rng = random.Random(seed)
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
    return candidates

def _norm_sq(v):
    return sum(x*x for x in v)

def _dist_sq(u, v):
    return sum((a-b)*(a-b) for a, b in zip(u, v))

def _greedy_max_independent(candidates, deadline=145.0, n_trials=5):
    """
    Find a large subset of candidates where min pairwise distance >= max norm.
    The constraint: for all pairs u,v in S, ||u-v|| >= ||max|| = max_{w in S} ||w||
    Since all candidates have norm <= sqrt(3), and the max norm in S is at most sqrt(3),
    we need ||u-v|| >= max_norm_in_S.
    
    A simpler sufficient condition: ensure ||u-v||^2 >= 3 for all pairs.
    Then the constraint holds regardless of which vectors are selected (since max norm^2 <= 3).
    """
    start_time = time.time()
    best_set = []
    best_size = 0
    
    candidates = list(candidates)
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(42 + trial)
        rng.shuffle(candidates)
        
        selected = []
        selected_norm_sq_max = 0
        
        for v in candidates:
            if time.time() > deadline - 2:
                break
            
            v_norm_sq = _norm_sq(v)
            
            # Check if adding v would violate the constraint
            # New max norm would be max(selected_norm_sq_max, v_norm_sq)
            new_max_norm_sq = max(selected_norm_sq_max, v_norm_sq)
            
            # Need ||u - v||^2 >= new_max_norm_sq for all u in selected
            compatible = True
            for u in selected:
                if _dist_sq(u, v) < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_norm_sq_max = new_max_norm_sq
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    return best_set

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Return as int64 array of shape (k, d).
    """
    deadline = 145.0  # 5s safety margin before 150s kill
    start_time = time.time()
    
    # Generate candidates with norm^2 up to 3
    candidates = _make_candidates(max_norm_sq=3, seed=42)
    
    # Use greedy with multiple random orderings
    best_set = _greedy_max_independent(candidates, deadline=deadline, n_trials=5)
    
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
    if len(best_set) >= 2:
        max_norm_sq = max(_norm_sq(v) for v in best_set)
        valid = True
        for i, j in combinations(range(len(best_set)), 2):
            if _dist_sq(best_set[i], best_set[j]) < max_norm_sq:
                valid = False
                break
        if not valid:
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
