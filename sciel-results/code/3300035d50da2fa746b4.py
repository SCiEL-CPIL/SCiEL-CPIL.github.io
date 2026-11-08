import numpy as np
from itertools import combinations, product
import time
import sys

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries, each +/- 1."""
    vectors = []
    for pos in combinations(range(D), k):
        for signs in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = 1 if (signs >> i) & 1 else -1
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_pack_vectorized(vectors, max_dot, n_trials=100, seed=42):
    """
    Greedily select a subset of vectors such that pairwise dot product <= max_dot.
    Uses vectorized operations for speed.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    N = len(vectors)
    dots = vectors @ vectors.T
    
    best_count = 0
    best_indices = np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    
    for trial in range(n_trials):
        perm = rng.permutation(N)
        current = []
        
        for idx in perm:
            if len(current) == 0:
                current.append(idx)
                continue
            
            # Check compatibility with all currently selected
            # dots[idx, current] <= max_dot for all
            if np.all(dots[idx, current] <= max_dot):
                current.append(idx)
        
        if len(current) > best_count:
            best_count = len(current)
            best_indices = np.array(current, dtype=np.int64)
            
    return vectors[best_indices].copy()

def _leech_projection_search(n_rounds=5, seed=123):
    """
    Generate candidate vectors by projecting Leech lattice minimal vectors
    onto 11D, then rounding to nearest integers.
    """
    # Leech lattice minimal vectors: (4,0,...,0) permutations and (±2,±2,0,...,0) with specific signs.
    # There are 196,560 minimal vectors of norm squared 4.
    # Construction:
    # Type 1: (±4, 0, ..., 0) : 24 * 2 = 48
    # Type 2: (±2, ±2, 0, ..., 0) with sum of coordinates ≡ 0 mod 4 (which is always true for ±2,±2)
    #         Number of pairs: C(24,2) = 276. For each pair, 4 sign combinations.
    #         But we must ensure the vector is in the Leech lattice.
    #         The condition for (±2, ±2, 0,...) to be in Leech is that the number of +2's is even? 
    #         Actually, the Leech lattice contains (2,2,0,...), (2,-2,0,...), etc.
    #         Total type 2: 276 * 4 = 1104? No, that's not 196,560.
    #         Wait, 48 + 1104 = 1152. That's too few.
    #         Ah, the full set of minimal vectors in Leech lattice includes more types?
    #         No, minimal norm is 4.
    #         Let me re-check. 196,560 is correct for Leech minimal vectors.
    #         Type 1: 24 * 2 = 48.
    #         Type 2: (±2, ±2, 0, ..., 0). There are 24*23/2 = 276 pairs.
    #                 For each pair, there are 4 sign patterns: (++), (+-), (-+), (--).
    #                 But not all are in the Leech lattice.
    #                 The Leech lattice is defined as vectors in Z^24 with sum ≡ 0 mod 4 and sum of squares ≡ 0 mod 8?
    #                 Actually, a common construction uses the Gleason-Pless construction.
    #                 Let's just use a simpler approach: generate a large random subset of integer vectors
    #                 that are "close" to Leech projections, or just skip Leech and use smarter greedy.
    #         
    # Given time constraints and complexity of exact Leech generation, let's use a different heuristic:
    # We know that weight-4 vectors with dot <= 2 can form large sets.
    # Let's try a more sophisticated packing for weight-4 vectors using local search.
    
    # For now, return empty to indicate this path is not fully implemented in this iteration.
    return np.array([], dtype=np.int64).reshape(0, D)

def _local_search_improve(vectors, max_dot, n_iterations=100, seed=999):
    """
    Try to improve a packing by removing one vector and adding another that fits.
    """
    if len(vectors) == 0:
        return vectors
    
    N = len(vectors)
    dots = vectors @ vectors.T
    
    selected = list(range(N))
    rng = np.random.default_rng(seed)
    
    for _ in range(n_iterations):
        # Try to remove a random vector and add a better one
        if len(selected) < 10:
            break
            
        idx_to_remove = rng.integers(len(selected))
        removed_idx = selected[idx_to_remove]
        
        # Find a candidate that fits with the rest
        rest = [i for i in selected if i != removed_idx]
        
        # Shuffle candidates
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        added = False
        for cand in candidates:
            if cand == removed_idx:
                continue
            # Check if cand fits with rest
            if np.all(dots[cand, rest] <= max_dot):
                selected[idx_to_remove] = cand
                added = True
                break
        
        if not added:
            # Put it back
            selected[idx_to_remove] = removed_idx
            break
            
    return vectors[selected].copy()

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    best_vectors = None
    best_k = 0
    
    # 1. Weight 2 vectors: All 220 are compatible
    v2 = _generate_weight_k_vectors(2)
    best_vectors = v2
    best_k = len(v2)
    
    if time.time() - start_time > timeout:
        return best_vectors

    # 2. Weight 3 vectors: dot <= 1
    v3 = _generate_weight_k_vectors(3)
    packed_v3 = _greedy_pack_vectorized(v3, max_dot=1, n_trials=100, seed=42)
    if len(packed_v3) > best_k:
        best_vectors = packed_v3
        best_k = len(packed_v3)
        
    # Local search on weight 3
    if len(packed_v3) > 0:
        improved_v3 = _local_search_improve(packed_v3, max_dot=1, n_iterations=50, seed=123)
        if len(improved_v3) > best_k:
            best_vectors = improved_v3
            best_k = len(improved_v3)
            
    if time.time() - start_time > timeout:
        return best_vectors

    # 3. Weight 4 vectors: dot <= 2
    v4 = _generate_weight_k_vectors(4)
    packed_v4 = _greedy_pack_vectorized(v4, max_dot=2, n_trials=50, seed=777)
    if len(packed_v4) > best_k:
        best_vectors = packed_v4
        best_k = len(packed_v4)
        
    # Local search on weight 4
    if len(packed_v4) > 0 and time.time() - start_time < timeout - 10:
        improved_v4 = _local_search_improve(packed_v4, max_dot=2, n_iterations=100, seed=888)
        if len(improved_v4) > best_k:
            best_vectors = improved_v4
            best_k = len(improved_v4)

    if time.time() - start_time > timeout:
        return best_vectors

    # 4. Try weight 5 vectors if time permits: dot <= 2
    if time.time() - start_time < timeout - 30:
        v5 = _generate_weight_k_vectors(5)
        # 14784 vectors. Might be slow.
        packed_v5 = _greedy_pack_vectorized(v5, max_dot=2, n_trials=20, seed=555)
        if len(packed_v5) > best_k:
            best_vectors = packed_v5
            best_k = len(packed_v5)

    return best_vectors
