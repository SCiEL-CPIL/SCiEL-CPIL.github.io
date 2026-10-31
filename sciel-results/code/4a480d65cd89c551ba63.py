"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import linprog


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Generate all candidate vectors with small integer entries
    # We'll use vectors with entries in {-2, -1, 0, 1, 2} but to keep it manageable,
    # let's start with a structured set and then expand.
    
    # First, generate the seed set from the current program
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Add more vectors: single non-zero entries
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vectors.append(v)
    
    # Add vectors with two non-zero entries of different magnitudes
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (2, -2):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
            for si in (2, -2):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
    
    # Add vectors with three non-zero entries
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vectors.append(v)
    
    # Add vectors with entries in {-2,-1,1,2} in two positions
    for i, j in itertools.combinations(range(D), 2):
        for si in (-2, -1, 1, 2):
            for sj in (-2, -1, 1, 2):
                if si == 0 and sj == 0:
                    continue
                v = [0] * D
                v[i], v[j] = si, sj
                # Check if not already in list
                v_tuple = tuple(v)
                if v_tuple not in [tuple(x) for x in vectors]:
                    vectors.append(v)
    
    # Deduplicate
    seen = set()
    unique_vectors = []
    for v in vectors:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_vectors.append(v)
    
    vectors = unique_vectors
    n = len(vectors)
    
    # Compute norms squared
    vecs = np.array(vectors, dtype=np.float64)
    norms_sq = np.sum(vecs ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # We need: for all i != j, ||v_i - v_j||^2 >= max_norm_sq
    # i.e., ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm_sq
    # i.e., 2*v_i.v_j <= ||v_i||^2 + ||v_j||^2 - max_norm_sq
    
    # This is a combinatorial problem. Let's use a greedy + local search approach.
    # First, try to find a large subset using a greedy algorithm.
    
    # Sort vectors by norm (larger norms are more restrictive)
    order = np.argsort(-norms_sq)
    
    # Greedy selection
    selected = []
    selected_set = set()
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        if idx in selected_set:
            continue
        # Check if this vector can be added
        can_add = True
        for sel_idx in selected:
            diff = vecs[idx] - vecs[sel_idx]
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm_sq - 1e-9:
                can_add = False
                break
        if can_add:
            selected.append(idx)
            selected_set.add(idx)
    
    # Local search: try to add more vectors or swap
    # Try adding vectors that were skipped
    for idx in order:
        if time.time() > deadline - 3:
            break
        if idx in selected_set:
            continue
        # Check if this vector can be added
        can_add = True
        for sel_idx in selected:
            diff = vecs[idx] - vecs[sel_idx]
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm_sq - 1e-9:
                can_add = False
                break
        if can_add:
            selected.append(idx)
            selected_set.add(idx)
    
    # Now try to swap: remove one selected and add two others
    # This is complex, so let's try a simpler improvement: 
    # Try to find vectors that can replace a selected one to allow more total
    
    # For now, let's also try a different approach: 
    # Use the fact that we can scale vectors. But the problem requires integer vectors.
    # Actually, the problem says "integer vectors" and "normalizing each vector to norm 2".
    # So we need integer vectors, and after normalization they should have pairwise distance >= 1 (since norm is 2, distance >= norm means distance >= 2 in original scale? Let me re-read).
    
    # "minimum pairwise distance is at least the maximum vector norm"
    # So for all i,j: ||v_i - v_j|| >= max_k ||v_k||
    
    # Let's verify our current selection satisfies this.
    if len(selected) > 0:
        selected_vecs = vecs[selected]
        # Check all pairs
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                diff = selected_vecs[i] - selected_vecs[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            result = np.array([vectors[i] for i in selected], dtype=np.int64)
            return result
    
    # Fallback: return just the original seed
    return np.array(vectors[:220], dtype=np.int64)

# EVOLVE-BLOCK-END
