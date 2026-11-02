"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Generate all integer vectors in R^11 with coordinates in {-1, 0, 1}
    # That's 3^11 = 177147 vectors. But we can be smarter.
    
    # Actually, let's generate vectors with norm squared up to some limit.
    # The kissing number in dimension 11 is known to be 456.
    # Let's try to construct a good set.
    
    # Strategy: Generate candidate vectors with small norms and find a large
    # subset where min pairwise distance >= max norm.
    
    # Let's generate all vectors with coordinates in {-2, -1, 0, 1, 2}
    # but that's 5^11 = 48828125 which is too many.
    
    # Better: generate vectors with norm squared <= some threshold.
    # Let's start with norm squared <= 4 (coordinates in {-2, -1, 0, 1, 2} with small norm)
    
    # Actually, let's use a different approach.
    # Generate vectors with exactly 2 or 3 nonzero coordinates.
    
    candidates = []
    seen = set()
    
    # Vectors with 2 nonzero coordinates: values in {-2, -1, 1, 2}
    for i in range(D):
        for j in range(i+1, D):
            for vi in (-2, -1, 1, 2):
                for vj in (-2, -1, 1, 2):
                    v = [0]*D
                    v[i] = vi
                    v[j] = vj
                    vtuple = tuple(v)
                    if vtuple not in seen:
                        seen.add(vtuple)
                        candidates.append(v)
    
    # Vectors with 3 nonzero coordinates: values in {-1, 1}
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for vi in (-1, 1):
                    for vj in (-1, 1):
                        for vk in (-1, 1):
                            v = [0]*D
                            v[i] = vi
                            v[j] = vj
                            v[k] = vk
                            vtuple = tuple(v)
                            if vtuple not in seen:
                                seen.add(vtuple)
                                candidates.append(v)
    
    C = np.array(candidates, dtype=np.int64)
    n = len(C)
    
    # Compute squared norms
    norms2 = np.sum(C**2, axis=1)
    
    # We want min pairwise distance >= max norm
    # i.e., for all i != j: ||v_i - v_j||^2 >= max_norm2
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    
    # Let's group by norm squared
    unique_norms = np.unique(norms2)
    
    # Start with the smallest norm group and try to build up
    # Actually, let's just try to find a large independent set in a conflict graph
    
    # But first, let's try a simpler approach: pick all vectors of the same norm
    # and check if min distance >= norm
    
    best_result = None
    
    for norm2 in unique_norms:
        if time.time() > deadline:
            break
        mask = norms2 == norm2
        subset = C[mask]
        m = len(subset)
        if m < 2:
            continue
        
        # Check if min pairwise distance >= norm2
        # dist2 = norms2_i + norms2_j - 2*dot = 2*norm2 - 2*dot
        # Need 2*norm2 - 2*dot >= norm2 => norm2 >= 2*dot => dot <= norm2/2
        dots = subset @ subset.T
        # For i != j, need dots[i,j] <= norm2/2
        np.fill_diagonal(dots, -np.inf)
        max_offdiag = np.max(dots)
        if max_offdiag <= norm2/2:
            if best_result is None or m > len(best_result):
                best_result = subset
    
    # Now try to augment with mixed norms
    # Use a greedy approach with local search
    
    if time.time() < deadline - 5:
        # Build conflict graph for candidates
        # Two candidates conflict if dist2 < max(norms2_i, norms2_j)
        # Actually the constraint is: min pairwise dist >= MAX norm in the set
        # So if we have a set S, let M = max(norms2 in S)
        # For all i,j in S: dist2(i,j) >= M
        
        # This is hard to check incrementally. Let's try a different approach.
        
        # Try to find a large set where all vectors have norm <= T
        # and pairwise dist2 >= T
        
        # For each possible T, find max subset with norms2 <= T and pairwise dist2 >= T
        
        # Let's try T values
        for T in range(1, 9):
            if time.time() > deadline:
                break
            mask = norms2 <= T
            subset = C[mask]
            m = len(subset)
            if m < 2:
                continue
            
            # Build conflict graph: i,j conflict if dist2(i,j) < T
            dots = subset @ subset.T
            norms_sub = norms2[mask]
            dist2 = norms_sub[:, None] + norms_sub[None, :] - 2*dots
            np.fill_diagonal(dist2, np.inf)
            
            conflict = dist2 < T
            adj = [set(np.where(conflict[i])[0]) for i in range(m)]
            
            # Greedy independent set
            selected = []
            available = set(range(m))
            for i in sorted(range(m), key=lambda x: len(adj[x])):
                if i in available:
                    selected.append(i)
                    available.discard(i)
                    for j in adj[i]:
                        available.discard(j)
            
            sel_vecs = subset[selected]
            if len(sel_vecs) > 0:
                if best_result is None or len(sel_vecs) > len(best_result):
                    best_result = sel_vecs
    
    if best_result is None:
        # Fallback: just return the seed-style vectors
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0]*D
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return best_result
# EVOLVE-BLOCK-END
