import numpy as np
import itertools
import time

D = 11

def generate_q2_vectors():
    """Generate all 220 weight-2 vectors (norm squared 2)."""
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_core_vectors():
    """Generate 5280 weight-4 vectors with 4 non-zeros (norm squared 4)."""
    vectors = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0]*D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_sat_vectors():
    """Generate 22 vectors of form 2*e_i (norm squared 4)."""
    vectors = []
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def find_max_clique(candidates, deadline):
    """
    Find a large clique in the compatibility graph of candidates.
    Compatibility: dot(u, v) <= 2.
    """
    N = len(candidates)
    if N == 0:
        return []
    
    # Compute dot products
    M = candidates @ candidates.T
    
    # Threshold for Q=4 is 2.0
    threshold = 2.0
    
    # Build adjacency matrix (boolean)
    # A[i,j] = True if i != j and M[i,j] <= threshold
    A = (M <= threshold)
    np.fill_diagonal(A, False)
    
    # Greedy Max Clique with random restarts
    best_clique = []
    best_size = 0
    
    # Use a few random restarts to escape local optima
    for _ in range(5):
        if time.time() > deadline:
            break
            
        # Randomize order slightly or use different heuristic
        # We'll use degree-based but shuffle ties or add noise
        degrees = np.sum(A, axis=1)
        # Add small random noise to break ties
        noise = np.random.uniform(0, 1, size=N)
        order = np.argsort(-(degrees + noise))
        
        remaining = set(range(N))
        clique = []
        
        for v in order:
            if time.time() > deadline:
                break
            if v not in remaining:
                continue
            
            clique.append(v)
            
            # Remove v and its non-neighbors (conflicts)
            non_neigh = np.where(~A[v])[0]
            remaining.difference_update(non_neigh)
            
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique

    return best_clique

def construct():
    # Generate 240 vectors from 2*E8 (first 8 dimensions)
    # E8 minimal vectors have weight 2, entries +/-1.
    # Scaled by 2, they have weight 2, entries +/-2.
    e8_vectors = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*11
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
    
    # Generate 12 vectors from 2*D3 (last 3 dimensions, indices 8,9,10)
    # D3 minimal vectors have weight 2, entries +/-1.
    # Scaled by 2, they have weight 2, entries +/-2.
    d3_vectors = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*11
                v[8+i] = 2 * si
                v[9+j] = 2 * sj # Wait, indices are 8,9,10.
                # Let's map local indices 0,1,2 to global 8,9,10.
                # i, j are from 0,1,2.
                # Global indices are 8+i, 8+j.
                # So v[8+i] and v[8+j].
                d3_vectors.append(v)
                
    # Correction for D3 generation
    d3_vectors = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*11
                v[8+i] = 2 * si
                v[8+j] = 2 * sj
                d3_vectors.append(v)

    all_vectors = np.array(e8_vectors + d3_vectors, dtype=np.int64)
    
    return all_vectors
