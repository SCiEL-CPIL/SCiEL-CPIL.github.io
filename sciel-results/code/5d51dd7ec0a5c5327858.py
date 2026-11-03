import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate all weight-3 vectors
    # Indices 0..10
    # Signs +1/-1
    vectors = []
    vec_to_idx = {}
    
    # C(11,3) = 165 combinations
    for indices in itertools.combinations(range(D), 3):
        i, j, k = indices
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    key = (i, j, k, si, sj, sk)
                    idx = len(vectors)
                    vec_to_idx[key] = idx
                    vectors.append(v)
                    
    N = len(vectors) # 1320
    
    # 2. Build conflict graph
    # Conflict if share 2 indices and signs match on those 2.
    # We can compute this by iterating pairs? 1320^2 / 2 is ~800k. Fast enough.
    # Or use the structure.
    neighbors = [[] for _ in range(N)]
    
    # Map from (index1, index2, sign1, sign2) to list of vector indices having this "face"
    # Actually, we just need to know if two vectors conflict.
    # v1: (i,j,k, si,sj,sk)
    # v2: (i,j,l, si,sj,sl) -> Conflict if si==si, sj==sj.
    
    # Efficient construction:
    # For each vector, identify its 3 "faces" (pairs of indices + signs).
    # Face 1: (i, j, si, sj)
    # Face 2: (i, k, si, sk)
    # Face 3: (j, k, sj, sk)
    # Two vectors conflict if they share a face.
    # Wait, if they share a face, they share 2 indices and signs.
    # Do they conflict? Yes.
    # Is it possible they share 2 indices but NOT signs?
    # If they share indices i,j but si != si', then no conflict (dist sq >= 6).
    # So conflict <=> share a face (pair of indices with specific signs).
    
    face_to_vecs = {}
    for idx, v in enumerate(vectors):
        i, j, k = None, None, None
        # Recover indices from vector
        non_zero = np.nonzero(v)[0]
        i, j, k = non_zero[0], non_zero[1], non_zero[2]
        si, sj, sk = v[i], v[j], v[k]
        
        f1 = (i, j, si, sj)
        f2 = (i, k, si, sk)
        f3 = (j, k, sj, sk)
        
        for f in (f1, f2, f3):
            if f not in face_to_vecs:
                face_to_vecs[f] = []
            face_to_vecs[f].append(idx)
            
    # Now, for each face, all vectors in that list conflict with each other.
    for f, vecs in face_to_vecs.items():
        if len(vecs) > 1:
            for a in range(len(vecs)):
                for b in range(a+1, len(vecs)):
                    u = vecs[a]
                    w = vecs[b]
                    neighbors[u].append(w)
                    neighbors[w].append(u)
                    
    # 3. Greedy Selection
    selected = np.zeros(N, dtype=bool)
    conf_counts = np.zeros(N, dtype=np.int32)
    
    order = np.random.permutation(N)
    for idx in order:
        if conf_counts[idx] == 0:
            selected[idx] = True
            for nb in neighbors[idx]:
                conf_counts[nb] += 1
                
    current_size = np.sum(selected)
    
    # 4. Local Search / Shake
    # Try to add free vectors
    # Then if stuck, shake
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add any free vector
        free_indices = np.where((~selected) & (conf_counts == 0))[0]
        if len(free_indices) > 0:
            # Pick one (or all? Picking one and updating is safer for counts)
            # Actually, we can pick all that are mutually non-conflicting?
            # Simpler: pick one at a time, or a random one.
            # Let's pick a random free one to vary the path.
            pick = free_indices[rng.integers(len(free_indices))]
            selected[pick] = True
            for nb in neighbors[pick]:
                conf_counts[nb] += 1
            current_size += 1
            improved = True
            # Don't break, try to find more
            
        # Phase 2: If no free vectors, try to swap or shake
        if not improved:
            # Try 1-for-1 swaps to escape local optima?
            # Or just Shake: Remove a random selected, then add all possible.
            # Let's do a limited shake.
            
            # Find selected vectors
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
                
            # Remove a random selected vector
            remove_idx = sel_indices[rng.integers(len(sel_indices))]
            selected[remove_idx] = False
            for nb in neighbors[remove_idx]:
                conf_counts[nb] -= 1
            current_size -= 1
            
            # Now try to add as many as possible (Greedy on the freed space)
            # We can just loop the "Add free" phase a few times
            for _ in range(10):
                free_indices = np.where((~selected) & (conf_counts == 0))[0]
                if len(free_indices) == 0:
                    break
                pick = free_indices[rng.integers(len(free_indices))]
                selected[pick] = True
                for nb in neighbors[pick]:
                    conf_counts[nb] += 1
                current_size += 1
            improved = True # We changed state
            
    # Convert to array
    result_vectors = [vectors[i] for i in range(N) if selected[i]]
    return np.array(result_vectors, dtype=np.int64)
