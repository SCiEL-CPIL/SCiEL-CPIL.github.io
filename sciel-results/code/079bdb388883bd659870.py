"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate E8 lattice vectors (norm^2 = 2 in 8D)
    e8_roots = []
    
    # Type 1: two ±1 entries, rest 0 (112 vectors)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i], v[j] = si, sj
                e8_roots.append(v)
    
    # Type 2: all entries ±1/2, even number of minus signs (128 vectors)
    for signs in itertools.product([1, -1], repeat=8):
        minus_count = sum(1 for s in signs if s == -1)
        if minus_count % 2 == 0:
            v = [s / 2.0 for s in signs]
            e8_roots.append(v)
    
    e8_roots = np.array(e8_roots)
    n_e8 = len(e8_roots)  # 240
    
    # Pad to 11D and scale by 2 to get integers with norm^2=8
    e8_padded = np.zeros((n_e8, D), dtype=np.float64)
    e8_padded[:, :8] = e8_roots
    e8_int = (e8_padded * 2).astype(np.int64)
    
    # Generate candidate vectors in the 3D subspace (dims 8,9,10)
    extra_candidates = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                if a == 0 and b == 0 and c == 0:
                    continue
                norm_sq = a*a + b*b + c*c
                if 1 <= norm_sq <= 8:
                    extra_candidates.append([a, b, c])
    
    extra_candidates = np.array(extra_candidates, dtype=np.int64)
    n_extra = len(extra_candidates)
    
    # Greedy selection of extra vectors with pairwise dist^2 >= 8
    extra_norms_sq = np.sum(extra_candidates**2, axis=1)
    order = np.argsort(-extra_norms_sq)
    extra_sorted = extra_candidates[order]
    
    selected_extra = []
    for i in range(n_extra):
        c = extra_sorted[i]
        if len(selected_extra) == 0:
            selected_extra.append(c)
            continue
        ok = True
        for s in selected_extra:
            dist_sq = np.dot(c, c) + np.dot(s, s) - 2 * np.dot(c, s)
            if dist_sq < 8:
                ok = False
                break
        if ok:
            selected_extra.append(c)
    
    selected_extra = np.array(selected_extra, dtype=np.int64) if selected_extra else np.array([], dtype=np.int64).reshape(0, 3)
    
    # Pad to 11D
    extra_padded = np.zeros((len(selected_extra), D), dtype=np.int64)
    if len(selected_extra) > 0:
        extra_padded[:, 8:11] = selected_extra
    
    # Combine
    all_vectors = np.vstack([e8_int, extra_padded]) if len(extra_padded) > 0 else e8_int
    
    # Verify and fix if needed
    n_total = len(all_vectors)
    if n_total > 0:
        norms_sq = np.sum(all_vectors**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        
        dots = all_vectors @ all_vectors.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = int(np.min(dist_sq))
        
        if min_dist_sq < max_norm_sq:
            # Remove problematic vectors
            # Simple approach: just use E8 vectors which we know are valid
            all_vectors = e8_int
            n_total = len(all_vectors)
    
    # Try to add more vectors from the full 11D space using local search
    if time.time() < deadline - 5.0:
        current = all_vectors
        current_norms_sq = np.sum(current**2, axis=1)
        current_max_norm_sq = int(np.max(current_norms_sq))
        
        # Generate more candidates: vectors with small integer entries
        more_candidates = []
        # Vectors with 1-4 non-zero entries, values in {-2,-1,1,2}
        for support_size in range(1, 5):
            for support in itertools.combinations(range(D), support_size):
                for values in itertools.product([-2, -1, 1, 2], repeat=support_size):
                    norm_sq = sum(v*v for v in values)
                    if 1 <= norm_sq <= current_max_norm_sq:
                        v = [0] * D
                        for idx, val in zip(support, values):
                            v[idx] = val
                        more_candidates.append(v)
        
        more_candidates = np.array(more_candidates, dtype=np.int64)
        
        # Remove duplicates with existing
        existing_set = set(map(tuple, current))
        new_candidates = []
        for c in more_candidates:
            if tuple(c) not in existing_set:
                new_candidates.append(c)
        
        new_candidates = np.array(new_candidates, dtype=np.int64) if new_candidates else np.array([], dtype=np.int64).reshape(0, D)
        
        # Greedily add compatible vectors
        for c in new_candidates:
            if time.time() > deadline - 2.0:
                break
            c_norm_sq = int(np.dot(c, c))
            new_max = max(current_max_norm_sq, c_norm_sq)
            
            # Check distance to all current vectors
            dots = current @ c
            dist_sq = c_norm_sq + current_norms_sq - 2 * dots
            if np.min(dist_sq) >= new_max:
                current = np.vstack([current, c])
                current_norms_sq = np.append(current_norms_sq, c_norm_sq)
                current_max_norm_sq = new_max
        
        all_vectors = current
    
    return all_vectors

# EVOLVE-BLOCK-END
