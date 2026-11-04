"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np

# Build the E8 kissing configuration (240 vectors of norm sqrt(2) in R8)
def _build_e8():
    vecs = []
    # 112 vectors of type (+-1, +-1, 0,...,0)
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0]*8
                    v[i] = float(si)
                    v[j] = float(sj)
                    vecs.append(v)
    # 128 vectors of type (+-1/2, ..., +-1/2) with even number of minus signs
    from itertools import product
    for signs in product((1, -1), repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            vecs.append([s*0.5 for s in signs])
    return np.array(vecs)  # shape (240, 8), norm sqrt(2)

# Build D3 (A3) kissing configuration in R3 (6 vectors of norm sqrt(2))
def _build_d3():
    vecs = []
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        for s0 in (1,-1):
            for s1 in (1,-1):
                v = [0.0, 0.0, 0.0]
                v[perm[0]] = float(s0)
                v[perm[1]] = float(s1)
                # we want exactly 2 nonzero entries; but D3 minimal vectors are (±1,±1,0) perms
                # that gives 3*2*2 = 12? Actually D3 = A3 has 6 minimal vectors of norm sqrt(2)
                # They are: (1,1,0), (1,0,1), (0,1,1) and their negatives -> 6
                # So we should only take positive-positive pairs for each unordered pair
                if s0 == 1 and s1 == 1:
                    vecs.append(v)
    # The above gives 6 vectors
    return np.array(vecs)  # shape (6, 3), norm sqrt(2)

def construct():
    t0 = time.time()
    e8 = _build_e8()   # (240, 8), each norm sqrt(2)
    d3 = _build_d3()   # (6, 3), each norm sqrt(2)

    # Scale to integer: multiply by 2
    e8_int = (2*e8).astype(np.int64)  # (240, 8), norm 2
    d3_int = (2*d3).astype(np.int64)  # (6, 3), norm 2

    # Embed into R11
    # E8 part: pad 3 zeros on the right
    e8_pad = np.zeros((240, 11), dtype=np.int64)
    e8_pad[:, :8] = e8_int

    # D3 part: pad 8 zeros on the left
    d3_pad = np.zeros((6, 11), dtype=np.int64)
    d3_pad[:, 8:11] = d3_int

    # Combine
    all_vecs = np.vstack([e8_pad, d3_pad])  # (246, 11)

    # Verify minimum pairwise distance >= max norm (which is 2 for all)
    # Norms: all should be 2
    norms = np.linalg.norm(all_vecs, axis=1)
    max_norm = norms.max()

    # Check distances
    # For a safe check within time: compute pairwise squared distances
    # 246 vectors -> 246*245/2 ~ 30k pairs, fine
    sq_norms = np.sum(all_vecs**2, axis=1)
    # squared distance = |a|^2 + |b|^2 - 2 a.b
    dots = all_vecs @ all_vecs.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2*dots
    np.fill_diagonal(sq_dists, np.inf)
    min_sq_dist = sq_dists.min()

    # Requirement: min distance >= max norm => min_sq_dist >= max_norm^2
    if min_sq_dist + 1e-9 < max_norm*max_norm:
        # Fallback to seed construction (220 vectors)
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return all_vecs
# EVOLVE-BLOCK-END
