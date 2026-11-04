import numpy as np
import time

N = 11
S3 = np.sqrt(3.0)

# 6 edge normals of a unit hexagon at rotation 0
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

UNIT_VERT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def rot2(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def hex_vertices(center, ang_deg):
    R = rot2(ang_deg)
    return center + (UNIT_VERT @ R.T)

def sep_axes(i_ang, j_ang):
    """All candidate separating axes for two hexagons at given rotations."""
    a1 = NORM @ rot2(i_ang).T
    a2 = NORM @ rot2(j_ang).T
    return np.vstack([a1, a2])

def min_separation(Ci, Ai, Cj, Aj):
    """Max over axes of |d · axis| — the separating distance."""
    d = Ci - Cj
    axes = sep_axes(Ai, Aj)
    return float(np.abs(d @ axes.T).max())

def cost_fast(C, A):
    """
    Computes the minimum side length of a regular hexagon (centered at origin, rot 0)
    that contains all inner hexagons.
    This is equivalent to the true minimum side length because we can WLOG fix the outer rotation to 0
    by rotating the entire configuration.
    """
    # Project all vertices of all hexagons onto the 6 normals
    # N x 2 centers, N angles
    # For each hexagon i, get vertices (6x2)
    # We can vectorize this slightly by computing all vertices at once?
    # No, angles are different.
    
    max_proj = 0.0
    for i in range(N):
        verts = hex_vertices(C[i], A[i])
        proj = NORM @ verts.T # 6x6
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
    # Side length S relates to apothem a by a = S * sqrt(3) / 2 => S = 2 * a / sqrt(3)
    # The max projection is the apothem of the bounding hexagon.
    return 2.0 * max_proj / S3

def nonoverlap(C, A, eps=1e-9):
    for i in range(N):
        for j in range(i + 1, N):
            if min_separation(C[i], A[i], C[j], A[j]) < 2.0 - eps:
                return False
    return True

def hex_close_pack_seed():
    """Dense hexagonal packing of 11 unit hexagons (center + ring + partial)."""
    C = np.array([[0.0, 0.0]])
    # inner ring of 6 at distance S3
    for k in range(6):
        a = k * np.pi / 3
        C = np.vstack([C, [S3 * np.cos(a), S3 * np.sin(a)]])
    # outer partial: 4 more
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C = np.vstack([C, [2 * S3 * np.cos(a), 2 * S3 * np.sin(a)]])
    C = C[:N]
    A = np.zeros(N)
    return C, A

def push_apart(C, A, rng):
    """Find most-overlapping pair and push them apart."""
    worst_i, worst_j = -1, -1
    worst_def = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s = min_separation(C[i], A[i], C[j], A[j])
            def_ = 2.0 - s
            if def_ > worst_def:
                worst_def = def_
                worst_i, worst_j = i, j
    if worst_i < 0:
        return C, A
    d = C[worst_i] - C[worst_j]
    n = np.linalg.norm(d)
    if n < 1e-9:
        d = np.array([1.0, 0.0])
    else:
        d = d / n
    push = min(0.3, worst_def * 0.6 + 0.05)
    C = C.copy()
    C[worst_i] += d * push
    C[worst_j] -= d * push
    return C, A

def local_search(C, A, deadline, rng, max_iter=50000):
    """Fast local search using cost_fast."""
    C = C.copy()
    A = A.copy()
    if not nonoverlap(C, A):
        # Try to fix overlaps first
        for _ in range(1000):
            if time.time() > deadline:
                break
            C, A = push_apart(C, A, rng)
            if nonoverlap(C, A):
                break
    
    S = cost_fast(C, A)
    best_S = S if nonoverlap(C, A) else np.inf
    best_C, best_A = C.copy(), A.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random move
        idx = rng.randint(0, N - 1)
        new_C = C.copy()
        new_A = A.copy()
        if rng.random() < 0.7:
            new_C[idx] += rng.normal(0, 0.1, 2)
        else:
            new_A[idx] += rng.normal(0, 3.0)
            
        # Check non-overlap
        if not nonoverlap(new_C, new_A):
            continue
            
        new_S = cost_fast(new_C, new_A)
        if new_S < S:
            C, A, S = new_C, new_A, new_S
            if S < best_S:
                best_S = S
                best_C, best_A = C.copy(), A.copy()
                
    return best_C, best_A, best_S

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin

    rng = np.random.RandomState(42)
    
    # Initial seed
    C0, A0 = hex_close_pack_seed()
    
    # If seed is invalid, push apart
    if not nonoverlap(C0, A0):
        for _ in range(1000):
            if time.time() > deadline:
                break
            C0, A0 = push_apart(C0, A0, rng)
            if nonoverlap(C0, A0):
                break

    best_C, best_A, best_S = C0.copy(), A0.copy(), cost_fast(C0, A0)

    # Run multiple local searches from different perturbations
    n_rounds = 10
    for r in range(n_rounds):
        if time.time() > deadline - 5:
            break
            
        # Perturb the best solution
        C_start = best_C + rng.normal(0, 0.2, (N, 2))
        A_start = best_A + rng.normal(0, 5.0, N)
        
        # Ensure validity at start (quick push apart)
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            C_start, A_start = push_apart(C_start, A_start, rng)
            if nonoverlap(C_start, A_start):
                break
        
        if not nonoverlap(C_start, A_start):
            continue

        # Run local search
        C_res, A_res, S_res = local_search(C_start, A_start, deadline - 5, rng, max_iter=20000)
        
        if S_res < best_S:
            best_S = S_res
            best_C = C_res.copy()
            best_A = A_res.copy()

    # Final polish on the best solution
    if time.time() < deadline - 2:
        C_res, A_res, S_res = local_search(best_C, best_A, deadline - 2, rng, max_iter=50000)
        if S_res < best_S:
            best_S = S_res
            best_C = C_res.copy()
            best_A = A_res.copy()

    inner = [[float(best_C[i, 0]), float(best_C[i, 1]), float(best_A[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0
