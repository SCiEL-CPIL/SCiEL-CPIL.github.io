"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import cdist

START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s margin for the 150s limit


def _compute_radius(c):
    """Compute the maximum radius for given centers (double precision)."""
    c = np.asarray(c)
    if not np.all(np.isfinite(c)):
        return 0.0
    boundary_dist = 0.5 - np.abs(c).max(axis=1)
    if boundary_dist.min() < 0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    return min(boundary_dist.min(), min_pair_dist / 2.0)


def _softmin_neg(c, beta=1000.0):
    """Smooth surrogate: negative soft-min of the radius constraints."""
    c = c.reshape(-1, 2)
    boundary = 0.5 - np.abs(c).max(axis=1)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    all_vals = np.concatenate([boundary, np.array([min_pair / 2.0])])
    max_val = all_vals.max()
    soft_min = max_val - np.log(np.sum(np.exp(-beta * (all_vals - max_val)))) / beta
    return -soft_min


def _initial_square_grid(n):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _hex_lattice(half_range=3.0, step=0.001):
    """Fine hexagonal lattice of candidate sites inside the square."""
    pts = []
    h = step * np.sqrt(3) / 2.0
    i = 0
    y = -half_range
    while y <= half_range:
        xoff = (step / 2.0) if (i % 2) else 0.0
        x = -half_range + xoff
        while x <= half_range:
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append((x, y))
            x += step
        y += h
        i += 1
    return np.array(pts)


def _greedy_select_fine(lattice, n, time_budget=30.0):
    """Improved greedy: use a running min-distance vector and boundary constraint."""
    pts = lattice
    start = time.time()
    
    # Precompute boundary distances
    b = 0.5 - np.abs(pts).max(axis=1)
    
    # First point: center of square (max boundary dist)
    first_idx = np.argmax(b)
    chosen = [first_idx]
    min_d = pts.copy()  # min distance to chosen set, initialized to self-distance (inf effectively)
    # Actually, initialize min_d to large values, then update
    min_d = np.full(len(pts), np.inf)
    min_d[first_idx] = 0.0
    
    for _ in range(n - 1):
        if time.time() - start > time_budget:
            break
        # Score: min of (min distance to chosen, boundary distance * 2)
        # We want to maximize the minimum radius, so we want to maximize min(min_d, b*2)
        # But we need to exclude already-chosen points
        score = np.minimum(min_d, b * 2.0)
        for idx in chosen:
            score[idx] = -1.0
        
        # Find best candidate
        best_idx = np.argmax(score)
        chosen.append(best_idx)
        
        # Update min_d: for each point, min distance to new chosen point
        d_to_new = np.linalg.norm(pts - pts[best_idx], axis=1)
        min_d = np.minimum(min_d, d_to_new)
    
    return pts[np.array(chosen[:n])]


def _greedy_select(lattice, n):
    """Greedy farthest-point selection: maximize min distance to chosen + boundary."""
    pts = lattice
    chosen = []
    # First point: maximize distance to boundary (centre of square)
    b = 0.5 - np.abs(pts).max(axis=1)
    chosen.append(pts[np.argmax(b)])
    for _ in range(n - 1):
        if time.time() > TIME_LIMIT - 10:
            break
        ch = np.array(chosen)
        d = cdist(pts, ch)
        min_d = d.min(axis=1)
        b = 0.5 - np.abs(pts).max(axis=1)
        score = np.minimum(min_d, b * 2.0)  # boundary dist as half-distance equivalent
        # exclude already chosen
        for idx in range(len(chosen)):
            score[np.where((pts == chosen[idx]).all(axis=1))[0]] = -1.0
        chosen.append(pts[np.argmax(score)])
    return np.array(chosen)


def _refine(c, time_budget=30.0):
    """Refine positions using L-BFGS-B with a smooth soft-min objective."""
    from scipy.optimize import minimize
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    beta = 300.0
    for _ in range(6):
        if time.time() - start > time_budget:
            break

        def obj(x):
            return _softmin_neg(x, beta)

        try:
            res = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            new_c = res.x.reshape(-1, 2)
            new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c
        except Exception:
            pass
        beta *= 1.8
    return best_c, best_r


def _refine_local(c, time_budget=20.0):
    """High-precision local polish via small random perturbations (steepest ascent)."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    step_size = 0.0005
    for _ in range(4000):
        if time.time() - start > time_budget:
            break
        c_trial = best_c.copy()
        n_move = min(4, N)
        idx = np.random.choice(N, n_move, replace=False)
        c_trial[idx] += np.random.normal(0, step_size, size=(n_move, 2))
        c_trial = np.clip(c_trial, -0.5 + 1e-12, 0.5 - 1e-12)
        r_trial = _compute_radius(c_trial)
        if r_trial > best_r:
            best_r = r_trial
            best_c = c_trial
            step_size *= 1.1
        else:
            step_size *= 0.85
            step_size = max(step_size, 1e-7)
    return best_c, best_r


def _discrete_local_refine(c, lattice, time_budget=15.0):
    """Try swapping each circle to a nearby lattice site to improve radius."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    
    # Build a KD-tree or use cdist for nearest lattice neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(lattice)
    
    # For each circle, find nearest few lattice sites
    # Try moving one circle at a time to a better lattice position
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start > time_budget:
                break
            # Find 5 nearest lattice sites to current position
            dists, idxs = tree.query(best_c[i], k=5)
            for j in idxs:
                c_trial = best_c.copy()
                c_trial[i] = lattice[j]
                c_trial = np.clip(c_trial, -0.5 + 1e-12, 0.5 - 1e-12)
                r_trial = _compute_radius(c_trial)
                if r_trial > best_r + 1e-12:
                    best_r = r_trial
                    best_c = c_trial
                    improved = True
                    break
    return best_c, best_r


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing minimum radius."""
    global START_TIME, TIME_LIMIT
    np.random.seed(7)
    candidates = []

    # Square grid
    candidates.append(_initial_square_grid(N))

    # Greedy discrete lattice selection (discrete optimization, avoids local-minima traps)
    try:
        lattice = _hex_lattice(half_range=0.5, step=0.001)
        candidates.append(_greedy_select_fine(lattice, N, time_budget=25.0))
    except Exception:
        pass
    
    # Also try a coarser lattice with different step for diversity
    try:
        lattice2 = _hex_lattice(half_range=0.5, step=0.002)
        candidates.append(_greedy_select_fine(lattice2, N, time_budget=10.0))
    except Exception:
        pass

    # Perturbed square grid
    c_pert = _initial_square_grid(N) + np.random.normal(0, 0.01, size=(N, 2))
    candidates.append(np.clip(c_pert, -0.49, 0.49))

    # Evaluate initial candidates
    best_c, best_r = None, 0.0
    for c in candidates:
        if len(c) != N:
            continue
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
    if best_c is None:
        best_c = _initial_square_grid(N)
        best_r = _compute_radius(best_c)

    # Refine each candidate with a time budget
    time_remaining = TIME_LIMIT - (time.time() - START_TIME) - 15
    if time_remaining > 8 and len(candidates) > 0:
        tpc = time_remaining / len(candidates)
        for c in candidates:
            if len(c) != N:
                continue
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            c_ref, r_ref = _refine(c, time_budget=tpc)
            if r_ref > best_r:
                best_r = r_ref
                best_c = c_ref

    # Final high-precision polish
    time_remaining = TIME_LIMIT - (time.time() - START_TIME) - 5
    if time_remaining > 3:
        # Try discrete local refinement first (fast, can jump to better lattice sites)
        try:
            lattice_fine = _hex_lattice(half_range=0.5, step=0.001)
            c_disc, r_disc = _discrete_local_refine(best_c, lattice_fine, time_budget=min(10, time_remaining * 0.4))
            if r_disc > best_r:
                best_r = r_disc
                best_c = c_disc
        except Exception:
            pass
        
        # Then continuous polish
        time_remaining2 = TIME_LIMIT - (time.time() - START_TIME) - 3
        if time_remaining2 > 2:
            c_final, r_final = _refine_local(best_c, time_budget=time_remaining2)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final

    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    if not np.all(np.isfinite(best_c)):
        return _initial_square_grid(N)
    return best_c
# EVOLVE-BLOCK-END
