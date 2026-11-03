"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist

_T0 = time.time()
_DEADLINE = 115.0  # safety margin below the 120 s wall-clock limit


def _score(pts):
    """Exact radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pdist(pts)
    min_pair = d.min() if d.size > 0 else np.inf
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


def _soft_score(pts, beta=200.0):
    """Smooth surrogate using log-sum-exp for soft-min.
    Returns (score, gradient) where score approximates the min radius
    and gradient is the subgradient of the soft-min."""
    # Pairwise distances
    d = pdist(pts)
    # Build full distance matrix
    dist_matrix = np.zeros((N, N))
    idx = np.triu_indices(N, k=1)
    dist_matrix[idx] = d
    dist_matrix = dist_matrix + dist_matrix.T
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Wall distances: 0.5 - |pts|
    wall = 0.5 - np.abs(pts)  # shape (N, 3)
    
    # Soft-min over all constraints: pair_dist/2 and wall_dist
    # All constraint values: pair distances / 2, and wall distances
    # Number of constraints: N*(N-1)/2 pairs + N*3 walls
    n_pairs = N * (N - 1) // 2
    n_walls = N * 3
    n_constr = n_pairs + n_walls
    
    # Collect all constraint values
    vals = np.empty(n_constr)
    vals[:n_pairs] = d * 0.5
    vals[n_pairs:] = wall.ravel()
    
    # Soft-min via log-sum-exp: softmin(x) = -logsumexp(-beta * x) / beta
    neg_beta_x = -beta * vals
    # Numerical stability
    m = neg_beta_x.max()
    logsumexp_val = m + np.log(np.exp(neg_beta_x - m).sum())
    soft_min_val = -logsumexp_val / beta
    
    # Gradient of soft-min w.r.t. each constraint value
    # d(softmin)/d(vals_i) = -exp(-beta*vals_i) / sum(exp(-beta*vals_j))
    exp_vals = np.exp(neg_beta_x - m)
    weights = exp_vals / exp_vals.sum()  # weights for each constraint
    
    # Now compute gradient w.r.t. pts
    grad = np.zeros_like(pts)
    
    # Pair constraints: constraint i corresponds to pair (a, b)
    # vals[i] = dist(a,b) / 2
    # d(vals[i])/d(pts[a]) = (pts[a] - pts[b]) / dist(a,b)
    # d(vals[i])/d(pts[b]) = (pts[b] - pts[a]) / dist(a,b)
    
    # We need to map flat index to pair indices
    # pdist returns in order: (0,1),(0,2),...,(0,N-1),(1,2),...
    pair_weights = weights[:n_pairs]
    # Build index mapping
    pair_idx_a = np.repeat(np.arange(N), np.arange(N-1, 0, -1))
    pair_idx_b = np.concatenate([np.arange(a+1, N) for a in range(N-1)])
    
    # For each pair, compute contribution
    for i in range(n_pairs):
        a, b = pair_idx_a[i], pair_idx_b[i]
        diff = pts[a] - pts[b]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            # d(vals[i])/d(pts[a]) = diff / dist
            # d(softmin)/d(pts[a]) = weights[i] * d(vals[i])/d(pts[a])
            grad[a] += pair_weights[i] * diff / dist
            grad[b] -= pair_weights[i] * diff / dist
    
    # Wall constraints: constraint j corresponds to point j, axis k
    # vals[n_pairs + j*3 + k] = 0.5 - |pts[j,k]|
    # d(vals)/d(pts[j,k]) = -sign(pts[j,k])
    wall_weights = weights[n_pairs:]
    for j in range(N):
        for k in range(3):
            w = wall_weights[j * 3 + k]
            if abs(pts[j, k]) > 1e-12:
                grad[j, k] += w * (-np.sign(pts[j, k]))
            else:
                grad[j, k] += w * 0.0
    
    # soft_min is a lower bound approximation; we want to maximize it
    # The gradient points in the direction that increases soft_min
    return soft_min_val, grad


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with smaller kicks."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.01 * (T / T0 + 0.05)
        idx = rng.choice(N, size=max(1, N // 5), replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 500 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def _gradient_ascent(pts, iters, step0, step1, rng, beta=200.0):
    """Smooth gradient ascent using soft-min surrogate."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(iters):
        step = step0 * (step1 / step0) ** (i / iters)
        soft_val, g = _soft_score(cur, beta=beta)
        
        # Add small noise to escape flat regions
        g = g + rng.normal(0.0, 0.002, size=cur.shape)
        
        # Normalize gradient to avoid overshooting
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g = g / gn
        
        cand = cur + step * g
        cand_r = _score(cand)
        if cand_r >= cur_r - 1e-9:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 100 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def _polish(pts, max_iters=5000, beta_start=500.0, beta_end=5000.0, rng=None):
    """High-precision polish pass with increasing beta (sharper soft-min)."""
    if rng is None:
        rng = np.random.default_rng(42)
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    
    for i in range(max_iters):
        # Increase beta over time for sharper approximation
        frac = i / max_iters
        beta = beta_start * (beta_end / beta_start) ** frac
        step = 0.001 * (1.0 - 0.9 * frac)  # Decrease step size
        
        soft_val, g = _soft_score(cur, beta=beta)
        
        # Very small noise
        g = g + rng.normal(0.0, 0.0005, size=cur.shape)
        
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g = g / gn
        
        cand = cur + step * g
        cand_r = _score(cand)
        if cand_r >= cur_r - 1e-10:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        
        if i % 200 == 0 and time.time() > _DEADLINE:
            break
    
    return best, best_r


def construct():
    global _T0
    rng = np.random.default_rng(12345)

    # Start from a 4x4x4 grid (the seed's arrangement)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base = grid[:N].copy()
    base_r = _score(base)
    best = base.copy()
    best_r = base_r

    starts = [base]
    
    # 5x5x2 slab arrangement
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    if len(slab) >= N:
        starts.append(slab[:N].copy())
    
    # A few random valid starts (jittered grid) to escape the local optimum.
    for _ in range(3):
        s = base + rng.normal(0.0, 0.02, size=base.shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)

    for s in starts:
        if time.time() > _DEADLINE:
            break
        a, ar = _anneal(s, T0=0.03, T1=0.0005, steps=3000, rng=rng)
        if ar > best_r:
            best_r, best = ar, a
        if time.time() > _DEADLINE:
            break
        g, gr = _gradient_ascent(a, iters=1500, step0=0.008, step1=0.0003, rng=rng, beta=200.0)
        if gr > best_r:
            best_r, best = gr, g

    # Final high-precision polish pass
    if time.time() < _DEADLINE - 10:
        p, pr = _polish(best, max_iters=3000, beta_start=500.0, beta_end=5000.0, rng=rng)
        if pr > best_r:
            best_r, best = pr, p

    # Final safety: ensure all centres are strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)
    
    return best
# EVOLVE-BLOCK-END
