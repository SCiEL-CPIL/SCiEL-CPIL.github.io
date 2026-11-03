"""Sphere packing in a unit cube: 100 equal spheres, maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _flatten(x):
    return x.reshape(-1)

def _unflatten(v):
    return v.reshape(-1, 3)

def _score(c):
    """Compute the achievable radius for center array c of shape (N,3)."""
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to walls
    min_wall = np.min(0.5 - np.abs(c))
    return min(min_pair, min_wall)

def _energy_and_grad(v):
    """
    Smooth surrogate energy: soft-min of inverse pair distances + wall penalty.
    We minimize this to maximize the minimum distance.
    """
    c = v.reshape(-1, 3)
    n = c.shape[0]
    
    # Pairwise distances (avoid zero on diagonal)
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2)) + 1e-12
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min via log-sum-exp of -log(dist) ~ -logsumexp(-log(dist))
    # Equivalent: softmin of dist = logsumexp(-log(dist)) with large beta
    # Use: E_pair = (1/beta) * log( sum exp(-beta * dist) )  -> approximates min(dist)
    beta = 50.0
    # For numerical stability, subtract max
    neg_beta_dist = -beta * dist
    # Only consider upper triangle to avoid double counting, but for gradient simplicity use all
    # Actually let's use all off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    vals = neg_beta_dist[mask]
    max_val = np.max(vals)
    exp_vals = np.exp(vals - max_val)
    logsumexp = max_val + np.log(np.sum(exp_vals))
    E_pair = logsumexp / beta  # This approximates -min(dist) ... wait, let me reconsider.
    
    # Actually, let's use a simpler smooth objective:
    # Minimize sum of (1/dist^p) for all pairs + wall penalty
    # This is differentiable and pushes spheres apart.
    p = 4.0
    inv_dist_p = np.power(dist, -p)
    np.fill_diagonal(inv_dist_p, 0.0)
    E_pair = np.sum(inv_dist_p)
    
    # Wall penalty: large quadratic penalty when approaching boundary
    wall_dist = 0.5 - np.abs(c)
    # Penalty grows rapidly as wall_dist -> 0
    E_wall = np.sum(np.power(np.maximum(0.0, 0.1 - wall_dist), 2)) * 1000.0
    
    energy = E_pair + E_wall
    
    # Gradient
    # dE_pair/dv: for pair (i,j), d(1/dist^p)/dc_i = p * (c_i - c_j) / dist^(p+2)
    # Sum over all j != i
    grad = np.zeros_like(c)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = c[i] - c[j]
            dist_ij = dist[i, j]
            grad[i] += p * d / (dist_ij ** (p + 2))
    
    # Wall gradient
    # dE_wall/dv = -2000 * max(0, 0.1 - wall_dist) * sign(c)  for each component
    for i in range(n):
        for k in range(3):
            wd = 0.5 - abs(c[i, k])
            if wd < 0.1:
                grad[i, k] += -2000.0 * (0.1 - wd) * np.sign(c[i, k])
    
    return energy, grad.flatten()

def _optimize_from_start(v0, time_budget):
    """Run L-BFGS-B from initial position v0."""
    try:
        res = minimize(
            _energy_and_grad,
            v0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 5000,
            }
        )
        return res.x
    except Exception:
        return v0

def _make_grid_init(m):
    """Create a jittered cubic grid initialization."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _make_random_init(seed):
    """Random initialization inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _make_fibonacci_init():
    """Fibonacci sphere-like distribution projected into cube."""
    n = N
    golden = (1 + np.sqrt(5)) / 2
    y = 1 - 2 * (np.arange(n) + 0.5) / n
    radius = np.sqrt(np.maximum(0, 1 - y * y))
    theta = np.arange(n) * 2 * np.pi / golden
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    c = np.column_stack([x, y, z]) * 0.45
    return c

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Jittered grid (m=5 gives 125 points, take 100)
    rng0 = np.random.RandomState(42)
    grid = _make_grid_init(5)
    grid_jitter = grid + rng0.normal(0, 0.01, size=(N, 3))
    inits.append(grid_jitter)
    
    # 2. Pure random
    for seed in [0, 1, 2, 3, 4]:
        inits.append(_make_random_init(seed))
    
    # 3. Fibonacci
    inits.append(_make_fibonacci_init())
    
    # 4. More grid variations
    for m in [4, 6]:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(grid) >= N:
            inits.append(grid[:N] + np.random.RandomState(m).normal(0, 0.005, size=(N, 3)))
    
    best_v = None
    best_score = -1.0
    
    for idx, c0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        # Clip to valid range
        c0 = np.clip(c0, -0.49, 0.49)
        v0 = c0.flatten()
        
        # Optimize
        v_opt = _optimize_from_start(v0, DEADLINE - elapsed)
        
        c_opt = v_opt.reshape(-1, 3)
        # Clip to ensure validity
        c_opt = np.clip(c_opt, -0.4999, 0.4999)
        
        s = _score(c_opt)
        if s > best_score:
            best_score = s
            best_v = c_opt.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
    
    # Final polish: run a tighter optimization from the best candidate
    if best_v is not None and time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(
                _energy_and_grad,
                best_v.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 10000,
                }
            )
            c_polished = res.x.reshape(-1, 3)
            c_polished = np.clip(c_polished, -0.4999, 0.4999)
            s_polished = _score(c_polished)
            if s_polished > best_score:
                best_v = c_polished
                best_score = s_polished
        except Exception:
            pass
    
    # Final validity check
    if best_v is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Ensure all values are finite and within bounds
    if not np.all(np.isfinite(best_v)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_v

# EVOLVE-BLOCK-END
