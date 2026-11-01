"""
Strategy: Multi-start BFGS on a smooth soft-min objective with structured lattice initializations.
Instead of stochastic MD/SA, we use deterministic gradient-based optimization (L-BFGS-B from scipy)
on a smooth approximation of the max-min packing objective. We initialize from FCC/BCC lattice
points with small random perturbations and run multiple independent BFGS optimizations to
escape local minima. The soft-min uses a polynomial (power) approximation that is smooth
everywhere and closely tracks the true minimum distance.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100


def _soft_min_value(distances, power=20):
    """
    Smooth approximation of min(distances) using negative power mean.
    As power -> inf, this approaches the true min.
    Returns a smooth scalar.
    """
    # Use log-sum-exp trick for numerical stability
    # soft_min = (sum(d_i^(-p)))^(-1/p)
    # log(soft_min) = -(1/p) * log(sum(d_i^(-p)))
    # For stability: let m = min(d_i), then d_i^(-p) = exp(-p * log(d_i))
    # log(sum(d_i^(-p))) = -p*m + log(sum(exp(-p*(log(d_i)-m))))  -- wait, that's not right
    # Actually: d_i^(-p) = exp(-p * ln(d_i))
    # Let a_i = -p * ln(d_i). Then sum(exp(a_i)).
    # For stability: a_max = max(a_i) = -p * min(ln(d_i)) = -p * ln(max(d_i))
    # Hmm, let me just do it directly with clipping.
    
    d = np.maximum(distances, 1e-15)
    # negative power mean of order p: (sum d_i^(-p))^(1/(-p)) = (sum d_i^(-p))^(-1/p)
    # This is smooth and <= min(d_i)
    neg_p = -power
    # To avoid overflow, use log domain
    log_d = np.log(d)
    a = neg_p * log_d  # = -p * ln(d_i), these are negative large for small d_i
    a_max = np.max(a)
    log_sum = a_max + np.log(np.sum(np.exp(a - a_max)))
    # soft_min = exp(log_sum / neg_p) = exp(log_sum / (-p))
    soft_min_val = np.exp(log_sum / neg_p)
    return soft_min_val


def _compute_radius(pos):
    """Compute the actual packing radius (score)."""
    if pos.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    
    # Pairwise distances
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(d**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pos)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_objective(x, power=20):
    """
    Objective to MINIMIZE: -soft_min(all relevant distances).
    x is flat array of shape (N*3,).
    """
    pos = x.reshape(N, 3)
    
    # Pairwise distances (upper triangle only to save time, but for gradient we need all)
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(d**2, axis=2)
    np.fill_diagonal(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pos)
    
    # Combine all distances into one vector
    # We only need off-diagonal pairs (i<j) and wall distances
    # For the soft-min, we can just use all off-diagonal + walls
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    all_dists = np.concatenate([pair_dists, wall_dists.ravel()])
    
    sm = _soft_min_value(all_dists, power)
    return -sm


def _soft_min_gradient(x, power=20):
    """
    Analytical gradient of -soft_min(all distances).
    """
    pos = x.reshape(N, 3)
    
    # Pairwise distances
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(d**2, axis=2)
    np.fill_diagonal(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pos)  # shape (N, 3)
    
    # Get upper triangle pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]  # shape (M,)
    
    all_dists = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # soft_min = (sum d_i^(-p))^(-1/p)
    # d(soft_min)/d(d_k) = (1/p) * (sum d_i^(-p))^(-1/p - 1) * (-p) * d_k^(-p-1)
    #                     = -(1/p) * S^(-1/p - 1) * d_k^(-p-1)  where S = sum d_i^(-p)
    # Wait: soft_min = S^(-1/p) where S = sum d_i^(-p)
    # d(soft_min)/dS = (-1/p) * S^(-1/p - 1)
    # dS/d(d_k) = -p * d_k^(-p-1)
    # d(soft_min)/d(d_k) = (-1/p) * S^(-1/p-1) * (-p) * d_k^(-p-1)
    #                     = S^(-1/p-1) * d_k^(-p-1)
    
    p = power
    d_safe = np.maximum(all_dists, 1e-15)
    neg_p = -p
    log_d = np.log(d_safe)
    a = neg_p * log_d
    a_max = np.max(a)
    log_sum = a_max + np.log(np.sum(np.exp(a - a_max)))
    S = np.exp(log_sum)
    
    # weights: w_k = d_k^(-p-1) / S^(1/p + 1)  ... let me redo
    # soft_min = S^(-1/p)
    # d(soft_min)/d(d_k) = S^(-1/p - 1) * d_k^(-p-1)
    # Let's compute this directly
    d_k_neg_p1 = d_safe ** (-(p + 1))
    S_neg_1p_1 = S ** (-1.0/p - 1.0)
    weights = S_neg_1p_1 * d_k_neg_p1  # shape (M + N*3,)
    
    # Now gradient of -soft_min w.r.t. x
    # d(-soft_min)/d(x) = -d(soft_min)/d(x)
    # d(soft_min)/d(x_j) = sum_k weights[k] * d(d_k)/d(x_j)
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    # For pair (i, j) with i < j, distance d_ij = |pos_i - pos_j|
    # d(d_ij)/d(pos_i) = (pos_i - pos_j) / d_ij
    # d(d_ij)/d(pos_j) = (pos_j - pos_i) / d_ij
    n_pairs = len(iu)
    pair_weights = weights[:n_pairs]
    
    # Build gradient contribution from pairs
    # For each pair (i,j), contribution to grad[i] += pair_weights[k] * (pos[i]-pos[j])/d_ij
    # contribution to grad[j] += pair_weights[k] * (pos[j]-pos[i])/d_ij
    
    # Vectorized: 
    # d_vectors[i,j] = pos[i] - pos[j], shape (N, N, 3)
    # For upper triangle pair k corresponding to (i, j):
    # grad[i] += w_k * d_vec[i,j] / d_ij
    # grad[j] += w_k * d_vec[j,i] / d_ij = -w_k * d_vec[i,j] / d_ij
    
    # Let's accumulate
    ii, jj = iu
    for k in range(n_pairs):
        i, j = ii[k], jj[k]
        diff = pos[i] - pos[j]
        d_val = pair_dists[k]
        if d_val > 1e-15:
            unit = diff / d_val
            grad[3*i:3*i+3] += pair_weights[k] * unit
            grad[3*j:3*j+3] -= pair_weights[k] * unit
    
    # Wall part
    # wall_dist for particle i, coordinate c: 0.5 - |pos[i,c]|
    # d(wall_dist)/d(pos[i,c]) = -sign(pos[i,c])
    # d(soft_min)/d(pos[i,c]) = sum over wall terms for (i,c): weight * (-sign(pos[i,c]))
    
    wall_weights = weights[n_pairs:]  # shape (N*3,)
    wall_w = wall_weights.reshape(N, 3)
    signs = np.sign(pos)
    # d(soft_min)/d(pos) contribution from walls:
    # For each (i,c): wall_w[i,c] * (-sign(pos[i,c]))
    wall_grad = -wall_w * signs
    grad += wall_grad.ravel()
    
    return -grad  # gradient of -soft_min


def _init_fcc():
    """
    Generate FCC lattice points in [-0.5, 0.5]^3.
    FCC: simple cubic + face centers.
    For a cube of side 1, we can fit multiple FCC unit cells.
    """
    # FCC unit cell has 4 atoms. In a cube of side 1, if we use n cells per side,
    # we get n^3 * 4 points. For N=100, we want ~100 points.
    # n=3: 27*4 = 108 points. Close enough.
    n = 3
    a = 1.0 / n  # cell size
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Corners of FCC cell (but we only add one corner to avoid double counting)
                # Actually, standard FCC: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in cell coords
                base = np.array([i, j, k]) * a
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0]),
                ]
                for off in offsets:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= 0.5  # center in [-0.5, 0.5]
    
    # Select N points (randomly if more than N)
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # Pad with random points
        extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 3))
        pts = np.vstack([pts, extra])
    
    return pts


def _init_bcc():
    """
    Generate BCC lattice points.
    BCC: simple cubic + body center. 2 atoms per cell.
    n=4: 64*2 = 128 points.
    """
    n = 4
    a = 1.0 / n
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                base = np.array([i, j, k]) * a
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0.5*a, 0.5*a, 0.5*a]),
                ]
                for off in offsets:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= 0.5
    
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 3))
        pts = np.vstack([pts, extra])
    
    return pts


def _init_grid():
    """
    Simple cubic grid initialization.
    """
    m = int(np.ceil(N ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        return grid[:N].copy()
    else:
        extra = np.random.uniform(-0.4, 0.4, (N - len(grid), 3))
        return np.vstack([grid, extra])


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_pos = None
    best_score = -1.0
    
    # Generate diverse initializations
    np.random.seed(42)
    inits = []
    
    # FCC with various jitter levels
    for jitter in [0.01, 0.02, 0.03, 0.05]:
        base = _init_fcc()
        jittered = base + np.random.normal(0, jitter, base.shape)
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # BCC with various jitter levels
    for jitter in [0.01, 0.02, 0.03, 0.05]:
        base = _init_bcc()
        jittered = base + np.random.normal(0, jitter, base.shape)
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # Grid with jitter
    for jitter in [0.02, 0.04]:
        base = _init_grid()
        jittered = base + np.random.normal(0, jitter, base.shape)
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # Pure random
    inits.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # Random with FCC-like structure (shuffle FCC points)
    base = _init_fcc()
    shuffled = base[np.random.permutation(len(base))]
    if len(shuffled) > N:
        shuffled = shuffled[:N]
    inits.append(shuffled + np.random.normal(0, 0.02, shuffled.shape))
    
    power = 30  # Higher power = closer to true min, but less smooth
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        x0 = init_pos.ravel()
        
        # Evaluate initial score
        init_score = _compute_radius(init_pos)
        if init_score > best_score:
            best_score = init_score
            best_pos = init_pos.copy()
        
        # Run BFGS optimization
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                jac=_soft_min_gradient,
                args=(power,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 2000,
                    'disp': False,
                }
            )
            
            if result.success or result.fun < 0:
                opt_pos = result.x.reshape(N, 3)
                opt_score = _compute_radius(opt_pos)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pos = opt_pos.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final: try a few more random restarts if time permits
    remaining = deadline - time.time()
    if remaining > 8.0:
        for trial in range(5):
            if time.time() > deadline - 3.0:
                break
            x0 = np.random.uniform(-0.45, 0.45, (N, 3)).ravel()
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    jac=_soft_min_gradient,
                    args=(power,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': 1000,
                        'disp': False,
                    }
                )
                if result.fun < 0:
                    opt_pos = result.x.reshape(N, 3)
                    opt_score = _compute_radius(opt_pos)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_pos = opt_pos.copy()
            except Exception:
                pass
    
    if best_pos is None:
        return _init_grid()
    
    return best_pos
