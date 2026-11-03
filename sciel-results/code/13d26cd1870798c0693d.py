"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the exact packing radius."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Boundary distances
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    r = min(min_pair / 2.0, min_boundary)
    return max(r, 0.0)

def _softmin_objective(params, t=0.01):
    """Negative soft-min of all constraints (to maximize min)."""
    centers = params.reshape(N, 3)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # Use squared distances for efficiency, but we need actual distances for softmin
    # Actually let's use a different approach: softmin of (dists/2) and (boundary)
    # For efficiency, use a log-sum-exp approximation
    # Softmin of values v_i: -t*log(sum(exp(-v_i/t)))
    # We want to maximize min(all constraints), so minimize -softmin
    
    # Boundary constraints
    boundary = 0.5 - np.abs(centers)
    flat_boundary = boundary.ravel()
    
    # Pairwise: use dists/2
    dists = np.sqrt(np.maximum(dists_sq, 0))
    np.fill_diagonal(dists, np.inf)
    upper_tri = np.triu_indices(N, k=1)
    pair_dists = dists[upper_tri] / 2.0
    
    all_vals = np.concatenate([pair_dists, flat_boundary])
    
    # Softmin
    # To avoid overflow, subtract max
    m = np.min(all_vals)
    if m < -1000:
        return 1000.0
    exp_vals = np.exp(-(all_vals - m) / t)
    log_sum = np.log(np.sum(exp_vals))
    softmin_val = m - t * log_sum
    return -softmin_val  # minimize negative softmin

def _softmin_grad(params, t=0.01):
    """Gradient of the softmin objective."""
    centers = params.reshape(N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    np.fill_diagonal(dists, np.inf)
    
    boundary = 0.5 - np.abs(centers)
    flat_boundary = boundary.ravel()
    
    upper_tri = np.triu_indices(N, k=1)
    pair_dists = dists[upper_tri] / 2.0
    
    all_vals = np.concatenate([pair_dists, flat_boundary])
    
    m = np.min(all_vals)
    if m < -1000:
        return np.zeros_like(params)
    
    exp_vals = np.exp(-(all_vals - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    grad = np.zeros_like(params)
    
    # Gradient of -softmin = -d/dparams [m - t*log(sum(exp(-(vi-m)/t)))]
    # = -sum_j w_j * (-d v_j/dparams) = sum_j w_j * d v_j/dparams
    # Wait, let me be more careful.
    # softmin = m - t*log(sum(exp(-(vi-m)/t)))
    # d(softmin)/dparams = -t * d/dparams [log(sum(exp(-(vi-m)/t)))]
    # = -t * [sum_j exp(-(vj-m)/t) * (-1/t) * dvj/dparams] / sum(exp)
    # = sum_j w_j * dvj/dparams
    # So d(-softmin)/dparams = -sum_j w_j * dvj/dparams
    
    # Pairwise contributions
    for idx in range(len(upper_tri[0])):
        i, j = upper_tri[0][idx], upper_tri[1][idx]
        w = weights[idx]
        if w < 1e-15:
            continue
        d_ij = centers[i] - centers[j]
        dist = dists[i, j]
        if dist < 1e-15:
            continue
        # d(v_j)/d(centers[i]) where v_j = dist(i,j)/2
        grad[i] += w * d_ij / (2 * dist) * 2  # wait, v = dist/2, dv/dci = d_i/dci / dist * 1/2 * 2? 
        # Actually: v = |ci - cj| / 2, dv/dci = (ci-cj)/|ci-cj| * 1/2
        # d(-softmin)/dci += -w * dv/dci = -w * (ci-cj)/(2*dist)
        # But we want to minimize -softmin, so grad = d(-softmin)/dparams
        # Let me redo: objective = -softmin
        # d(objective)/dci = -d(softmin)/dci = -sum_k w_k * dv_k/dci
        # For pair (i,j): dv/dci = (ci-cj)/(2*dist)
        grad[i] -= w * d_ij / (2 * dist)
        grad[j] += w * d_ij / (2 * dist)
    
    # Boundary contributions
    for idx in range(N * 3):
        w = weights[len(pair_dists) + idx]
        if w < 1e-15:
            continue
        i = idx // 3
        k = idx % 3
        # boundary[i,k] = 0.5 - |centers[i,k]|
        # d(boundary)/d(centers[i,k]) = -sign(centers[i,k])
        sign_val = np.sign(centers[i, k])
        if sign_val == 0:
            sign_val = 1.0
        grad[i, k] += w * sign_val  # d(-softmin)/dparam = -w * d(boundary)/dparam = -w * (-sign) = w*sign
    
    return grad

def _local_optimize(centers, t=0.01, max_iter=500):
    """Run local optimization on centers."""
    params = centers.ravel()
    try:
        result = minimize(
            _softmin_objective,
            params,
            args=(t,),
            jac=_softmin_grad,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x.reshape(N, 3)
    except:
        return centers

def _bcc_lattice(n):
    """Generate BCC lattice points, take first n."""
    # BCC: simple cubic + body center
    m = int(np.ceil((n / 2) ** (1.0 / 3.0) - 1e-9))
    m = max(m, 1)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
                points.append([x + 0.5/m, y + 0.5/m, z + 0.5/m])
    points = np.array(points)
    # Scale to fit in cube
    # Find the extent
    min_c = points.min(axis=0)
    max_c = points.max(axis=0)
    extent = max_c - min_c
    scale = 1.0 / np.max(extent)
    points = (points - min_c) * scale - 0.5
    return points[:n]

def _fcc_lattice(n):
    """Generate FCC lattice points, take first n."""
    # FCC: simple cubic + face centers
    m = int(np.ceil((n / 4) ** (1.0 / 3.0) - 1e-9))
    m = max(m, 1)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
                points.append([x + 0.5/m, y + 0.5/m, z])
                points.append([x + 0.5/m, y, z + 0.5/m])
                points.append([x, y + 0.5/m, z + 0.5/m])
    points = np.array(points)
    min_c = points.min(axis=0)
    max_c = points.max(axis=0)
    extent = max_c - min_c
    scale = 1.0 / np.max(extent)
    points = (points - min_c) * scale - 0.5
    return points[:n]

def _random_init(n, seed=42):
    """Random initialization with rejection sampling for minimum distance."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n):
        for attempt in range(1000):
            p = rng.uniform(-0.45, 0.45, 3)
            if len(points) == 0:
                points.append(p)
                break
            d = np.min(np.linalg.norm(np.array(points) - p, axis=1))
            if d > 0.05:
                points.append(p)
                break
        else:
            points.append(p)
    return np.array(points[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Generate initial configurations
    inits = []
    
    # BCC
    try:
        bcc = _bcc_lattice(N)
        if bcc.shape == (N, 3):
            inits.append(('bcc', bcc))
    except:
        pass
    
    # FCC
    try:
        fcc = _fcc_lattice(N)
        if fcc.shape == (N, 3):
            inits.append(('fcc', fcc))
    except:
        pass
    
    # Simple cubic grid (from seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(('grid', grid[:N]))
    
    # Random inits
    for seed in [42, 123, 456, 789]:
        try:
            rnd = _random_init(N, seed)
            if rnd.shape == (N, 3):
                inits.append(('rand', rnd))
        except:
            pass
    
    # Optimize each initial configuration
    for name, centers in inits:
        if time.time() > deadline - 20:
            break
        
        # Try different temperature parameters for softmin
        for t in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 10:
                break
            
            try:
                opt_centers = _local_optimize(centers, t=t, max_iter=200)
                # Project back to valid region
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                score = _score(opt_centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = opt_centers.copy()
            except:
                pass
    
    # Final polish with very small t
    if best_centers is not None and time.time() < deadline - 5:
        try:
            opt_centers = _local_optimize(best_centers, t=0.005, max_iter=100)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            score = _score(opt_centers)
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
        except:
            pass
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
