"""Pack 200 equal circles in a unit square using hexagonal init + gradient-based relaxation."""
import time
import numpy as np

N = 200
DEADLINE = 115.0  # safety margin before 120s limit


def _score(c):
    """Compute the radius: min of half min-pair-distance and min wall distance."""
    # Wall distances
    wall = np.min(np.minimum(0.5 - np.abs(c), 0.5 - np.abs(c)))
    # Pairwise distances
    # Use broadcasting for 200x200 which is fine
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(wall, min_pair)


def _soft_min_score(c, temp=0.005):
    """Soft approximation of the score for gradient computation."""
    # Wall term: min_i min_k (0.5 - |c_ik|)
    # Use soft-min over all wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_flat = wall_dists.ravel()
    # Soft min: -T * log(sum(exp(-x/T)))
    # For numerical stability, subtract max
    m = np.max(wall_flat)
    soft_wall = m - temp * np.log(np.sum(np.exp((m - wall_flat) / temp)))
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_flat = dist.ravel()
    m2 = np.max(pair_flat)
    soft_pair = m2 - temp * np.log(np.sum(np.exp((m2 - pair_flat) / temp)))
    
    # Combine: soft min of soft_wall and soft_pair
    # Actually, the score is min(soft_wall, soft_pair/2)
    # Let's just use a combined soft-min
    vals = np.array([soft_wall, soft_pair / 2.0])
    m3 = np.max(vals)
    soft_score = m3 - temp * np.log(np.sum(np.exp((m3 - vals) / temp)))
    
    return soft_score


def _grad_soft_score(c, temp=0.005):
    """Compute gradient of soft score w.r.t. centers."""
    # Wall term
    wall_dists = 0.5 - np.abs(c)
    wall_flat = wall_dists.ravel()
    m = np.max(wall_flat)
    exp_w = np.exp((m - wall_flat) / temp)
    sum_exp_w = np.sum(exp_w)
    # d(soft_wall)/d(c_ik) = -d(0.5-|c_ik|)/dc_ik * (exp_w_i / sum_exp_w)
    # d(0.5-|x|)/dx = -sign(x)
    # So d(soft_wall)/dc_ik = sign(c_ik) * (exp_w_i / sum_exp_w)
    
    # Reshape exp_w back to (N, 2)
    exp_w_2d = exp_w.reshape(N, 2)
    grad_wall = np.sign(c) * (exp_w_2d / sum_exp_w)
    
    # Pair term
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_flat = dist.ravel()
    m2 = np.max(pair_flat)
    exp_p = np.exp((m2 - pair_flat) / temp)
    sum_exp_p = np.sum(exp_p)
    exp_p_2d = exp_p.reshape(N, N)
    np.fill_diagonal(exp_p_2d, 0)
    
    # d(soft_pair)/d(c_i) = sum_j (1/(2*dist_ij)) * (diff_ij) * (exp_p_ij / sum_exp_p)
    # where diff_ij = c_i - c_j
    # grad_pair[i] = sum_j (diff_ij / (2 * dist_ij)) * (exp_p_ij / sum_exp_p)
    
    # Normalize: diff_ij / dist_ij is unit vector from j to i
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-12, dist, 1.0)
    unit_vec = diff / dist_safe[:, :, np.newaxis]
    
    # Weighted sum over j
    weights = exp_p_2d / sum_exp_p
    grad_pair = np.sum(unit_vec * weights[:, :, np.newaxis], axis=1)
    
    # The score uses soft_pair / 2, so gradient is halved
    grad_pair *= 0.5
    
    # Combined gradient
    vals = np.array([m - temp * np.log(sum_exp_w), m2 - temp * np.log(sum_exp_p)])
    m3 = np.max(vals)
    exp_v = np.exp((m3 - vals) / temp)
    sum_exp_v = np.sum(exp_v)
    
    grad_soft = exp_v[0] / sum_exp_v * grad_wall + exp_v[1] / sum_exp_v * grad_pair
    
    return grad_soft


def _hex_init():
    """Initialize centers on a hexagonal lattice clipped to the square."""
    # Hexagonal packing: rows offset by half spacing
    # Try to fit ~200 points
    # For hex packing, density is pi/(2*sqrt(3)) ~ 0.9069
    # Area per circle ~ 1/200 = 0.005, so spacing ~ sqrt(0.005 * 2*sqrt(3)/pi) ~ 0.073
    # Let's try spacing s and see how many fit
    
    best_c = None
    best_score = -1
    
    for s in [0.07, 0.072, 0.074, 0.076, 0.078, 0.08]:
        dx = s
        dy = s * np.sqrt(3) / 2
        # Generate hex lattice points
        points = []
        # Number of columns
        ncols = int(1.0 / dx) + 2
        nrows = int(1.0 / dy) + 2
        for row in range(nrows):
            for col in range(ncols):
                x = (col + 0.5 * (row % 2)) * dx - 0.5
                y = row * dy - 0.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        
        if len(points) < N:
            continue
        
        # If too many, select N points (e.g., random or grid-based selection)
        if len(points) > N:
            # Select N points that maximize minimum distance (simple: take first N or random)
            # For simplicity, take a random subset
            idx = np.random.choice(len(points), N, replace=False)
            points = [points[i] for i in idx]
        
        c = np.array(points)
        # Ensure exactly N points
        if len(c) != N:
            if len(c) < N:
                # Pad with random points
                extra = np.random.uniform(-0.45, 0.45, (N - len(c), 2))
                c = np.vstack([c, extra])
            else:
                c = c[:N]
        
        sc = _score(c)
        if sc > best_score:
            best_score = sc
            best_c = c.copy()
    
    if best_c is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c


def _relax(c, time_limit):
    """Gradient-based relaxation with adaptive learning rate."""
    start_time = time.time()
    c = c.copy()
    lr = 0.001
    temp = 0.005
    
    best_c = c.copy()
    best_score = _score(c)
    
    n_iter = 0
    while time.time() - start_time < time_limit:
        n_iter += 1
        
        # Compute gradient
        grad = _grad_soft_score(c, temp)
        
        # Gradient ascent (we want to maximize score)
        c_new = c + lr * grad
        
        # Project back into square with margin
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Check score
        sc = _score(c_new)
        
        if sc > best_score:
            best_score = sc
            best_c = c_new.copy()
            # Decrease learning rate slightly
            lr *= 0.999
            # Decrease temperature for sharper gradient
            temp = max(0.001, temp * 0.999)
        else:
            # Rejected step, reduce learning rate
            lr *= 0.95
            temp = max(0.001, temp * 0.99)
        
        c = c_new
        
        # Early stop if improvement is tiny
        if n_iter > 100 and best_score > 0.036:
            # Check convergence
            if n_iter % 50 == 0:
                # Optional: check if we've plateaued
                pass
    
    return best_c, best_score


def construct():
    start_time = time.time()
    
    # Multiple restarts with different initializations
    best_c = None
    best_score = -1
    
    # Restart 1: Hexagonal init
    c_init = _hex_init()
    remaining = DEADLINE - (time.time() - start_time) - 5
    c1, s1 = _relax(c_init, max(0, remaining * 0.4))
    if s1 > best_score:
        best_score = s1
        best_c = c1
    
    # Restart 2: Perturbed hex
    if time.time() - start_time < DEADLINE - 10:
        c_init2 = _hex_init()
        c_init2 += np.random.normal(0, 0.005, c_init2.shape)
        c_init2 = np.clip(c_init2, -0.49, 0.49)
        remaining = DEADLINE - (time.time() - start_time) - 5
        c2, s2 = _relax(c_init2, max(0, remaining * 0.3))
        if s2 > best_score:
            best_score = s2
            best_c = c2
    
    # Restart 3: Square grid init
    if time.time() - start_time < DEADLINE - 10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c_init3 = grid[:N].copy()
        remaining = DEADLINE - (time.time() - start_time) - 5
        c3, s3 = _relax(c_init3, max(0, remaining * 0.3))
        if s3 > best_score:
            best_score = s3
            best_c = c3
    
    # Final polish: a few more iterations with tighter parameters
    if best_c is not None and time.time() - start_time < DEADLINE - 3:
        remaining = DEADLINE - (time.time() - start_time) - 2
        c_final, s_final = _relax(best_c, max(0, remaining))
        if s_final > best_score:
            best_score = s_final
            best_c = c_final
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    assert best_c.shape == (N, 2)
    assert np.all(np.isfinite(best_c))
    
    return best_c
