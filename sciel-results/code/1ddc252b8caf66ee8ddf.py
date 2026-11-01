"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b


def _score_from_centres(c):
    """Compute the largest common radius for given centres.
    Returns (radius, valid_flag)."""
    n = c.shape[0]
    if not np.all(np.isfinite(c)):
        return 0.0, False
    if not (np.all(c >= -0.5) and np.all(c <= 0.5)):
        return 0.0, False
    if n < 2:
        return 0.5, True
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if min_boundary < 0:
        return 0.0, False
    
    min_pair_dist = np.inf
    for i in range(n - 1):
        diffs = c[i+1:] - c[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        chunk_min = np.min(dists)
        if chunk_min < min_pair_dist:
            min_pair_dist = chunk_min
    
    radius = min(min_pair_dist / 2.0, min_boundary)
    return radius, True


def _soft_min(values, temperature=0.01):
    """Compute soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    if len(values) == 0:
        return np.inf
    min_val = np.min(values)
    # Soft min: -t * log(sum(exp(-values/t)))
    # Numerically stable: min_val - t * log(sum(exp(-(values - min_val)/t)))
    shifted = (values - min_val) / temperature
    log_sum = np.log(np.sum(np.exp(-shifted)))
    return min_val - temperature * log_sum


def _soft_min_objective(params, temperature=0.01):
    """Smooth surrogate objective using soft-min of all constraints."""
    c = np.clip(params.reshape(-1, 2), -0.5, 0.5)
    n = c.shape[0]
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances / 2
    pair_dists = []
    for i in range(n - 1):
        diffs = c[i+1:] - c[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        pair_dists.append(dists / 2.0)
    
    if pair_dists:
        all_dists = np.concatenate([boundary_dists.flatten(), np.concatenate(pair_dists)])
    else:
        all_dists = boundary_dists.flatten()
    
    # Soft min of all distances
    soft_min_val = _soft_min(all_dists, temperature)
    
    # Add penalty for being outside (shouldn't happen due to clip, but just in case)
    out_of_bounds = np.sum(np.abs(c) > 0.5)
    
    return -soft_min_val + 10.0 * out_of_bounds


def _initialize_hexagonal(n_circles):
    """Initialize circles in a hexagonal pattern."""
    circles = []
    # Try different spacings to get close to n_circles
    best_spacing = 0.04
    best_count = 0
    
    for spacing in np.arange(0.03, 0.06, 0.001):
        count = 0
        x = -0.5 + spacing
        row = 0
        while x < 0.5:
            y = -0.5 + spacing
            while y < 0.5:
                x_offset = x + (row % 2) * spacing * 0.866
                if -0.5 + spacing/2 <= x_offset <= 0.5 - spacing/2 and \
                   -0.5 + spacing/2 <= y <= 0.5 - spacing/2:
                    count += 1
                y += spacing * 0.866
            x += spacing
            row += 1
        
        if count >= n_circles and count < best_count or best_count == 0:
            if count >= n_circles:
                best_spacing = spacing
                best_count = count
    
    # Generate hexagonal positions with best_spacing
    spacing = best_spacing
    x = -0.5 + spacing
    row = 0
    while x < 0.5 and len(circles) < n_circles:
        y = -0.5 + spacing
        while y < 0.5 and len(circles) < n_circles:
            x_offset = x + (row % 2) * spacing * 0.866
            if -0.5 + spacing/2 <= x_offset <= 0.5 - spacing/2 and \
               -0.5 + spacing/2 <= y <= 0.5 - spacing/2:
                circles.append([x_offset, y])
            y += spacing * 0.866
        x += spacing
        row += 1
    
    if len(circles) >= n_circles:
        return np.array(circles[:n_circles])
    else:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]


def _force_directed_init(n_circles, iterations=100):
    """Initialize with force-directed layout to spread circles."""
    np.random.seed(42)
    c = np.random.uniform(-0.45, 0.45, (n_circles, 2))
    
    for it in range(iterations):
        # Compute repulsion forces
        forces = np.zeros_like(c)
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                    diff = np.random.uniform(-1e-5, 1e-5, 2)
                # Repulsion force (inverse square)
                force = diff / (dist**2) * 0.01
                forces[i] += force
                forces[j] -= force
        
        # Boundary forces (push back if too close to edge)
        for i in range(n_circles):
            for k in range(2):
                if c[i, k] < -0.4:
                    forces[i, k] += (0.4 + c[i, k]) * 0.1
                elif c[i, k] > 0.4:
                    forces[i, k] -= (c[i, k] - 0.4) * 0.1
        
        # Update positions
        c += forces * 0.1
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _local_optimize(c, max_iter=200, pgtol=1e-8):
    """Local optimization using L-BFGS-B with soft-min objective."""
    def obj(params):
        return _soft_min_objective(params, temperature=0.005)
    
    try:
        result = fmin_l_bfgs_b(
            obj,
            c.flatten(),
            bounds=[(-0.5, 0.5)] * (2 * N),
            maxiter=max_iter,
            pgtol=pgtol,
            iprint=0
        )
        c_opt = np.clip(result[0].reshape(-1, 2), -0.5, 0.5)
        return c_opt
    except Exception:
        return c


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_radius = 0.0
    best_c = None
    
    def _update_best(c):
        nonlocal best_radius, best_c
        r, valid = _score_from_centres(c)
        if valid and r > best_radius:
            best_radius = r
            best_c = c.copy()
    
    # Phase 1: Multiple initializations
    inits = []
    
    # Hexagonal initialization
    c_hex = _initialize_hexagonal(N)
    inits.append(c_hex)
    
    # Grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # Force-directed initialization
    if time.time() < deadline - 60:
        c_fd = _force_directed_init(N, iterations=50)
        inits.append(c_fd)
    
    # Evaluate initializations
    for c_init in inits:
        c_init = np.clip(c_init, -0.5, 0.5)
        _update_best(c_init)
    
    # Phase 2: Optimize from each initialization with decreasing temperature
    temperatures = [0.05, 0.02, 0.005]
    
    for c_init in inits:
        if time.time() > deadline - 20:
            break
        
        c_current = c_init.copy()
        
        for temp in temperatures:
            if time.time() > deadline - 15:
                break
            
            # Optimize with this temperature
            def obj(params):
                return _soft_min_objective(params, temperature=temp)
            
            try:
                result = fmin_l_bfgs_b(
                    obj,
                    c_current.flatten(),
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    maxiter=100,
                    pgtol=1e-6,
                    iprint=0
                )
                c_opt = np.clip(result[0].reshape(-1, 2), -0.5, 0.5)
                _update_best(c_opt)
                c_current = c_opt
            except Exception:
                break
    
    # Phase 3: Final polish with very tight tolerance
    if time.time() < deadline - 5 and best_c is not None:
        try:
            def obj_polish(params):
                return _soft_min_objective(params, temperature=0.001)
            
            result = fmin_l_bfgs_b(
                obj_polish,
                best_c.flatten(),
                bounds=[(-0.5, 0.5)] * (2 * N),
                maxiter=300,
                pgtol=1e-10,
                iprint=0
            )
            c_polish = np.clip(result[0].reshape(-1, 2), -0.5, 0.5)
            _update_best(c_polish)
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement if time permits
    if time.time() < deadline - 3 and best_c is not None:
        c_cd = best_c.copy()
        step = 0.002
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                for k in range(2):
                    old = c_cd[i, k]
                    for delta in [step, -step]:
                        new_val = old + delta
                        if -0.5 <= new_val <= 0.5:
                            c_cd[i, k] = new_val
                            r_new, _ = _score_from_centres(c_cd)
                            if r_new > best_radius:
                                best_radius = r_new
                                best_c = c_cd.copy()
                                improved = True
                            else:
                                c_cd[i, k] = old
            if improved:
                step = max(step * 0.7, 0.0001)
            else:
                step = min(step * 1.3, 0.01)
    
    # Ensure validity
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
