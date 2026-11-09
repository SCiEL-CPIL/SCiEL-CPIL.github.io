import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(points):
    """Compute packing score: min of all pairwise distances/2 and boundary distances."""
    if not np.all(np.isfinite(points)) or points.shape != (N, 2):
        return -np.inf
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)

def _hex_grid_init(n, seed=42):
    """Generate hexagonal lattice initial configuration."""
    rng = np.random.RandomState(seed)
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.18, 0.32, 25):
        pts = []
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) < 1.0 - spacing * 0.6:
                    pts.append([x, y])
        if len(pts) < n:
            continue
        pts = np.array(pts)
        # Sort by distance from center, take closest n
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii)[:n]
        candidate = pts[idx]
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_pts = candidate.copy()
    if best_pts is None:
        # Fallback: random
        r = np.sqrt(rng.uniform(0, 0.9, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        best_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return best_pts

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hex grid
    inits.append(_hex_grid_init(N))
    
    # 2. Ring-based configurations
    for ring_r in [0.2, 0.25, 0.3]:
        pts = [[0.0, 0.0]]
        remaining = N - 1
        r = ring_r
        while remaining > 0 and r < 0.95:
            circumference = 2 * np.pi * r
            count = min(int(2 * np.pi * r / (ring_r * 1.1)), remaining)
            if count < 1:
                count = 1
            for i in range(count):
                theta = 2 * np.pi * i / count + np.pi / count
                pts.append([r * np.cos(theta), r * np.sin(theta)])
            remaining -= count
            r += ring_r * 0.8
        pts = np.array(pts[:N])
        if len(pts) == N:
            inits.append(pts)
    
    # 3. Random configurations
    rng = np.random.RandomState(123)
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 0.92, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    best_points = inits[0].copy()
    best_score = _score(best_points)
    
    # PSO optimization
    n_particles = 12
    max_iterations = 80
    
    # Initialize particles around the best initial config
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    p_best = np.zeros((n_particles, N, 2))
    p_best_scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        if i < len(inits):
            particles[i] = inits[i].copy()
        else:
            base = inits[i % len(inits)]
            particles[i] = base + rng.normal(0, 0.02, (N, 2))
        # Project back inside unit circle
        norms = np.linalg.norm(particles[i], axis=1)
        mask = norms > 0.95
        if np.any(mask):
            particles[i][mask] *= 0.95 / norms[mask]
        s = _score(particles[i])
        p_best[i] = particles[i].copy()
        p_best_scores[i] = s
    
    # Global best
    g_best_idx = np.argmax(p_best_scores)
    g_best = p_best[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    
    if g_best_score > best_score:
        best_score = g_best_score
        best_points = g_best.copy()
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for iteration in range(max_iterations):
        if time_left() < 15.0:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = rng.random((N, 2))
            r2 = rng.random((N, 2))
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (g_best - particles[i]))
            
            # Clamp velocity
            v_max = 0.05
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Update position
            particles[i] += velocities[i]
            
            # Project back inside unit circle
            norms = np.linalg.norm(particles[i], axis=1)
            mask = norms > 0.98
            if np.any(mask):
                particles[i][mask] *= 0.98 / norms[mask]
            
            # Evaluate
            s = _score(particles[i])
            
            # Update personal best
            if s > p_best_scores[i]:
                p_best_scores[i] = s
                p_best[i] = particles[i].copy()
                # Update global best
                if s > g_best_score:
                    g_best_score = s
                    g_best = p_best[i].copy()
                    if s > best_score:
                        best_score = s
                        best_points = g_best.copy()
        
        # Adaptive inertia
        w = 0.7 - 0.3 * (iteration / max_iterations)
    
    # Final local polish on the best configuration
    if time_left() > 10.0:
        def objective(x_flat):
            pts = x_flat.reshape(N, 2)
            if not np.all(np.isfinite(pts)):
                return 1e10
            # Soft min with high beta
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, 1.0)
            pair_vals = dists / 2.0
            norms = np.sqrt(np.sum(pts ** 2, axis=1))
            boundary_vals = 1.0 - norms
            all_vals = np.concatenate([pair_vals[np.triu_indices(N, k=1)], boundary_vals])
            beta = 100.0
            shifted = all_vals - np.min(all_vals)
            log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
            soft_min = np.min(all_vals) - np.log(np.sum(np.exp(-beta * all_vals))) / beta
            return -soft_min
        
        def gradient(x_flat):
            pts = x_flat.reshape(N, 2)
            grad = np.zeros_like(x_flat)
            beta = 100.0
            
            # Compute all constraint values
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, 1.0)
            pair_vals = dists / 2.0
            norms = np.sqrt(np.sum(pts ** 2, axis=1))
            boundary_vals = 1.0 - norms
            all_vals = np.concatenate([pair_vals[np.triu_indices(N, k=1)], boundary_vals])
            
            exp_vals = np.exp(-beta * all_vals)
            sum_exp = np.sum(exp_vals)
            
            # Pairwise gradients
            i_idx, j_idx = np.triu_indices(N, k=1)
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                diff = pts[i] - pts[j]
                d = dists[i, j]
                if d < 1e-12:
                    continue
                dv_di = diff / (2.0 * d)
                dv_dj = -dv_di
                w_k = exp_vals[k] / sum_exp
                grad[i*2:i*2+2] += w_k * dv_di
                grad[j*2:j*2+2] += w_k * dv_dj
            
            # Boundary gradients
            for i in range(N):
                norm_p = norms[i]
                if norm_p < 1e-12:
                    continue
                dv_di = -pts[i] / norm_p
                w_k = exp_vals[N*(N-1)//2 + i] / sum_exp
                grad[i*2:i*2+2] += w_k * dv_di
            
            return -grad
        
        try:
            x0 = best_points.flatten()
            res = minimize(
                objective, x0, jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < objective(x0):
                candidate = res.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        best_points = inits[0]
    
    # Final validity check and projection
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        best_points[mask] *= 0.999 / norms[mask]
    
    return best_points
