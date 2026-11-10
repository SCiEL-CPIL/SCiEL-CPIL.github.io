import numpy as np
import time
import math

N = 150

def _energy(points):
    """Compute logarithmic energy using math.fsum for exact accumulation."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dist[iu[0], iu[1]])
    return math.fsum(vals.tolist())

def _grad_and_energy(points):
    """Compute log energy and Euclidean gradient."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dist[iu[0], iu[1]])
    energy = math.fsum(vals.tolist())

    coeff = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]

    row_sums = coeff.sum(axis=1)
    grad = points * row_sums[:, np.newaxis] - (coeff @ points)

    return energy, grad

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _tangent(v, x):
    dot = np.sum(v * x, axis=1, keepdims=True)
    return v - dot * x

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _halton_sequence(n, dims=3, start=0):
    """Generate Halton sequence points in [0,1)^dims, mapped to sphere."""
    def _halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    pts = np.zeros((n, dims))
    for d in range(dims):
        for i in range(n):
            pts[i, d] = _halton(i + start + 1, d + 2)
    
    # Map to sphere via uniform sampling
    z = 2.0 * pts[:, 2] - 1.0
    theta = 2.0 * np.pi * pts[:, 0]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _lbfgs_optimize(points, max_iter=800, ftol=1e-16, gtol=1e-14, deadline=None):
    """Riemannian L-BFGS via scipy on the sphere (projected)."""
    from scipy.optimize import minimize

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _grad_and_energy(pts)
        g_t = _tangent(g, pts)
        return e, g_t.reshape(-1)

    try:
        result = minimize(
            objective_and_grad,
            points.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 60,
                     'iprint': -1}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            cand = _project_to_sphere(result.x.reshape(-1))
            return cand, _energy(cand)
    except Exception:
        pass
    return points, _energy(points)

def _riemannian_cg(points, max_iter=400, tol=1e-13, deadline=None):
    """Riemannian conjugate gradient with backtracking on the sphere."""
    X = _project_to_sphere(points.copy())
    n = X.shape[0]

    e, g = _grad_and_energy(X)
    g_t = _tangent(g, X)

    if np.linalg.norm(g_t) < tol:
        return X, e

    d = -g_t
    g_old_t = g_t

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Backtracking line search along geodesic (projected) direction
        best_alpha = 1e-6
        best_e = e
        best_X = X.copy()
        alpha = 1.0
        for _ in range(30):
            X_new = _project_to_sphere(X + alpha * d)
            e_new = _energy(X_new)
            if e_new < best_e:
                best_e = e_new
                best_alpha = alpha
                best_X = X_new
            alpha *= 0.5
            if alpha < 1e-12:
                break

        X = best_X
        e = best_e

        e_new, g_new = _grad_and_energy(X)
        g_new_t = _tangent(g_new, X)
        if np.linalg.norm(g_new_t) < tol:
            break

        beta = (np.sum(g_new_t * g_new_t) / (np.sum(g_old_t * g_old_t) + 1e-30))
        if np.sum(g_new_t * g_old_t) > 0:
            beta = 0.0
        d = -g_new_t + beta * d
        d = _tangent(d, X)
        g_old_t = g_new_t

    return X, e

def _simulated_annealing(points, n_steps=5000, deadline=None):
    """Simulated annealing to escape local minima."""
    X = _project_to_sphere(points.copy())
    e = _energy(X)
    best_X = X.copy()
    best_e = e
    
    T = 0.1
    T_min = 1e-10
    cooling = (T_min / T) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and move it
        i = np.random.randint(N)
        # Random direction on tangent space
        rand_dir = np.random.randn(3)
        rand_dir -= np.sum(rand_dir * X[i]) * X[i]
        rand_dir /= (np.linalg.norm(rand_dir) + 1e-15)
        
        step_size = T * 0.1
        X_new = X.copy()
        X_new[i] = _project_to_sphere(X[i] + step_size * rand_dir)
        
        e_new = _energy(X_new)
        delta_e = e_new - e
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-15)):
            X = X_new
            e = e_new
            if e < best_e:
                best_e = e
                best_X = X.copy()
        
        T *= cooling
    
    return best_X, best_e

def _model_parameterize(theta_params):
    """MODEL-AND-OPTIMIZE parameterization using spherical coordinates."""
    lon = theta_params[:, 0]
    lat = np.clip(theta_params[:, 1], -np.pi/2 + 1e-12, np.pi/2 - 1e-12)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # ---- Build diverse initial configurations ----
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_halton_sequence(N))
    for s in range(3):
        inits.append(_random_points(N, seed=100 + s * 7))
    for s in range(8):
        inits.append(_random_points(N, seed=500 + s * 13))

    best_points = None
    best_energy = np.inf

    # ---- Phase 1: L-BFGS on each init ----
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        try:
            cand, cand_energy = _lbfgs_optimize(
                init_pts, max_iter=500, ftol=1e-14, gtol=1e-12, deadline=deadline
            )
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 2: Model-and-optimize fit ----
    if best_points is not None and time.time() < deadline - 30.0:
        try:
            bp = best_points
            lon = np.arctan2(bp[:, 1], bp[:, 0])
            lat = np.arcsin(np.clip(bp[:, 2], -1.0, 1.0))
            theta0 = np.column_stack([lon, lat, np.zeros(N)])

            def model_obj_and_grad(theta_flat):
                th = theta_flat.reshape(-1, 3)
                pts = _model_parameterize(th)
                pts = _project_to_sphere(pts)
                e, g = _grad_and_energy(pts)
                
                lat_th = np.clip(th[:, 1], -np.pi/2 + 1e-12, np.pi/2 - 1e-12)
                lon_th = th[:, 0]
                sl, cl = np.sin(lat_th), np.cos(lat_th)
                
                dx_dlon = -cl * np.sin(lon_th)
                dy_dlon = cl * np.cos(lon_th)
                dz_dlon = 0.0
                dx_dlat = -sl * np.cos(lon_th)
                dy_dlat = -sl * np.sin(lon_th)
                dz_dlat = cl
                
                g_lon = g[:, 0] * dx_dlon + g[:, 1] * dy_dlon + g[:, 2] * dz_dlon
                g_lat = g[:, 0] * dx_dlat + g[:, 1] * dy_dlat + g[:, 2] * dz_dlat
                
                grad_theta = np.zeros_like(theta_flat)
                grad_theta[0::3] = g_lon
                grad_theta[1::3] = g_lat
                return e, grad_theta

            from scipy.optimize import minimize
            res = minimize(
                model_obj_and_grad, theta0.reshape(-1), jac=True,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60, 'iprint': -1}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = _model_parameterize(res.x.reshape(-1, 3))
                cand = _project_to_sphere(cand)
                cand_energy = _energy(cand)
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 3: Simulated annealing to escape local minima ----
    if best_points is not None and time.time() < deadline - 20.0:
        try:
            cand, cand_energy = _simulated_annealing(best_points, n_steps=3000, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 4: Riemannian CG polish ----
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-14, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 5: Final tight L-BFGS polish ----
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            cand, cand_energy = _lbfgs_optimize(best_points, max_iter=800, ftol=1e-16, gtol=1e-15, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 6: Final CG polish ----
    if best_points is not None and time.time() < deadline - 2.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=400, tol=1e-14, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Fallback ----
    if best_points is None:
        best_points = _fibonacci_spiral(N)

    best_points = _project_to_sphere(best_points)

    # ---- Validation: ensure no points closer than 1e-9, finite, nonzero ----
    rng = np.random.RandomState(0)
    for _ in range(200):
        ok = True
        for i in range(N):
            if not np.all(np.isfinite(best_points[i])) or np.linalg.norm(best_points[i]) < 1e-12:
                ok = False
                break
        if not ok:
            best_points = _project_to_sphere(best_points + rng.randn(N, 3) * 1e-6)
            continue
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-9:
                    best_points[j] += rng.randn(3) * 1e-7
                    ok = False
                    break
            if not ok:
                break
        best_points = _project_to_sphere(best_points)
        if ok:
            break

    return best_points
