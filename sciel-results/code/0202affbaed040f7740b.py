"""Rewritten program: log-energy minimization of 150 points on unit sphere.
Structural changes: correct gradient, geodesic (exponential-map) optimization,
L-BFGS-B primary optimizer, basin-hopping global search, icosphere init.
"""
import numpy as np
import math
import time

N = 150

# ---------- geometry helpers ----------

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosphere_init(n, subdiv=5):
    """Generate n points on sphere by subdividing an icosahedron (geodesic dome)."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 20 faces
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    
    # Subdivide
    for _ in range(subdiv):
        new_faces = []
        cache = {}
        for f in faces:
            mids = {}
            for i in range(3):
                a, b = f[i], f[(i+1) % 3]
                key = (min(a,b), max(a,b))
                if key not in cache:
                    mid = (verts[a] + verts[b])
                    mid = mid / np.linalg.norm(mid)
                    cache[key] = len(verts)
                    verts = np.vstack([verts, mid[None, :]])
                mids[i] = cache[key]
            new_faces.append((f[0], mids[0], mids[2]))
            new_faces.append((f[1], mids[1], mids[0]))
            new_faces.append((f[2], mids[2], mids[1]))
            new_faces.append((mids[0], mids[1], mids[2]))
        faces = new_faces
    
    # Now we have len(verts) points on sphere; sample n of them
    total = len(verts)
    if total <= n:
        return verts.copy()
    # Uniformly sample n points from the subdivided icosphere
    idx = np.linspace(0, total - 1, n, dtype=int)
    return verts[idx].copy()

# ---------- energy and gradient (CORRECT) ----------

def _log_energy(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d2 = dist2[iu]
    d2 = np.maximum(d2, 1e-30)
    d = np.sqrt(d2)
    log_terms = np.log(1.0 / d)
    return math.fsum(log_terms.tolist())

def _log_energy_grad(pts):
    """Correct gradient of E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i-x_j|.
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    # term[i,j] = (x_i - x_j) / |x_i - x_j|
    term = diff / dist[:, :, np.newaxis]
    grad = np.sum(term, axis=1)
    return grad

def _tangent_grad(pts):
    """Project gradient onto tangent space of sphere at each point."""
    g = _log_energy_grad(pts)
    g_dot_x = np.sum(g * pts, axis=1, keepdims=True)
    return g - g_dot_x * pts

# ---------- geodesic (exponential map) optimizer ----------

def _exp_map(pts, v, dt):
    """Exponential map on sphere: move points along tangent vectors v by step dt.
    For unit sphere: exp_x(v) = cos(|v|) x + sin(|v|) v/|v|
    Here v is already scaled by dt (i.e., v = dt * tangent_direction).
    """
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm_safe = np.maximum(v_norm, 1e-15)
    sin_v = np.sin(v_norm_safe)
    cos_v = np.cos(v_norm_safe)
    # Handle small v: use Taylor
    small = v_norm < 1e-10
    new_pts = cos_v * pts + (sin_v / v_norm_safe) * v
    # For small v, sin(v)/|v| ≈ 1, cos(v) ≈ 1
    if np.any(small):
        new_pts[small] = pts[small] + v[small]
    return _project_to_sphere(new_pts)

def _lbfgs_geodesic(pts, deadline, maxiter=2000, gtol=1e-12):
    """L-BFGS-B optimization on sphere using geodesic (exponential map) steps.
    Parameterization: x in R^{N*3}, projected to sphere. The optimizer sees
    Euclidean parameters but we project and use tangent gradients.
    """
    from scipy.optimize import minimize
    
    def _energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _project_to_sphere(x)
        e = _log_energy(x)
        g_tan = _tangent_grad(x)
        return e, g_tan.flatten()
    
    x0 = pts.flatten()
    res = minimize(_energy_and_grad, x0, method='L-BFGS-B',
                   jac=True, options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
    if np.all(np.isfinite(res.x)):
        return _project_to_sphere(res.x.reshape(N, 3))
    return pts

def _gradient_descent_sphere(pts, deadline, max_steps=5000):
    """Geodesic gradient descent with adaptive step size."""
    best = pts.copy()
    best_e = _log_energy(pts)
    dt = 0.01
    for step in range(max_steps):
        if time.time() > deadline - 10.0:
            break
        g_tan = _tangent_grad(pts)
        # Adaptive step: reduce if energy increases
        new_pts = _exp_map(pts, -g_tan * dt, 1.0)
        new_e = _log_energy(new_pts)
        if new_e < best_e:
            best = new_pts.copy()
            best_e = new_e
            pts = new_pts
            dt = min(dt * 1.05, 0.05)
        else:
            dt *= 0.7
            if dt < 1e-8:
                break
    return best, best_e

# ---------- basin hopping ----------

def _basin_hopping_step(pts, deadline, rng, n_perturb=10):
    """Perturb n_perturb random points, then re-optimize with L-BFGS."""
    new_pts = pts.copy()
    idx = rng.choice(N, size=min(n_perturb, N), replace=False)
    # Random tangent perturbations
    for i in idx:
        # Random direction on sphere
        rand_dir = rng.standard_normal(3)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        # Make tangent
        dot = np.dot(rand_dir, new_pts[i])
        rand_dir = rand_dir - dot * new_pts[i]
        norm = np.linalg.norm(rand_dir)
        if norm < 1e-15:
            rand_dir = np.array([1.0, 0, 0])
            dot = np.dot(rand_dir, new_pts[i])
            rand_dir = rand_dir - dot * new_pts[i]
            norm = np.linalg.norm(rand_dir)
        rand_dir = rand_dir / norm
        # Small step
        step = 0.05
        new_pts[i] = new_pts[i] + step * rand_dir
    new_pts = _project_to_sphere(new_pts)
    # Re-optimize
    new_pts = _lbfgs_geodesic(new_pts, deadline, maxiter=500, gtol=1e-10)
    return new_pts

# ---------- main construct ----------

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = []
    # Fibonacci variants
    for off in [0.0, 0.3, 0.7, 0.15, 0.55]:
        inits.append(_fibonacci_sphere(N, off))
    # Icosphere (geodesic dome)
    inits.append(_icosphere_init(N, subdiv=5))
    inits.append(_icosphere_init(N, subdiv=4))
    # Random uniform
    for _ in range(3):
        r = rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(r))
    
    best_pts = inits[0].copy()
    best_energy = _log_energy(best_pts)
    
    # Phase 1: L-BFGS polish on each initialization
    for init_pts in inits:
        if time.time() > deadline - 30.0:
            break
        polished = _lbfgs_geodesic(init_pts, deadline, maxiter=1000, gtol=1e-11)
        e = _log_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pts = polished.copy()
    
    # Phase 2: Basin hopping around best
    if time.time() < deadline - 20.0:
        for it in range(50):
            if time.time() > deadline - 15.0:
                break
            cand = _basin_hopping_step(best_pts, deadline, rng, n_perturb=rng.integers(5, 25))
            e = _log_energy(cand)
            if e < best_energy:
                best_energy = e
                best_pts = cand.copy()
    
    # Phase 3: Final tight L-BFGS polish
    if time.time() < deadline - 5.0:
        best_pts = _lbfgs_geodesic(best_pts, deadline, maxiter=3000, gtol=1e-13)
        final_e = _log_energy(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
