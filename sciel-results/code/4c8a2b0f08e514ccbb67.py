"""Dzugutov cluster global minima: Icosahedral geometric optimization + relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

_START = time.time()
TIME_LIMIT = 290.0  # Safety margin under 300s kill
_STOP = threading.Event()

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, TIME_LIMIT - _elapsed())


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    rv = r[iu]
    if rv.size == 0:
        return 0.0
    e = np.zeros_like(rv)
    m1 = rv < _a
    m2 = (rv >= _a) & (rv < _b)
    if m1.any():
        r1 = rv[m1]
        e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    if m2.any():
        r2 = rv[m2]
        e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(np.sum(e))


def _relax(positions, deadline=None):
    """Relax using the provided relaxer."""
    pos = np.asarray(positions, dtype=float)
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    # Fallback: simple gradient descent if relax fails
    p = pos.copy()
    best_e = _energy(p)
    best_p = p.copy()
    for it in range(200):
        if deadline is not None and _elapsed() > deadline:
            break
        if _STOP.is_set():
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        s = min(0.2, 0.01 * gn)
        new_p = p - s * g / gn
        if not np.all(np.isfinite(new_p)):
            s *= 0.1
            new_p = p - s * g / gn
            if not np.all(np.isfinite(new_p)):
                break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
        else:
            if s < 1e-6:
                break
    return best_e, best_p


def _get_icosahedron_vertices():
    """Returns 12 unit vectors for icosahedron vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _get_shell_points(shell_idx):
    """
    Generate unit vectors for Mackay icosahedral shells.
    shell_idx 0: 1 point (center)
    shell_idx 1: 12 points (vertices)
    shell_idx 2: 42 points (face centers + edge midpoints)
    shell_idx 3: 92 points (approx)
    shell_idx 4: 120 points (approx)
    """
    if shell_idx == 0:
        return np.array([[0, 0, 0]])
    
    ico_verts = _get_icosahedron_vertices()
    
    if shell_idx == 1:
        return ico_verts
    
    if shell_idx == 2:
        # 20 face centers + 22 edge midpoints (subset of 30)
        # We need 42 points total.
        # Faces of icosahedron:
        faces = [
            [0, 1, 8], [0, 8, 4], [0, 4, 11], [0, 11, 2], [0, 2, 1],
            [1, 2, 5], [2, 11, 3], [11, 4, 9], [4, 8, 10], [8, 1, 6],
            [5, 2, 3], [3, 11, 7], [7, 9, 3], [9, 4, 10], [10, 8, 6],
            [6, 1, 5], [5, 3, 7], [7, 9, 10], [10, 6, 5], [6, 10, 8]
        ]
        pts = []
        for f in faces:
            c = ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]
            c /= np.linalg.norm(c)
            pts.append(c)
        
        # Edges (need 22 more to make 42)
        edges = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 11],
            [1, 2], [1, 5], [1, 6], [1, 8],
            [2, 3], [2, 5], [2, 11],
            [3, 7], [3, 9], [3, 11],
            [4, 9], [4, 10], [4, 11],
            [5, 6], [5, 7],
            [6, 7], [6, 10]
        ]
        for e in edges:
            c = ico_verts[e[0]] + ico_verts[e[1]]
            c /= np.linalg.norm(c)
            pts.append(c)
        
        return np.array(pts)
    
    # For higher shells, use a more dense distribution or known Mackay points
    # Simplified: Use Fibonacci sphere for higher shells as exact Mackay points are complex
    # But Mackay shells have specific symmetry. Let's try to generate points on the sphere 
    # that respect icosahedral symmetry if possible, or just use a dense set.
    # For N=38,45,50,55,57,60, we only need shells 0, 1, 2, and part of 3.
    # Shell 3 has 92 points. We need to generate 92 points.
    if shell_idx == 3:
        # Generate 92 points. 
        # We can use the icosahedron vertices, face centers, edge midpoints, and then add more.
        # A better approach for Shell 3 in Mackay is to take the icosahedron and subdivide.
        # However, for the purpose of seeding, a dense Fibonacci sphere is a reasonable approximation 
        # for the "shape" if we allow the optimizer to adjust radii.
        # Actually, let's try to construct it properly if possible.
        # The 92 points of the 3rd shell in Mackay icosahedron correspond to the vertices of 
        # a more complex polyhedron.
        # Given time constraints, I will use a high-quality Fibonacci sphere for shell 3+ 
        # and rely on the optimization to find the best radii. The symmetry breaking is minimal 
        # if the initial guess is good.
        n_pts = 92
        rng = np.random.default_rng(shell_idx)
        # Use a deterministic Fibonacci sphere for reproducibility
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            z = 1.0 - (i + 0.5) / n_pts * 2.0
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        pts = np.array(pts)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    # Shell 4 and above
    n_pts = 120 if shell_idx == 4 else 150
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = []
    for i in range(n_pts):
        z = 1.0 - (i + 0.5) / n_pts * 2.0
        r = np.sqrt(1.0 - z*z)
        theta = golden_angle * i
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _build_mackay_cluster(n, params):
    """
    Build a cluster from Mackay shells given parameters.
    params: array of shell radii [r0, r1, r2, ...]
    """
    # Determine how many shells we need
    shell_counts = [1, 12, 42, 92, 120]
    points = []
    remaining = n
    for i, count in enumerate(shell_counts):
        if remaining <= 0:
            break
        take = min(count, remaining)
        if i == 0:
            # Center
            points.append(np.zeros(3) * params[0])
            remaining -= 1
        else:
            shell_pts = _get_shell_points(i)
            # Take first 'take' points from this shell
            # For shell 1, 2, we have exact counts. For 3, 4, we take a subset.
            # It's better to take a symmetric subset if possible, but for now just first N.
            # To be more symmetric, we could shuffle with a fixed seed.
            if take < len(shell_pts):
                # Simple selection
                selected = shell_pts[:take]
            else:
                selected = shell_pts
            
            r_i = params[i] if i < len(params) else 1.0
            for p in selected:
                points.append(p * r_i)
            remaining -= take
    
    if len(points) < n:
        # Fill remaining with random points on outer shell
        rng = np.random.default_rng(42)
        last_shell_idx = len(shell_counts) - 1
        r_last = params[last_shell_idx] if last_shell_idx < len(params) else 1.0
        while len(points) < n:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            points.append(v * r_last)
            
    return np.array(points[:n])


def _optimize_geometric_params(n, initial_params, deadline):
    """
    Optimize shell radii to minimize energy.
    """
    def objective(params):
        if _STOP.is_set() or _elapsed() > deadline:
            return 1e10
        # Ensure radii are positive
        params = np.abs(params)
        try:
            pos = _build_mackay_cluster(n, params)
            # Check minimum distance constraint roughly
            # If atoms are too close, energy will be huge anyway due to r^-m term
            e = _energy(pos)
            return e
        except Exception:
            return 1e10

    # Initial guess: Mackay radii are 1, 1+1/phi, 1+2/phi...
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ideal_radii = [0.0, 1.0, 1.0 + 1.0/phi, 1.0 + 2.0/phi, 1.0 + 3.0/phi]
    # Scale factor is usually around 1.0 for Dzugutov
    x0 = initial_params if initial_params is not None else np.array(ideal_radii[:5])
    
    # Bound radii to reasonable values
    bounds = [(0.5, 2.5) for _ in range(len(x0))]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'maxfun': 100, 'disp': False})
        if res.success or res.fun < 1e9:
            return res.x, res.fun
    except Exception:
        pass
    
    return x0, objective(x0)


def _worker(seed_type, n, deadline, worker_id):
    """
    Worker function: Build a specific type of seed, optimize geometry, then relax.
    """
    best_e = np.inf
    best_pos = None
    
    # Define different geometric strategies
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ideal_radii = [0.0, 1.0, 1.0 + 1.0/phi, 1.0 + 2.0/phi, 1.0 + 3.0/phi]
    
    if seed_type == 0:
        # Standard Mackay
        init_params = np.array(ideal_radii[:5])
    elif seed_type == 1:
        # Slightly expanded Mackay
        init_params = np.array(ideal_radii[:5]) * 1.05
    elif seed_type == 2:
        # Slightly contracted Mackay
        init_params = np.array(ideal_radii[:5]) * 0.95
    elif seed_type == 3:
        # Twinned Icosahedron approximation (break symmetry slightly)
        init_params = np.array(ideal_radii[:5])
        # Add some noise to break symmetry
        rng = np.random.default_rng(worker_id)
        init_params = init_params + rng.normal(0, 0.1, size=5)
    else:
        # Randomized radii
        rng = np.random.default_rng(worker_id)
        init_params = np.array(ideal_radii[:5]) + rng.normal(0, 0.2, size=5)
    
    # Step 1: Geometric Optimization
    try:
        if _elapsed() < deadline - 10:
            opt_params, opt_e = _optimize_geometric_params(n, init_params, deadline - 10)
            pos = _build_mackay_cluster(n, opt_params)
            e = opt_e
        else:
            pos = _build_mackay_cluster(n, init_params)
            e = _energy(pos)
    except Exception:
        pos = _build_mackay_cluster(n, ideal_radii[:5])
        e = _energy(pos)
    
    if not np.all(np.isfinite(pos)):
        return np.inf, None
        
    best_e = e
    best_pos = pos.copy()
    
    # Step 2: Coordinate Relaxation
    if _elapsed() < deadline - 5:
        try:
            e_relaxed, pos_relaxed = _relax(pos, deadline - 5)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
        except Exception:
            pass
            
    # Step 3: Basin Hopping on the relaxed structure (short)
    if _elapsed() < deadline - 5:
        current_pos = best_pos.copy()
        current_e = best_e
        amp = 0.2
        rng = np.random.default_rng(1000 + worker_id)
        
        for _ in range(20): # Limited iterations
            if _elapsed() > deadline - 5 or _STOP.is_set():
                break
            noise = rng.normal(0, amp, size=current_pos.shape)
            perturbed = current_pos + noise
            if not np.all(np.isfinite(perturbed)):
                continue
            try:
                e_new, p_new = _relax(perturbed, deadline - 5)
                if e_new < current_e:
                    current_e = e_new
                    current_pos = p_new.copy()
                    if e_new < best_e:
                        best_e = e_new
                        best_pos = p_new.copy()
                    amp = min(amp * 1.1, 0.5)
                else:
                    amp = max(amp * 0.9, 0.05)
            except Exception:
                break
                
    return best_e, best_pos


def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    deadline = _elapsed() + max(10.0, _remaining() - 10.0)
    
    # Use 4 parallel workers with different geometric seeds
    n_workers = 4
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            f = executor.submit(_worker, i, n, deadline, i)
            futures.append(f)
            
        for f in futures:
            try:
                e, p = f.result(timeout=max(5.0, _remaining()))
                if p is not None and np.all(np.isfinite(p)):
                    results.append((e, p))
            except Exception:
                continue
                
    if not results:
        # Fallback: Simple icosahedron
        pos = _build_mackay_cluster(n, [0.0, 1.0, 1.0 + 1.0/((1+np.sqrt(5))/2), 1.0 + 2.0/((1+np.sqrt(5))/2), 1.0 + 3.0/((1+np.sqrt(5))/2)])
        try:
            _, pos = relax(pos, "dzugutov")
        except Exception:
            pass
        return pos
    
    # Return the best result
    results.sort(key=lambda x: x[0])
    best_e, best_pos = results[0]
    
    return best_pos
# EVOLVE-BLOCK-END
