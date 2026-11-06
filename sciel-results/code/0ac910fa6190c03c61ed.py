import numpy as np
import time
import sys

# Try to import chem_clusters, which is mentioned in the task description as available
try:
    import chem_clusters
    HAS_CC = True
except ImportError:
    HAS_CC = False

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    exp_val = np.exp(rho * (1.0 - dists))
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=2000):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0, 0])
                r = 1.0
            push = (min_d - r + 0.01) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, centered at origin."""
    pts = []
    a = spacing
    # Estimate max radius needed
    max_r = int(np.ceil(n ** (1/3))) + 5
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return np.zeros((n, 3))
    # Center points
    center = pts.mean(axis=0)
    pts -= center
    # Sort by distance from center to pick the most compact cluster
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points, centered at origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 5
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    if len(pts) == 0:
        return np.zeros((n, 3))
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, centered at origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 5
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    if len(pts) == 0:
        return np.zeros((n, 3))
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _mckay_icosahedron(n, spacing=1.0):
    """Generate Mackay icosahedron cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v.copy())
    
    # Add points for larger n by scaling up
    if n > 13:
        # Simple scaling strategy for Mackay-like
        scale = 2.0
        verts2 = verts * scale
        for v in verts2:
            pts.append(v.copy())
        
    # Fill with random points if still not enough, but prefer symmetry
    while len(pts) < n:
        # Add points on the surface of the convex hull or random nearby
        base_pt = pts[np.random.randint(len(pts))]
        p = base_pt + np.random.normal(0, 0.2, 3)
        pts.append(p)
        
    pts = np.array(pts[:n])
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            # Normalize to have min distance ~1 (equilibrium)
            pts = pts * (1.0 / min_d)
            
    return pts

def _random_dense(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.8
    min_d = 0.55 * spacing
    max_attempts = 5000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
        
    return np.array(pts)

def _relax_with_cc(pos, rho, deadline):
    """Relax using chem_clusters if available, otherwise L-BFGS-B."""
    if HAS_CC:
        try:
            # chem_clusters.relax usually takes positions and returns relaxed positions
            # It might have a specific signature. Let's assume standard relax(pos, potential, rho)
            # Or it might return (pos, energy). We need to be careful.
            # Based on task description: chem_clusters.relax(positions, 'morse', rho)
            relaxed = chem_clusters.relax(pos, 'morse', rho)
            # Check if it returns just positions or a tuple
            if isinstance(relaxed, tuple):
                rel_pos, rel_energy = relaxed
            else:
                rel_pos = relaxed
                rel_energy = _morse_energy(rel_pos, rho)
            
            if _min_dist_check(rel_pos):
                return rel_pos, rel_energy
        except Exception as e:
            pass # Fallback to manual relaxation
            
    # Fallback: L-BFGS-B from scipy
    from scipy.optimize import minimize
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        # Use a shorter iteration count if deadline is close
        time_left = deadline - time.time()
        if time_left < 5:
            maxiter = 100
        elif time_left < 20:
            maxiter = 500
        else:
            maxiter = 2000
            
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 100})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass

    return pos, _morse_energy(pos, rho)

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    n = instance["n"]
    rho = instance.get("rho", 10)
    
    start_time = time.time()
    # Total budget 300s for 6 instances. We get approx 50s per instance.
    # We should use a bit less to be safe, say 45s per instance.
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(n * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Select initial guesses based on rho
    # rho=10: Icosahedral
    # rho=6: FCC/HCP
    # rho=3: FCC
    
    if rho >= 10:
        # Icosahedral structures
        initial_configs = [
            _mckay_icosahedron(n),
            _mckay_icosahedron(n) + rng.normal(0, 0.1, (n, 3)),
            _fcc_points(n), # Fallback
            _random_dense(n, rng)
        ]
    elif rho >= 6:
        # FCC/HCP structures
        initial_configs = [
            _fcc_points(n),
            _hcp_points(n),
            _bcc_points(n),
            _fcc_points(n) + rng.normal(0, 0.1, (n, 3)),
            _random_dense(n, rng)
        ]
    else:
        # Long range, FCC dominant
        initial_configs = [
            _fcc_points(n),
            _hcp_points(n),
            _fcc_points(n) + rng.normal(0, 0.1, (n, 3)),
            _random_dense(n, rng)
        ]
    
    # Try each initial configuration
    for pos in initial_configs:
        if time.time() > deadline - 2.0:
            break
            
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
            
        try:
            relaxed_pos, energy = _relax_with_cc(pos, rho, deadline)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
            
    # If we have a good candidate, try to improve it further with kicks
    if best_pos is not None:
        # Remaining time for kicks
        while time.time() < deadline - 5.0:
            # Reduce kick size as we get closer to deadline
            time_left = deadline - time.time()
            kick_size = 0.1 * (time_left / 40.0) + 0.05
            kicked_pos = best_pos + rng.normal(0, kick_size, best_pos.shape)
            kicked_pos = _ensure_min_dist(kicked_pos)
            if not _min_dist_check(kicked_pos):
                continue
                
            try:
                relaxed_pos, energy = _relax_with_cc(kicked_pos, rho, deadline)
                if _min_dist_check(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                continue
            
    # Final polish if time permits
    if best_pos is not None and time.time() < deadline - 2.0:
        try:
            relaxed_pos, energy = _relax_with_cc(best_pos, rho, deadline)
            if _min_dist_check(relaxed_pos) and energy < best_energy:
                best_pos = relaxed_pos
                best_energy = energy
        except Exception:
            pass

    if best_pos is None:
        # Emergency fallback
        pos = _random_dense(n, rng)
        pos = _ensure_min_dist(pos)
        best_pos, best_energy = _relax_with_cc(pos, rho, deadline)
        
    return best_pos
