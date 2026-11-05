"""Dzugutov cluster global minima search using continuous annealing on softness parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

# Global time tracker
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

def _dzugutov_energy(positions, c_val=None, d_val=None):
    """Compute Dzugutov potential energy with optional softness parameter override."""
    if c_val is None:
        c_val = c_true
    if d_val is None:
        d_val = d_true
        
    n = len(positions)
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        # Prevent overflow/underflow issues in exp
        arg = c_val / (r1 - a)
        # Clip to avoid extreme values
        arg = np.clip(arg, -500, 500)
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(arg))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        arg = d_val / (r2_ - b)
        arg = np.clip(arg, -500, 500)
        energy += np.sum(B * np.exp(arg))
    
    return float(energy)

def _dzugutov_gradient(positions, c_val=None, d_val=None):
    """Compute gradient of Dzugutov potential energy with optional softness parameter override."""
    if c_val is None:
        c_val = c_true
    if d_val is None:
        d_val = d_true
        
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
    
    i_idx = iu[0][valid]
    j_idx = iu[1][valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[i_idx, j_idx]
    
    dV_dr = np.zeros_like(r)
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        # r1 - a is negative. c_val / (r1-a) is large negative if r1 is close to a from below?
        # No, r1 < a => r1 - a < 0. c_val > 0. So arg is negative.
        # exp(arg) is small.
        # However, r1 ** (-m) is huge if r1 is small.
        arg = c_val / (r1 - a)
        arg = np.clip(arg, -500, 500)
        exp_term = np.exp(arg)
        
        # Term 1: A * (-m * r1^(-m-1)) * exp
        # Term 2: A * (r1^(-m) - B) * exp * (-c / (r1-a)^2)
        
        # Check for numerical stability
        # r1^(-17) can be huge.
        # If r1 is very small, energy is huge.
        # Let's just compute and let numpy handle it, clipping exp is good.
        
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_val / (r1 - a) ** 2)
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        arg = d_val / (r2_ - b)
        arg = np.clip(arg, -500, 500)
        exp_term = np.exp(arg)
        dV_dr[mask2] = B * exp_term * (-d_val / (r2_ - b) ** 2)
    
    factor = -dV_dr / r
    # Clip factor to avoid huge forces causing divergence
    factor = np.clip(factor, -1e6, 1e6)
    forces = factor[:, np.newaxis] * rij
    
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    return grad

def _get_icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _get_icosahedron_edges(verts):
    """Get the 30 edge midpoint directions of an icosahedron given its vertices."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    
    edge_dirs = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_dirs.append(mid)
    
    return np.array(edge_dirs)

def _get_icosahedron_face_centers(verts):
    """Get the 20 face center directions of an icosahedron given its 12 vertices."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    
    adj = {i: [] for i in range(n_verts)}
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                adj[i].append(j)
                adj[j].append(i)
    
    face_dirs = []
    seen = set()
    for i in range(n_verts):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j or k == i:
                    continue
                if k in adj[i]:
                    face_key = tuple(sorted([i, j, k]))
                    if face_key not in seen:
                        seen.add(face_key)
                        center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_dirs.append(center)
    
    return np.array(face_dirs)

def _build_mackay_55(r1, r2v, r2e):
    """
    Build Mackay 55 icosahedron.
    """
    points = [np.zeros(3)]
    
    v1 = _get_icosahedron_vertices(r1)
    points.extend([v for v in v1])
    
    v1_unit = v1 / r1
    v2 = v1_unit * r2v
    points.extend([v for v in v2])
    
    e_dirs = _get_icosahedron_edges(_get_icosahedron_vertices(1.0))
    e2 = e_dirs * r2e
    points.extend([e for e in e2])
    
    return np.array(points)

def _build_mackay_55_with_faces(r1, r2v, r2e):
    """
    Build Mackay 55 icosahedron and return face center directions.
    """
    pos = _build_mackay_55(r1, r2v, r2e)
    v1 = _get_icosahedron_vertices(1.0)
    face_dirs = _get_icosahedron_face_centers(v1)
    return pos, face_dirs

def _check_min_distance(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    if len(pos) < 2:
        return True
    tree = cKDTree(pos)
    pairs = tree.query_pairs(min_dist)
    return len(pairs) == 0

def _relax_with_annealing(pos, n_atoms, c_start=5.0, d_start=2.0, n_steps=40):
    """
    Relax structure using continuous annealing on softness parameters c and d.
    Starts with large c, d (smooth landscape) and gradually reduces to true values.
    """
    current_pos = pos.copy()
    
    for step in range(n_steps):
        if _time_left() < 5:
            break
            
        # Interpolate c and d from start to true values
        # Use a smooth schedule
        alpha = step / max(1, n_steps - 1)
        # Ease-in-out for better convergence
        alpha_smooth = alpha * alpha * (3 - 2 * alpha)
        
        c_val = c_start + (c_true - c_start) * alpha_smooth
        d_val = d_start + (d_true - d_start) * alpha_smooth
        
        # Use L-BFGS-B with the current softness parameters
        x0 = current_pos.flatten()
        
        def fun(x):
            p = x.reshape(n_atoms, 3)
            return _dzugutov_energy(p, c_val, d_val)
            
        def jac(x):
            p = x.reshape(n_atoms, 3)
            return _dzugutov_gradient(p, c_val, d_val).flatten()

        try:
            # Reduce maxiter as we approach final parameters to save time
            max_iter = 50 if alpha_smooth < 0.5 else 100
            res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            current_pos = res.x.reshape(n_atoms, 3)
        except Exception:
            break
    
    # Final polish with true parameters
    x0 = current_pos.flatten()
    def fun_final(x):
        p = x.reshape(n_atoms, 3)
        return _dzugutov_energy(p)
        
    def jac_final(x):
        p = x.reshape(n_atoms, 3)
        return _dzugutov_gradient(p).flatten()

    try:
        res = minimize(fun_final, x0, jac=jac_final, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x.reshape(n_atoms, 3), res.fun
    except Exception:
        return current_pos, _dzugutov_energy(current_pos)

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    
    if _time_left() < 5:
        pos = np.random.randn(n, 3) * 0.5
        return pos

    best_energy = float('inf')
    best_pos = None
    
    # Diverse initial guesses
    candidates = []
    
    # 1. Mackay icosahedron variants
    radii_sets = [
        (1.5, 2.5, 2.5),
        (1.6, 2.6, 2.6),
        (1.7, 2.8, 2.8),
        (1.5, 2.4, 2.4),
        (1.8, 3.0, 3.0),
        (1.55, 2.55, 2.55),
        (1.65, 2.65, 2.65),
        (1.45, 2.45, 2.45),
        (1.75, 2.75, 2.75),
    ]
    
    for r1, r2v, r2e in radii_sets:
        if _time_left() < 15:
            break
        m55 = _build_mackay_55(r1, r2v, r2e)
        
        if n == 55:
            candidates.append(m55)
        else:
            # Create subset by removing outermost atoms
            com = np.mean(m55, axis=0)
            dists = np.linalg.norm(m55 - com, axis=1)
            idx_sorted = np.argsort(dists)[::-1]
            remove_idx = idx_sorted[:55-n]
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[remove_idx] = False
            pos_subset = m55[keep_mask]
            candidates.append(pos_subset)
            
            # Also try adding atoms for n > 55
            if n > 55:
                extra_n = n - 55
                # Use face centers for placement
                v1 = _get_icosahedron_vertices(1.0)
                face_dirs = _get_icosahedron_face_centers(v1)
                
                # Scale face dirs to be outside the Mackay cluster
                max_r = np.max(np.linalg.norm(m55, axis=1))
                extra_pos = []
                for i in range(extra_n):
                    direction = face_dirs[i % len(face_dirs)]
                    radius = max_r + 0.5 + 0.2 * (i // len(face_dirs))
                    extra_pos.append(direction * radius)
                
                if extra_pos:
                    pos_n = np.vstack([m55, np.array(extra_pos)])
                    candidates.append(pos_n)

    # 2. Random spherical clusters
    for _ in range(3):
        if _time_left() < 10:
            break
        # Generate points in a sphere
        pos = np.random.randn(n, 3)
        norms = np.linalg.norm(pos, axis=1)
        # Scale to fit within a reasonable radius
        target_r = 3.0
        pos = pos / norms[:, np.newaxis] * (target_r * np.cbrt(np.random.rand(n)))
        candidates.append(pos)
        
    # 3. Simple cubic lattice
    if _time_left() > 20:
        side = int(np.ceil(n ** (1/3)))
        lattice = np.array([
            [i, j, k] for i in range(side) for j in range(side) for k in range(side)
        ])
        if len(lattice) >= n:
            pos = lattice[:n] * 0.8
            candidates.append(pos)
            
    # Relax candidates
    for pos in candidates:
        if _time_left() < 10:
            break
        
        # Check minimum distance constraint
        if not _check_min_distance(pos, 0.5):
            continue
        
        try:
            pos_rel, e_rel = _relax_with_annealing(pos, n, c_start=5.0, d_start=2.0, n_steps=30)
            if e_rel < best_energy:
                best_energy = e_rel
                best_pos = pos_rel
        except Exception:
            continue

    if best_pos is None and candidates:
        best_pos = candidates[0]
        best_energy = _dzugutov_energy(best_pos)
    elif best_pos is None:
        best_pos = np.random.randn(n, 3) * 0.5

    return best_pos

# EVOLVE-BLOCK-END
