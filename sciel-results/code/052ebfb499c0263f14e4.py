import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin within 150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    # Generate FCC points
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _make_icosahedral_cluster():
    """
    Generate a 38-atom icosahedral cluster.
    1 center + 12 (1st shell) + 20 (2nd shell) + 5 (3rd shell) = 38
    """
    # 1 center
    pts = [[0,0,0]]
    
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico_coords = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    pts.extend(ico_coords)
    
    # 20 vertices of second shell (face centers of icosahedron)
    # The 20 face centers of a regular icosahedron correspond to the vertices of a dodecahedron
    # scaled appropriately. The dodecahedron vertices can be generated from the icosahedron.
    # Actually, the dual of the icosahedron is the dodecahedron.
    # The face centers of the icosahedron are the vertices of the dodecahedron.
    # Let's compute them by averaging the 3 vertices of each face.
    
    # First, define the faces of the icosahedron.
    # The icosahedron has 20 faces. We need to find them.
    # A regular icosahedron has 20 triangular faces.
    # Let's use a known list of faces for the standard icosahedron.
    
    # The standard icosahedron vertices (indices 0-11) are:
    # 0: [0, 1, phi]
    # 1: [0, -1, phi]
    # 2: [0, 1, -phi]
    # 3: [0, -1, -phi]
    # 4: [1, phi, 0]
    # 5: [-1, phi, 0]
    # 6: [1, -phi, 0]
    # 7: [-1, -phi, 0]
    # 8: [phi, 0, 1]
    # 9: [-phi, 0, 1]
    # 10: [phi, 0, -1]
    # 11: [-phi, 0, -1]
    
    # Faces (triangles) of the icosahedron:
    # This is a standard list.
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 9], [0, 9, 5], [0, 5, 1],
        [1, 5, 7], [1, 7, 6], [1, 6, 4],
        [4, 6, 10], [4, 10, 8],
        [8, 10, 11], [8, 11, 9],
        [9, 11, 7], [9, 7, 5],
        [5, 7, 3], [5, 3, 2], [5, 2, 0],
        [2, 3, 11], [2, 11, 8],
        [3, 7, 11]
    ]
    
    ico_verts = np.array(ico_coords, dtype=np.float64)
    
    # Compute face centers (dodecahedron vertices)
    dodeca_coords = []
    for face in faces:
        center = np.mean(ico_verts[face], axis=0)
        dodeca_coords.append(center)
    
    dodeca_coords = np.array(dodeca_coords, dtype=np.float64)
    
    # Scale dodecahedron to appropriate radius
    # The icosahedron vertices have radius sqrt(1 + phi^2)
    ico_radius = np.linalg.norm(ico_verts[0])
    dodeca_radius = np.linalg.norm(dodeca_coords[0])
    
    # The second shell should be at a distance that makes sense for LJ.
    # In an icosahedral cluster, the second shell is typically at a distance
    # that is a multiple of the first shell distance.
    # Let's just add them and scale later.
    pts.extend([list(c) for c in dodeca_coords])
    
    # Now we have 1 + 12 + 20 = 33 atoms. We need 5 more.
    # Add 5 atoms in a cap or ring.
    # Let's add 5 atoms in a ring around the top.
    # The top vertex is [0, 1, phi].
    # Let's add 5 atoms in a circle around the z-axis at a height between 1 and phi.
    height = (1 + phi) / 2
    radius = np.sqrt(phi**2 - height**2)  # Approximate
    if radius < 0:
        radius = 0.5
    
    for k in range(5):
        angle = 2 * np.pi * k / 5
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        pts.append([x, y, z])
    
    pts = np.array(pts, dtype=np.float64)
    pts = pts[:38]
    pts -= np.mean(pts, axis=0)
    
    # Scale to optimal nearest neighbor distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        pts *= (target_r / min_r)
        
    return pts

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, n_iter=30):
    """
    1D global search over uniform scale factor s applied to centered positions.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def _make_random_shell_cluster(rng):
    """
    Generate a random cluster with shell-like structure.
    """
    pts = []
    # Center
    pts.append([0, 0, 0])
    
    # Shell 1: 12 atoms
    n_shell1 = 12
    for _ in range(n_shell1):
        # Random direction on sphere
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = 1.0
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        pts.append([x * r, y * r, z * r])
    
    # Shell 2: 20 atoms
    n_shell2 = 20
    r2 = 1.8
    for _ in range(n_shell2):
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        pts.append([x * r2, y * r2, z * r2])
    
    # Shell 3: 5 atoms
    n_shell3 = 5
    r3 = 2.5
    for _ in range(n_shell3):
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        pts.append([x * r3, y * r3, z * r3])
    
    pts = np.array(pts, dtype=np.float64)
    pts = pts[:38]
    pts -= np.mean(pts, axis=0)
    
    # Scale to optimal nearest neighbor distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        pts *= (target_r / min_r)
        
    return pts

def construct():
    t0 = time.time()
    
    # 1. Generate initial geometries
    starts = []
    try:
        starts.append(_make_fcc_cluster())
    except Exception:
        pass
    
    try:
        starts.append(_make_icosahedral_cluster())
    except Exception:
        pass
    
    # Add random shell clusters
    rng = np.random.default_rng(42)
    for seed in [42, 123, 456, 789, 101]:
        try:
            r = np.random.default_rng(seed)
            starts.append(_make_random_shell_cluster(r))
        except Exception:
            pass
    
    # Add a random start
    rand_pos = rng.normal(0, 1.5, size=(N, 3))
    rand_pos -= np.mean(rand_pos, axis=0)
    starts.append(rand_pos)
    
    best_pos = None
    best_energy = np.inf

    # 2. Initial relaxation of all starts
    for pos0 in starts:
        if time.time() - t0 > DEADLINE - 20.0:
            break
        try:
            p, e = _relax(pos0, max_iter=500, gtol=1e-8)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 3. 1D scale optimization on the best
    if best_pos is not None and time.time() - t0 < DEADLINE - 15.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # 4. Basin Hopping using scipy.optimize.basin_hopping
    if best_pos is not None:
        def bh_energy(x):
            p = x.reshape(-1, 3)
            e, _ = _lj_energy_and_grad(p)
            return e
            
        def bh_minimizer(x):
            p = x.reshape(-1, 3)
            p_relaxed, e_relaxed = _relax(p, max_iter=200, gtol=1e-6)
            return {'x': p_relaxed.ravel(), 'fun': e_relaxed}

        # Run basin hopping
        try:
            res_bh = basin_hopping(
                bh_energy,
                best_pos.ravel(),
                minimizer=bh_minimizer,
                niter=100,
                stepsize=0.5,
                seed=42
            )
            if res_bh.fun < best_energy:
                best_energy = res_bh.fun
                best_pos = res_bh.x.reshape(-1, 3)
        except Exception:
            pass

    # 5. Final High-Precision Polish
    if best_pos is not None and time.time() - t0 < DEADLINE - 2.0:
        try:
            # Scale optimization
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            
            # Tight relaxation
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 6. Final separation check
    if best_pos is not None:
        best_pos = _ensure_min_sep(best_pos)
    else:
        # Fallback
        best_pos = _make_fcc_cluster()

    return best_pos
