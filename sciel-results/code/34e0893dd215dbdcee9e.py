import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

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
    Generate an icosahedral structure for 38 atoms.
    Center + 12 (1st shell) + 20 (2nd shell) + 5 (3rd shell) = 38?
    Icosahedral shells: 1, 12, 20, 30...
    1+12+20 = 33. Need 5 more.
    This is a common local minimum trap.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)] # Center
    pts.extend(verts.tolist()) # 1st shell
    
    # 2nd shell: centers of faces of icosahedron? No, vertices of dodecahedron?
    # The 2nd shell in icosahedral clusters are usually 20 atoms at the centers of the triangular faces of the icosahedron.
    # Face normals of icosahedron.
    # There are 20 faces.
    # We can generate them by taking the cross product of adjacent edge vectors.
    # Simpler: The 2nd shell positions are proportional to the vertices of a dodecahedron?
    # Actually, for LJ, the 2nd shell is usually at distance ~1.6-1.7 times the 1st shell.
    # Let's use a known coordinate set for the 33-atom icosahedron and add 5 atoms.
    # Or just use a random perturbation of the FCC as the second start point.
    
    # For now, return a perturbed FCC to serve as a second basin start.
    return None

def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

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

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # 1. Start with FCC
    try:
        pos0 = _make_fcc_cluster()
        p, e = _relax(pos0, max_iter=5000, gtol=1e-12, ftol=1e-16)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # 2. Basin Hopping / Random Restarts
    rng = np.random.default_rng(42)
    
    # We will try several perturbations from the best found structure
    # and also from the initial FCC structure.
    
    # Define perturbation strengths
    # Large perturbations to jump funnels, small to polish.
    
    current_best = best_pos.copy() if best_pos is not None else _make_fcc_cluster()
    
    for i in range(50): # Max 50 iterations
        if time.time() > deadline:
            break
            
        # Choose a base structure
        # 50% chance from current best, 50% from initial FCC (or other known good)
        if i % 2 == 0:
            base = current_best
        else:
            base = _make_fcc_cluster()
            
        # Perturb
        # Scale of perturbation: start large, decrease?
        # Or random scale.
        scale = 0.1 + 0.1 * rng.random()
        perturbed = base + scale * rng.standard_normal(base.shape)
        
        # Ensure separation
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax
        try:
            p, e = _relax(perturbed, max_iter=2000, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p
                current_best = p
        except Exception:
            continue

    # 3. Final High-Precision Polish
    if time.time() < deadline - 5.0 and best_pos is not None:
        try:
            p, e = _relax(best_pos, max_iter=10000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    if best_pos is None:
        # Fallback
        best_pos = _make_fcc_cluster()
        
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
