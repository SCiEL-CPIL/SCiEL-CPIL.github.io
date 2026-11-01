"""Lennard-Jones 55-atom cluster global minimum search.
Radial manifold optimization with high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 110.0  # safety margin before 150s limit

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _get_icosahedral_directions():
    """Return 54 unit direction vectors for Mackay icosahedron shells.
    Order: 12 inner-shell vertices, 12 outer-shell vertices, 30 mid-edge directions."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    # Normalize each vertex to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find the 30 edges of the icosahedron
    # In a regular icosahedron, each vertex is connected to 5 others
    # Edge length in unit sphere: 2*sin(pi/5) ≈ 1.17557
    edge_threshold = 2.0 * np.sin(np.pi / 5.0) * 1.01
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < edge_threshold:
                edges.append((i, j))
    
    # Should have exactly 30 edges
    assert len(edges) == 30, f"Expected 30 edges, got {len(edges)}"
    
    # Build direction list:
    # Indices 0-11: inner shell (icosahedron vertices)
    # Indices 12-23: outer shell (icosahedron vertices)
    # Indices 24-53: mid-edge directions (normalized sum of edge endpoints)
    directions = np.zeros((54, 3))
    for i in range(12):
        directions[i] = verts[i]
    for i in range(12):
        directions[12 + i] = verts[i]
    for k, (i, j) in enumerate(edges):
        mid = verts[i] + verts[j]
        mid = mid / np.linalg.norm(mid)
        directions[24 + k] = mid
    
    return directions


def _lj_energy(pos):
    """Compute Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    iu = np.triu_indices(n, k=1)
    e = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])
    return e


def _lj_energy_grad(pos):
    """Compute Lennard-Jones energy and gradient analytically."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e30)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])
    
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2_ij = np.dot(rij, rij)
            if r2_ij < 1e-12:
                continue
            r2_ij = max(r2_ij, 1e-12)
            inv_r2_ij = 1.0 / r2_ij
            inv_r6_ij = inv_r2_ij ** 3
            inv_r12_ij = inv_r6_ij ** 2
            coeff = 4.0 * (-12.0 * inv_r12_ij + 6.0 * inv_r6_ij) * inv_r2_ij
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return energy, grad


def _radial_to_cartesian(radii, directions):
    """Convert 54 radial parameters + center (fixed at origin) to 55 Cartesian positions.
    radii: array of shape (54,)
    Returns: array of shape (55, 3)"""
    pos = np.zeros((55, 3))
    # Atom 0: center at origin
    # Atoms 1-12: inner shell
    for i in range(12):
        pos[1 + i] = radii[i] * directions[i]
    # Atoms 13-24: outer shell vertices
    for i in range(12):
        pos[13 + i] = radii[12 + i] * directions[12 + i]
    # Atoms 25-54: mid-edge
    for i in range(30):
        pos[25 + i] = radii[24 + i] * directions[24 + i]
    return pos


def _radial_energy(radii, directions):
    """Energy as function of radial parameters."""
    pos = _radial_to_cartesian(radii, directions)
    return _lj_energy(pos)


def _radial_energy_grad(radii, directions):
    """Energy and gradient w.r.t. radial parameters."""
    pos = _radial_to_cartesian(radii, directions)
    e, grad_pos = _lj_energy_grad(pos)
    # dE/dr_i = grad_pos[atom_i] . direction_i
    grad_r = np.zeros(54)
    for i in range(54):
        grad_r[i] = np.dot(grad_pos[1 + i], directions[i])
    return e, grad_r


def _optimize_radial_manifold(directions, start_radii, deadline):
    """Optimize the 54 radial parameters using L-BFGS-B."""
    def obj_and_grad(r):
        try:
            e, g = _radial_energy_grad(r, directions)
            return e, g
        except Exception:
            return 1e10, np.zeros(54)
    
    best_r = start_radii.copy()
    best_e = _radial_energy(start_radii, directions)
    
    # Stage 1: moderate tolerance
    try:
        res = minimize(obj_and_grad, start_radii, jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10,
                                'maxls': 50})
        if res.fun < best_e:
            best_e = res.fun
            best_r = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_r, best_e
    
    # Stage 2: tighter tolerance
    try:
        res2 = minimize(obj_and_grad, best_r, jac=True, method='L-BFGS-B',
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12,
                                 'maxls': 50})
        if res2.fun < best_e:
            best_e = res2.fun
            best_r = res2.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_r, best_e
    
    # Stage 3: very tight
    try:
        res3 = minimize(obj_and_grad, best_r, jac=True, method='L-BFGS-B',
                        options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14,
                                 'maxls': 50})
        if res3.fun < best_e:
            best_e = res3.fun
            best_r = res3.x.copy()
    except Exception:
        pass
    
    return best_r, best_e


def _polish_cartesian(pos, deadline):
    """Full 3D L-BFGS polish starting from a given position."""
    def obj_and_grad(flat):
        p = flat.reshape(N, 3)
        try:
            e, g = _lj_energy_grad(p)
            return e, g
        except Exception:
            return 1e10, np.zeros_like(flat)
    
    best_p = pos.copy()
    best_e = _lj_energy(pos)
    
    # Stage 1
    try:
        res = minimize(obj_and_grad, best_p.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10,
                                'maxls': 50})
        if res.fun < best_e:
            best_e = res.fun
            best_p = res.x.reshape(N, 3).copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_p, best_e
    
    # Stage 2
    try:
        res2 = minimize(obj_and_grad, best_p.flatten(), jac=True, method='L-BFGS-B',
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12,
                                 'maxls': 50})
        if res2.fun < best_e:
            best_e = res2.fun
            best_p = res2.x.reshape(N, 3).copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_p, best_e
    
    # Stage 3: high precision
    try:
        res3 = minimize(obj_and_grad, best_p.flatten(), jac=True, method='L-BFGS-B',
                        options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14,
                                 'maxls': 50})
        if res3.fun < best_e:
            best_e = res3.fun
            best_p = res3.x.reshape(N, 3).copy()
    except Exception:
        pass
    
    return best_p, best_e


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    directions = _get_icosahedral_directions()
    
    # Initial radii: approximate Mackay icosahedron
    # Inner shell ~1.12, outer shell ~1.12 * (sqrt(5+2*sqrt(5))/3) ≈ 2.13
    r_inner = 1.12
    r_outer = 1.12 * np.sqrt(5.0 + 2.0 * np.sqrt(5.0)) / 3.0  # ≈ 2.131
    
    start_radii = np.zeros(54)
    for i in range(12):
        start_radii[i] = r_inner       # inner shell
    for i in range(12):
        start_radii[12 + i] = r_outer  # outer shell vertices
    for i in range(30):
        start_radii[24 + i] = r_outer  # mid-edge (same radius as outer in Mackay)
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize on radial manifold
    try:
        opt_radii, opt_e = _optimize_radial_manifold(directions, start_radii, deadline)
        pos_from_radial = _radial_to_cartesian(opt_radii, directions)
        e_check = _lj_energy(pos_from_radial)
        if e_check < best_energy:
            best_energy = e_check
            best_pos = pos_from_radial.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pos is not None:
            return best_pos
        # Fallback: return initial Mackay
        pos0 = _radial_to_cartesian(start_radii, directions)
        return pos0
    
    # Step 2: Full 3D polish from the radial manifold result
    if best_pos is not None:
        try:
            polished_pos, polished_e = _polish_cartesian(best_pos, deadline)
            if polished_e < best_energy:
                best_energy = polished_e
                best_pos = polished_pos.copy()
        except Exception:
            pass
    
    # Step 3: Try a few perturbations of the radial parameters and re-polish
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            # Perturb radii slightly
            perturbed = opt_radii + rng.normal(0, 0.05, size=54)
            # Ensure positive radii
            perturbed = np.maximum(perturbed, 0.5)
            try:
                p_pos, p_e = _optimize_radial_manifold(directions, perturbed, deadline - 10)
                if p_e < best_energy:
                    pos_p = _radial_to_cartesian(p_pos, directions)
                    ep = _lj_energy(pos_p)
                    if ep < best_energy:
                        best_energy = ep
                        best_pos = pos_p.copy()
                    # Polish in 3D
                    if time.time() < deadline - 10:
                        polished2, e2 = _polish_cartesian(best_pos, deadline - 5)
                        if e2 < best_energy:
                            best_energy = e2
                            best_pos = polished2.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            final_pos, final_e = _polish_cartesian(best_pos, deadline - 1)
            if final_e < best_energy:
                best_energy = final_e
                best_pos = final_pos.copy()
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = _radial_to_cartesian(start_radii, directions)
    
    return best_pos
