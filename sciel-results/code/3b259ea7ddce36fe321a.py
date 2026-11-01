"""Lennard-Jones 55-atom cluster global minimum search."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 55
DEADLINE_MARGIN = 10.0  # seconds to spare before 150s limit

def lj_energy(pos):
    """Compute Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, np.inf)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Sum over upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    e = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])
    return e

def lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1e30)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Energy: 4*sum(r^-12 - r^-6) over pairs (i<j)
    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])
    # Gradient: dE/dpos_i = sum_j 4*(-12*r^-14 + 6*r^-8)*(pos_i - pos_j)/r^2
    # More precisely: E_pair = 4*(r^-12 - r^-6), dE_pair/dpos_i = 4*(-12*r^-14 + 6*r^-8)*(pos_i - pos_j)/r^2 * (-1) ... 
    # Let's use: dE/dx_i = sum_{j!=i} 4*(-12*r_ij^(-14) + 6*r_ij^(-8)) * (x_i - x_j) / r_ij^2 * (-1)
    # Actually: E = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dx_i = dE/dr * dr/dx_i = 4*(-12*r^-13 + 6*r^-7) * (x_i - x_j)/r
    # = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2_ij = np.dot(rij, rij)
            if r2_ij < 1e-12:
                continue
            r2_ij = max(r2_ij, 1e-12)
            inv_r2 = 1.0 / r2_ij
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            # dE/dpos_i contribution from pair (i,j)
            coeff = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6) * inv_r2
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return energy, grad

def mackay_icosahedron():
    """Generate Mackay icosahedron coordinates for 55 atoms."""
    # Icosahedron vertices (12)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # normalize to unit radius
    
    # Inner shell: 12 vertices of icosahedron at radius r1
    # Outer shell: 12 at r2, plus 30 mid-edge vertices at r3
    # For 55 atoms: 1 center + 12 inner + 12 outer + 30 mid-edge
    # Standard Mackay icosahedron has shells at radii ratios:
    # r1=1, r2=sqrt(5+2*sqrt(5))/3 * r1... let's use known ratios
    # Inner shell radius: 1.0 (relative)
    # Outer shell radius: sqrt(5 + 2*sqrt(5)) / 3 ≈ 1.9021
    # Mid-edge radius: same as outer shell for Mackay icosahedron
    
    r_inner = 1.0
    r_outer = np.sqrt(5.0 + 2.0 * np.sqrt(5.0)) / 3.0  # ≈ 1.90211
    
    points = []
    # Center
    points.append(np.zeros(3))
    # Inner shell: 12 icosahedron vertices
    for v in ico_verts:
        points.append(r_inner * v)
    # Outer shell: 12 icosahedron vertices scaled
    for v in ico_verts:
        points.append(r_outer * v)
    # Mid-edge vertices: 30 edges of icosahedron, midpoint of each edge at outer radius
    # Icosahedron edges: pairs of vertices that are nearest neighbors
    # In a regular icosahedron, each vertex connects to 5 others
    # The 30 edges can be found by checking which pairs of the 12 vertices are closest
    verts_norm = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    dist_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            dist_matrix[i, j] = np.linalg.norm(verts_norm[i] - verts_norm[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    
    # Find edges: pairs with minimum distance
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    edge_threshold = min_dist * 1.1  # small tolerance
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dist_matrix[i, j] < edge_threshold:
                edges.append((i, j))
    
    # There should be 30 edges
    for (i, j) in edges:
        mid = (verts_norm[i] + verts_norm[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * r_outer
        points.append(mid)
    
    # Scale to match LJ equilibrium: inner shell at ~1.12, outer at ~1.12*r_outer/r_inner
    # Actually, let's just scale the whole thing so the inner shell is at 1.12
    scale = 1.12 / r_inner
    points = np.array(points) * scale
    
    # We should have 55 points
    assert len(points) == 55, f"Expected 55 points, got {len(points)}"
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_pos = None
    best_energy = np.inf
    
    # Start with Mackay icosahedron
    try:
        mackay_pos = mackay_icosahedron()
        e = lj_energy(mackay_pos)
        if e < best_energy:
            best_energy = e
            best_pos = mackay_pos.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos if best_pos is not None else np.zeros((N, 3))
    
    # Local relaxation with L-BFGS
    def obj_and_grad(pos_flat):
        pos = pos_flat.reshape(N, 3)
        try:
            e, g = lj_energy_and_grad(pos)
            return e, g
        except Exception:
            return 1e10, np.zeros_like(pos_flat)
    
    if best_pos is not None:
        try:
            pos_flat = best_pos.flatten()
            res = minimize(obj_and_grad, pos_flat, jac=True, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.success or res.fun < best_energy:
                e_res = lj_energy(res.x.reshape(N, 3))
                if e_res < best_energy:
                    best_energy = e_res
                    best_pos = res.x.reshape(N, 3).copy()
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Try differential evolution for global search
    # Use a reasonable bounds: atoms within a box
    bounds = [(-4.0, 4.0)] * (N * 3)
    
    def de_obj(pos_flat):
        pos = pos_flat.reshape(N, 3)
        try:
            return lj_energy(pos)
        except Exception:
            return 1e10
    
    try:
        # Use a subset of time for DE
        de_deadline = min(deadline, time.time() + 60.0)
        
        # Use population size that fits in time
        pop_size = 40
        max_iter = 50
        
        def de_callback(xk, convergence):
            nonlocal best_energy, best_pos
            pos = xk.reshape(N, 3)
            try:
                e = lj_energy(pos)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
            except Exception:
                pass
            return time.time() > de_deadline
        
        res_de = differential_evolution(de_obj, bounds, maxiter=max_iter, popsize=pop_size,
                                        tol=1e-8, seed=42, callback=de_callback,
                                        polish=False, init='sobol', mutation=(0.3, 0.8),
                                        recombination=0.7, updating='immediate')
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Final polish with L-BFGS
    if best_pos is not None:
        try:
            pos_flat = best_pos.flatten()
            res2 = minimize(obj_and_grad, pos_flat, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14})
            if res2.fun < best_energy:
                e_res = lj_energy(res2.x.reshape(N, 3))
                if e_res < best_energy:
                    best_energy = e_res
                    best_pos = res2.x.reshape(N, 3).copy()
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
