"""Dzugutov cluster global minima search using polytetrahedral initialization and parameter annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Dzugutov potential parameters (physical)
A_phys = 5.82
B_phys = 1.28
a_phys = 1.87
b_phys = 1.94
c_phys = 1.10
d_phys = 0.27
m_phys = 16.0

# Global time tracking
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

def _dzugutov_energy(pos, c_val, d_val):
    """Compute Dzugutov potential energy with annealed c and d parameters."""
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Vectorized distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    # Avoid division by zero or negative sqrt
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
        
    r = np.sqrt(r2_pair[valid])
    
    energy = 0.0
    
    # Region 1: r < a
    mask1 = r < a_phys
    if np.any(mask1):
        r1 = r[mask1]
        # V(r) = A(r^-m - B)exp(c/(r - a))
        # c_val is the annealed parameter
        term = A_phys * (r1 ** (-m_phys) - B_phys) * np.exp(c_val / (r1 - a_phys))
        energy += np.sum(term)
        
    # Region 2: a <= r < b
    mask2 = (r >= a_phys) & (r < b_phys)
    if np.any(mask2):
        r2_ = r[mask2]
        # V(r) = B exp(d/(r - b))
        # d_val is the annealed parameter
        term = B_phys * np.exp(d_val / (r2_ - b_phys))
        energy += np.sum(term)
        
    return float(energy)

def _dzugutov_gradient(pos, c_val, d_val):
    """Compute gradient of Dzugutov potential energy with annealed c and d parameters."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
        
    i_idx = iu[0][valid]
    j_idx = iu[1][valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[i_idx, j_idx] # pos[i] - pos[j]
    
    dV_dr = np.zeros_like(r)
    
    # Region 1: r < a
    mask1 = r < a_phys
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_val / (r1 - a_phys))
        # d/dr [ A(r^-m - B) e^{c/(r-a)} ]
        # = A [ -m r^{-m-1} e^{...} + (r^{-m}-B) e^{...} (-c/(r-a)^2) ]
        dV_dr[mask1] = A_phys * ( -m_phys * r1 ** (-m_phys - 1) * exp_term 
                                  + (r1 ** (-m_phys) - B_phys) * exp_term * (-c_val / (r1 - a_phys) ** 2) )
        
    # Region 2: a <= r < b
    mask2 = (r >= a_phys) & (r < b_phys)
    if np.any(mask2):
        r2_ = r[mask2]
        exp_term = np.exp(d_val / (r2_ - b_phys))
        # d/dr [ B e^{d/(r-b)} ] = B e^{...} (-d/(r-b)^2)
        dV_dr[mask2] = B_phys * exp_term * (-d_val / (r2_ - b_phys) ** 2)
        
    # Force on i from j: -dV/dr * (rij / r)
    # We need to be careful with signs. 
    # F_ij = - dV/dr * (r_i - r_j) / r
    factor = -dV_dr / r
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

def _get_icosahedron_topology():
    """Get faces and edges of unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    # Edge length of unit icosahedron
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(dist - edge_len) < 1e-6:
                edges.append((i, j))
                
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (tuple(sorted((i, j))) in edge_set and
                    tuple(sorted((j, k))) in edge_set and
                    tuple(sorted((i, k))) in edge_set):
                    faces.append((i, j, k))
                    
    return verts_unit, faces, edges

def _build_initial_guess(n, rng):
    """Build a Mackay-like icosahedral initial guess."""
    # Optimal bond lengths tuned for Dzugutov
    r1 = 1.05
    r2 = 1.85
    
    points = [np.zeros(3)]
    
    # First shell: icosahedron vertices
    ico = _get_icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= len(points):
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
        
    # Second shell candidates
    verts_unit, faces, edges = _get_icosahedron_topology()
    
    # Face centers (20 positions)
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
        
    # Edge midpoints (30 positions)
    edge_mids = []
    for i, j in edges:
        em = (verts_unit[i] + verts_unit[j]) / 2
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
        
    # Build second shell: prioritize face centers, then edge midpoints
    second_shell = []
    for fc in face_centers:
        second_shell.append(fc * r2)
    for em in edge_mids:
        second_shell.append(em * r2)
        
    second_shell = np.array(second_shell)
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
        
    # Select from second shell candidates
    n_select = min(needed, len(second_shell))
    idx = rng.permutation(len(second_shell))
    for i in range(n_select):
        points.append(second_shell[idx[i]])
        
    # If we still need more atoms, add random points in the shell region
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.15 * rng.standard_normal(3))
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
            
    return np.array(points[:n])

def _anneal_and_relax(pos0, n, c_start, d_start, steps=100, max_iter_per_step=50):
    """
    Perform annealing on c and d parameters, relaxing structure at each step.
    Returns the final relaxed position at physical parameters.
    """
    pos = pos0.copy()
    c_end = c_phys
    d_end = d_phys
    
    # Linear annealing schedule
    # We might want to anneal c and d differently, but linear is a good start
    # c goes from c_start to c_phys
    # d goes from d_start to d_phys
    
    for step in range(steps):
        # Check time
        if _time_left() < 5.0:
            break
            
        alpha = step / float(steps)
        c_val = c_start + (c_end - c_start) * alpha
        d_val = d_start + (d_end - d_start) * alpha
        
        # Define objective function for this step
        def fun(x, c_val=c_val, d_val=d_val):
            p = x.reshape(n, 3)
            return _dzugutov_energy(p, c_val, d_val)
            
        def jac(x, c_val=c_val, d_val=d_val):
            p = x.reshape(n, 3)
            return _dzugutov_gradient(p, c_val, d_val).flatten()
            
        try:
            res = minimize(fun, pos.flatten(), jac=jac, method='L-BFGS-B', 
                           options={'maxiter': max_iter_per_step, 'ftol': 1e-8, 'gtol': 1e-6})
            pos = res.x.reshape(n, 3)
        except Exception:
            # If optimization fails, keep current position
            pass
            
    return pos

def construct(instance):
    global _start_time
    _start_time = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size * 7 + 13)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy: Generate a few diverse initial guesses and anneal them
    # We will use 3-5 restarts depending on time
    n_restarts = 5 if size < 50 else 3
    
    for i in range(n_restarts):
        if _time_left() < 10.0:
            break
            
        # Generate initial guess
        pos0 = _build_initial_guess(size, rng)
        
        # Add small noise to break symmetry if i > 0
        if i > 0:
            pos0 = pos0 + 0.05 * rng.standard_normal((size, 3))
            
        # Annealing parameters
        # Start with larger c and d to smooth the landscape
        # c_phys = 1.1, d_phys = 0.27
        # Let's start with c=3.0, d=1.0 and anneal down
        c_start = 3.0
        d_start = 1.0
        
        # Number of annealing steps
        # More steps for larger clusters to allow smoother transitions
        n_steps = 50 if size < 50 else 30
        
        # Relaxation iterations per step
        # Keep low to allow many steps within time budget
        max_iter = 30
        
        pos_annealed = _anneal_and_relax(pos0, size, c_start, d_start, n_steps, max_iter)
        
        # Final relaxation at physical parameters to ensure convergence
        # This is effectively step n_steps+1 with c_phys, d_phys
        def fun_final(x):
            p = x.reshape(size, 3)
            return _dzugutov_energy(p, c_phys, d_phys)
            
        def jac_final(x):
            p = x.reshape(size, 3)
            return _dzugutov_gradient(p, c_phys, d_phys).flatten()
            
        try:
            res_final = minimize(fun_final, pos_annealed.flatten(), jac=jac_final, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            pos_final = res_final.x.reshape(size, 3)
            energy_final = res_final.fun
        except Exception:
            pos_final = pos_annealed
            energy_final = _dzugutov_energy(pos_final, c_phys, d_phys)
            
        # Check validity (minimum distance)
        # The potential has a repulsive core, but let's ensure no atoms overlap badly
        # If energy is inf or nan, skip
        if np.isfinite(energy_final) and energy_final < best_energy:
            best_energy = energy_final
            best_positions = pos_final
            
    if best_positions is None:
        # Fallback
        best_positions = _build_initial_guess(size, rng)
        
    return best_positions
# EVOLVE-BLOCK-END
