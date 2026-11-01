"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r6 = r2 ** -3
            r12 = r6 ** 2
            coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS-B."""
    return _lj_energy(pos), _lj_grad(pos)


def _get_icosahedron_vertices():
    """
    Returns the 12 unit vectors pointing to the vertices of a regular icosahedron.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _get_icosahedron_edges_and_faces(verts):
    """
    Identifies edges (min distance pairs) and face centers (normalized sum of 3 vertices forming a triangle)
    for the icosahedron defined by verts.
    """
    from scipy.spatial.distance import cdist
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Face centers:
    # An icosahedron has 20 faces. Each face is a triangle of 3 vertices.
    # We can find faces by looking for triplets (i, j, k) that form an equilateral triangle
    # and whose center is not already covered.
    # Alternatively, the face centers of an icosahedron are the vertices of a dodecahedron.
    # The dodecahedron vertices are:
    # (±1, ±1, ±1)
    # (0, ±1/phi, ±phi)
    # (±1/phi, ±phi, 0)
    # (±phi, 0, ±1/phi)
    # Let's generate these and normalize.
    inv_phi = 1.0 / phi
    face_dirs = []
    # Cube vertices
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                face_dirs.append([x, y, z])
    # Other 8 vertices of dodecahedron
    for x in [-1/phi, 1/phi]:
        for y in [-phi, phi]:
            for z in [0]:
                face_dirs.append([x, y, z])
    for x in [0]:
        for y in [-1/phi, 1/phi]:
            for z in [-phi, phi]:
                face_dirs.append([x, y, z])
    for x in [-phi, phi]:
        for y in [0]:
            for z in [-1/phi, 1/phi]:
                face_dirs.append([x, y, z])
                
    face_dirs = np.array(face_dirs, dtype=float)
    face_norms = np.linalg.norm(face_dirs, axis=1, keepdims=True)
    face_dirs = face_dirs / face_norms
    
    # Edge midpoints directions
    edge_mids = []
    for i, j in edges:
        mid_dir = (verts[i] + verts[j])
        mid_norm = np.linalg.norm(mid_dir)
        if mid_norm > 1e-10:
            edge_mids.append(mid_dir / mid_norm)
        else:
            edge_mids.append(np.array([0,0,0]))
            
    return np.array(edge_mids), face_dirs


def _construct_mackay_55(r1, r2):
    """
    Construct the 55-atom Mackay icosahedron with given shell radii.
    r1: radius of inner shell (12 atoms)
    r2: radius of outer shell (42 atoms)
    """
    verts = _get_icosahedron_vertices()
    edge_mids, face_dirs = _get_icosahedron_edges_and_faces(verts)
    
    # Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Inner shell: 12 vertices
    shell1 = verts * r1
    
    # Outer shell: 30 edge midpoints + 20 face centers = 50? 
    # Wait. 1 + 12 + 42 = 55.
    # The outer shell of the Mackay icosahedron (N=55) consists of:
    # - 12 atoms at the vertices of the outer icosahedron? No.
    # Let's check the counts.
    # Mackay icosahedron N=55:
    # Layer 0: 1 atom (center)
    # Layer 1: 12 atoms (icosahedron vertices)
    # Layer 2: 42 atoms.
    # The 42 atoms in the second shell of the Mackay icosahedron are located at:
    # - 20 positions corresponding to the face centers of the inner icosahedron.
    # - 30 positions corresponding to the edge midpoints of the inner icosahedron?
    # Actually, the standard Mackay icosahedron for N=55 is:
    # 1 + 12 + 42.
    # The 42 outer atoms are at the vertices of a "truncated" icosahedron? No.
    # Let's look at the symmetry.
    # The 42 outer atoms form two orbits under icosahedral symmetry:
    # 1. 20 atoms at the face centers of the inner icosahedron.
    # 2. 30 atoms at the edge midpoints of the inner icosahedron? 
    # Wait, 20 + 30 = 50. That's too many.
    
    # Correction:
    # The Mackay icosahedron for N=55 is actually:
    # 1 (center)
    # 12 (inner shell, icosahedron vertices)
    # 42 (outer shell)
    # The outer shell consists of:
    # - 12 atoms at the vertices of a larger icosahedron? No, that would be 1+12+12=25.
    # Let's re-verify the structure of N=55 LJ cluster.
    # It is the Mackay icosahedron with shells 1, 12, 42.
    # The 42 outer atoms are located at:
    # - 20 positions: face centers of the inner icosahedron.
    # - 22 positions? No.
    
    # Let's check the Cambridge Cluster Database description for LJ55.
    # "Mackay icosahedron: 1 + 12 + 42"
    # The 42 outer atoms are:
    # 1. 12 atoms at the vertices of the outer icosahedron (same directions as inner vertices).
    # 2. 30 atoms at the edge midpoints of the outer icosahedron?
    # If we take the outer icosahedron vertices (12) and edge midpoints (30), that's 42.
    # This matches the count.
    # So:
    # Inner shell: 12 atoms at directions of icosahedron vertices, radius r1.
    # Outer shell: 
    #   - 12 atoms at directions of icosahedron vertices, radius r2.
    #   - 30 atoms at directions of icosahedron edge midpoints, radius r2.
    
    # Let's verify this geometry.
    # If the outer shell has 12 vertices + 30 edge midpoints, it is a "truncated" icosahedron?
    # No, a truncated icosahedron has 12 pentagons and 20 hexagons, 60 vertices.
    # The set of 12 vertices + 30 edge midpoints of an icosahedron forms a set of 42 points.
    # This is the correct geometry for the Mackay icosahedron's outer shell.
    
    vertex_outer = verts * r2
    edge_outer = edge_mids * r2
    
    shell2 = np.vstack([vertex_outer, edge_outer])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def _energy_symmetric(params):
    """
    Objective function for symmetric optimization.
    params: [r1, r2]
    """
    r1, r2 = params
    pos = _construct_mackay_55(r1, r2)
    return _lj_energy(pos)


def _relax(positions, maxiter=500, gtol=1e-10, deadline=None):
    """L-BFGS-B local relaxation."""
    if deadline is None:
        deadline = _DEADLINE
    
    if _remaining() < 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(-1, 3)
    except Exception:
        return positions


def _polish(positions, deadline=None):
    """High-precision polish pass."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    # Multi-stage polish
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize the symmetric Mackay icosahedron parameters (r1, r2)
    # This is very fast and robust.
    # Initial guess: r1 ~ 1.12, r2 ~ 2.24
    x0 = np.array([1.12, 2.24])
    
    try:
        res_sym = minimize(_energy_symmetric, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        r1_opt, r2_opt = res_sym.x
        mackay_pos = _construct_mackay_55(r1_opt, r2_opt)
        e_sym = _lj_energy(mackay_pos)
        
        if _check_min_dist(mackay_pos) and e_sym < best_energy:
            best_energy = e_sym
            best_pos = mackay_pos.copy()
            
        # Step 2: Relax the symmetric structure in Cartesian space
        # This allows for any symmetry-breaking distortions if they exist (though unlikely for global min)
        if _remaining() > 10.0:
            relaxed = _relax(mackay_pos, maxiter=500, gtol=1e-10)
            e_relaxed = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = relaxed.copy()
                
    except Exception as e:
        print(f"Symmetric optimization failed: {e}")
        # Fallback to fixed Mackay
        mackay_pos = _construct_mackay_55(1.12, 2.24)
        best_pos = mackay_pos
        best_energy = _lj_energy(mackay_pos)

    # Step 3: High precision polish
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _polish(best_pos, deadline=_DEADLINE)
            e_polished = _lj_energy(polished)
            if _check_min_dist(polished) and e_polished < best_energy:
                best_energy = e_polished
                best_pos = polished.copy()
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
# EVOLVE-BLOCK-END
