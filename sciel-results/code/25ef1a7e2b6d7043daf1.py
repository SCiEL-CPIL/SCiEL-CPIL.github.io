# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 315.0  # Safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _min_distance(positions):
    """
    Vectorized computation of the minimum pairwise distance.
    """
    n = positions.shape[0]
    if n < 2:
        return float('inf')
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    iu = np.triu_indices(n, k=1)
    min_dist = np.min(dists[iu])
    return min_dist

def _generate_mackay_icosahedron(n_atoms, seed=42):
    """
    Generate a Mackay icosahedral cluster of exactly n_atoms.
    Uses the 63-atom core (1+12+30+20) and adds/removes atoms from the surface layers.
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (normalized to distance 1 from center)
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)] # Center
    
    # Layer 1: Vertices (12)
    for v in verts:
        points.append(v.copy())
        
    # Layer 2: Edges (30)
    edge_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                edge_points.append(mid)
    for p in edge_points:
        points.append(p)
        
    # Layer 3: Faces (20)
    face_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            for k in range(j+1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - 2.0) < 1e-6 and abs(d2 - 2.0) < 1e-6 and abs(d3 - 2.0) < 1e-6:
                    centroid = (verts[i] + verts[j] + verts[k]) / 3.0
                    face_points.append(centroid)
                    
    for p in face_points:
        points.append(p)
        
    points = np.array(points)
    
    if len(points) > n_atoms:
        # Remove outermost atoms
        distances = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(distances)
        points = points[sorted_indices[:n_atoms]]
    else:
        # Add atoms near the surface if needed (for n > 63)
        # This is a heuristic extension. We place atoms on a slightly larger icosahedron shell.
        while len(points) < n_atoms:
            # Pick a random face centroid and push it out slightly
            idx = rng.integers(20, len(points)) # Pick from face layer or beyond
            p = points[idx].copy()
            norm = np.linalg.norm(p)
            if norm > 0:
                p = p / norm * (norm + 0.1)
            else:
                p = rng.normal(0, 0.5, 3)
            points = np.vstack([points, p.reshape(1,3)])
            
    # Jitter
    jitter = rng.uniform(-0.01, 0.01, (n_atoms, 3))
    points = points + jitter
    return points

def _generate_fcc_chunk(n_atoms, seed=43):
    """
    Generate an FCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n_atoms ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
                        
    if not points:
        return _generate_mackay_icosahedron(n_atoms, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    cluster = points[sorted_indices[:n_atoms]]
    
    jitter = rng.uniform(-0.01, 0.01, (n_atoms, 3))
    cluster = cluster + jitter
    return cluster

def _rotate_vector(v, angle, axis):
    """Rotate vector v around axis by angle (Rodrigues' rotation formula)."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    k = axis
    v = np.array(v)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)

def _super_atom_energy_and_grad(params, super_atoms, rho):
    """
    Compute energy and gradient for a configuration of super-atoms.
    params: [x1, y1, z1, qx1, qy1, qz1, qw1, x2, y2, ...]
    super_atoms: list of arrays of shape (n_i, 3)
    """
    n_atoms_total = sum(sa.shape[0] for sa in super_atoms)
    positions = np.zeros((n_atoms_total, 3))
    
    # We need to compute the gradient with respect to the super-atom parameters.
    # Since the potential is pairwise and depends on distances, we can use the chain rule.
    # However, for simplicity and robustness within the time limit, we will use a 
    # numerical gradient for the super-atom parameters if analytic is too complex, 
    # or simply use L-BFGS-B on the flat atomic positions but initialized from 
    # the super-atom configuration. 
    # 
    # Actually, the prompt suggests optimizing relative positions/orientations.
    # Let's implement a numerical gradient for the super-atom parameters to be safe and fast enough.
    # 
    # Wait, the "Strategy" says: "optimizing only their relative positions and orientations via gradient descent".
    # This implies a lower-dimensional optimization.
    # 
    # Let's implement the energy function for the super-atom parameters.
    
    idx = 0
    for i, sa in enumerate(super_atoms):
        tx, ty, tz = params[idx], params[idx+1], params[idx+2]
        qx, qy, qz, qw = params[idx+3], params[idx+4], params[idx+5], params[idx+6]
        idx += 7
        
        # Normalize quaternion
        q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if q_norm < 1e-9:
            continue
        qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
        
        # Apply rotation and translation
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        rotated_sa = sa @ R.T
        translated_sa = rotated_sa + np.array([tx, ty, tz])
        
        positions[idx - 7 + np.arange(sa.shape[0]), :] = translated_sa # This index logic is wrong, let's fix it
        
    # Let's re-do the position assembly correctly
    positions = np.zeros((n_atoms_total, 3))
    start_idx = 0
    for i, sa in enumerate(super_atoms):
        tx, ty, tz = params[i*7], params[i*7+1], params[i*7+2]
        qx, qy, qz, qw = params[i*7+3], params[i*7+4], params[i*7+5], params[i*7+6]
        
        q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if q_norm < 1e-9:
            continue
        qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
        
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        rotated_sa = sa @ R.T
        translated_sa = rotated_sa + np.array([tx, ty, tz])
        
        positions[start_idx:start_idx+sa.shape[0]] = translated_sa
        start_idx += sa.shape[0]
        
    # Compute Energy
    try:
        energy, _ = morse_energy_gradient(positions, rho)
    except Exception:
        energy = 1e10
        
    # Compute Gradient (Numerical)
    grad = np.zeros_like(params)
    h = 1e-4
    for j in range(len(params)):
        params_plus = params.copy()
        params_plus[j] += h
        # Re-compute energy for params_plus
        positions_plus = np.zeros((n_atoms_total, 3))
        start_idx = 0
        for i, sa in enumerate(super_atoms):
            tx, ty, tz = params_plus[i*7], params_plus[i*7+1], params_plus[i*7+2]
            qx, qy, qz, qw = params_plus[i*7+3], params_plus[i*7+4], params_plus[i*7+5], params_plus[i*7+6]
            
            q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if q_norm < 1e-9:
                continue
            qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
            
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])
            
            rotated_sa = sa @ R.T
            translated_sa = rotated_sa + np.array([tx, ty, tz])
            
            positions_plus[start_idx:start_idx+sa.shape[0]] = translated_sa
            start_idx += sa.shape[0]
            
        try:
            energy_plus, _ = morse_energy_gradient(positions_plus, rho)
        except Exception:
            energy_plus = 1e10
            
        grad[j] = (energy_plus - energy) / h
        
    return energy, grad

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Define Super-Atom Decomposition
    # Strategy: Use 1 or 2 large super-atoms.
    # For n <= 63: Single icosahedron super-atom. Optimize only its center (fixed at 0) and maybe slight scaling? 
    # Actually, if it's a single super-atom, the internal structure is fixed. 
    # The "optimization" is trivial (just relaxation).
    # To make it interesting for n > 63 or to explore different isomers, we can split the cluster.
    # 
    # Let's try a 2-super-atom approach for n > 40.
    # Split n into n1 and n2.
    # Super-atom 1: Icosahedron of n1 atoms.
    # Super-atom 2: FCC chunk of n2 atoms.
    # Optimize their relative position and orientation.
    
    if n <= 30:
        # Single super-atom
        super_atoms = [_generate_mackay_icosahedron(n, seed=42)]
        n_super = 1
    else:
        # Two super-atoms
        n1 = int(n * 0.6)
        n2 = n - n1
        super_atoms = [
            _generate_mackay_icosahedron(n1, seed=42),
            _generate_fcc_chunk(n2, seed=43)
        ]
        n_super = 2
        
    # 2. Initial Parameters
    # For each super-atom: [tx, ty, tz, qx, qy, qz, qw]
    # Initial: Centered at 0, Identity Rotation.
    # For n_super > 1, place them apart.
    params = np.zeros(n_super * 7)
    if n_super == 2:
        # Place second super-atom away from the first
        # Estimate radius of super-atom 1
        r1 = 1.8 * (n1 ** (1.0/3.0))
        params[7:10] = [2*r1, 0, 0] # Translate SA2 to +x
        
    # 3. Optimize Super-Atom Parameters
    # Use L-BFGS-B to optimize the low-dimensional parameter space
    try:
        res = minimize(
            _super_atom_energy_and_grad, 
            params, 
            args=(super_atoms, rho),
            method='L-BFGS-B',
            jac=True, # We return (f, g)
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
        )
        
        if res.success:
            best_params = res.x
            # Reconstruct positions from best_params
            n_atoms_total = sum(sa.shape[0] for sa in super_atoms)
            positions = np.zeros((n_atoms_total, 3))
            start_idx = 0
            for i, sa in enumerate(super_atoms):
                tx, ty, tz = best_params[i*7], best_params[i*7+1], best_params[i*7+2]
                qx, qy, qz, qw = best_params[i*7+3], best_params[i*7+4], best_params[i*7+5], best_params[i*7+6]
                
                q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                if q_norm < 1e-9:
                    continue
                qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
                
                R = np.array([
                    [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
                ])
                
                rotated_sa = sa @ R.T
                translated_sa = rotated_sa + np.array([tx, ty, tz])
                
                positions[start_idx:start_idx+sa.shape[0]] = translated_sa
                start_idx += sa.shape[0]
                
            # Check validity
            min_dist = _min_distance(positions)
            if min_dist >= 0.5:
                energy, _ = morse_energy_gradient(positions, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    except Exception:
        pass

    # 4. Fallback: If super-atom optimization failed or yielded poor results, 
    #    use the atomic relaxation on the best super-atom configuration found so far.
    #    If no positions were found, generate a random one.
    if best_positions is None:
        # Generate a simple random cluster
        rng = np.random.default_rng(45)
        radius = 1.5 * (n ** (1.0/3.0))
        best_positions = rng.uniform(-radius, radius, (n, 3))
        # Ensure min distance
        # Simple rejection sampling
        for _ in range(1000):
            min_dist = _min_distance(best_positions)
            if min_dist >= 0.5:
                break
            # Nudge overlapping atoms
            iu = np.triu_indices(n, k=1)
            dists = np.linalg.norm(best_positions[iu[0]] - best_positions[iu[1]], axis=1)
            bad = np.where(dists < 0.5)[0]
            if len(bad) > 0:
                idx = bad[0]
                i, j = iu[0][idx], iu[1][idx]
                # Push them apart
                diff = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-6:
                    diff = rng.normal(0, 1, 3)
                else:
                    diff = diff / norm
                best_positions[i] += diff * 0.1
                best_positions[j] -= diff * 0.1

    # 5. Final Atomic Relaxation
    # This is crucial to fix any local distortions from the super-atom approximation
    try:
        if _get_remaining_time() > 10.0:
            relaxed_energy, relaxed_positions = relax(best_positions, potential, rho)
            min_dist = _min_distance(relaxed_positions)
            if min_dist >= 0.5:
                if relaxed_energy < best_energy:
                    best_energy = relaxed_energy
                    best_positions = relaxed_positions
    except Exception:
        pass

    # 6. If time permits, do a few basin hopping steps on the atomic level
    #    starting from the relaxed super-atom configuration.
    if best_positions is not None and _get_remaining_time() > 20.0:
        rng = np.random.default_rng(12345)
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        step_size = 0.2
        temp = 1.0
        
        for step in range(50):
            if _get_remaining_time() < 10.0:
                break
                
            noise = rng.normal(0, step_size, current_positions.shape)
            perturbed = current_positions + noise
            
            try:
                new_energy, new_positions = relax(perturbed, potential, rho)
            except Exception:
                continue
                
            min_dist = _min_distance(new_positions)
            if min_dist < 0.5:
                continue
                
            if new_energy < current_energy:
                accept = True
            else:
                delta_e = new_energy - current_energy
                if temp > 1e-6:
                    accept_prob = math.exp(-delta_e / temp)
                    accept = rng.random() < accept_prob
                else:
                    accept = False
                    
            if accept:
                current_energy = new_energy
                current_positions = new_positions
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_positions.copy()
                    
                temp *= 0.95
                step_size *= 0.98
            else:
                temp *= 0.98
                step_size *= 0.99

    return best_positions
# EVOLVE-BLOCK-END
