# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron.
    Returns:
      directions: (55, 3) array of unit vectors (first is [0,0,0] for center)
      shell_ids: (55,) array of shell indices (0=center, 1=inner verts, 2=midpoints, 3=outer verts)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Build direction array and shell assignments
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = get_mackay_directions()
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    return pos

def relax(pos, maxiter=300, gtol=1e-6):
    """Relax using L-BFGS-B with analytical gradient."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def rotate_shell(pos, shell_indices, angle, axis):
    """Rotate a specific set of atoms in pos by angle around axis."""
    # Rodrigues rotation formula
    c = np.cos(angle)
    s_ = np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = c * np.eye(3) + s_ * K + (1 - c) * np.outer(axis, axis)
    
    new_pos = pos.copy()
    new_pos[shell_indices] = pos[shell_indices] @ R.T
    return new_pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Get base Mackay geometry with good radii
    # These radii are close to the optimal LJ Mackay icosahedron
    base_radii = np.array([1.04, 1.76, 1.98])
    base_pos = make_mackay_from_radii(base_radii)
    
    # Quick relaxation to ensure we are in the basin
    base_pos, base_e = relax(base_pos, maxiter=100, gtol=1e-4)
    
    best_pos = base_pos.copy()
    best_e = base_e
    
    # 2. Explore "Twist" degrees of freedom
    # The Mackay icosahedron has 3 shells. We can rotate each shell independently.
    # This allows us to find lower energy configurations if the ideal Mackay is slightly distorted.
    
    # Define shell indices
    shell_1_idx = np.arange(1, 13)   # Inner 12
    shell_2_idx = np.arange(13, 43)  # Mid 30
    shell_3_idx = np.arange(43, 55)  # Outer 12
    
    # Try a set of random twists for each shell
    rng = np.random.RandomState(42)
    
    # We will try 20 random twist configurations
    for i in range(20):
        if time.time() > deadline - 20:
            break
            
        # Random small rotations for each shell
        # Angle range: -0.1 to 0.1 radians (approx 5 degrees)
        angle_1 = rng.uniform(-0.1, 0.1)
        angle_2 = rng.uniform(-0.1, 0.1)
        angle_3 = rng.uniform(-0.1, 0.1)
        
        axis_1 = rng.randn(3)
        axis_2 = rng.randn(3)
        axis_3 = rng.randn(3)
        
        axis_1 /= np.linalg.norm(axis_1)
        axis_2 /= np.linalg.norm(axis_2)
        axis_3 /= np.linalg.norm(axis_3)
        
        pos_try = base_pos.copy()
        
        # Apply rotations
        pos_try = rotate_shell(pos_try, shell_1_idx, angle_1, axis_1)
        pos_try = rotate_shell(pos_try, shell_2_idx, angle_2, axis_2)
        pos_try = rotate_shell(pos_try, shell_3_idx, angle_3, axis_3)
        
        # Relax
        pos_relaxed, e_relaxed = relax(pos_try, maxiter=100, gtol=1e-4)
        
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
            
    # 3. Final high-precision polish
    # Use the best position found so far and relax with very tight tolerances
    if time.time() < deadline - 5:
        # Pass 1
        pos_final, e_final = relax(best_pos, maxiter=500, gtol=1e-8)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
            
    if time.time() < deadline - 3:
        # Pass 2
        pos_final, e_final = relax(best_pos, maxiter=1000, gtol=1e-10)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
            
    if time.time() < deadline - 1:
        # Pass 3
        pos_final, e_final = relax(best_pos, maxiter=2000, gtol=1e-12)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final

    return best_pos

# EVOLVE-BLOCK-END
