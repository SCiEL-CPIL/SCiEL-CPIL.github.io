import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _rotation_matrix_from_quat(q):
    """Build 3x3 rotation matrix from unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-15:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def _quat_multiply(q1, q2):
    """Multiply two quaternions (w,x,y,z)."""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

_FCC_CACHE = {}

def _get_fcc_shell_structure(n_atoms=38):
    """
    Generate the 38 closest atoms in an FCC lattice.
    Returns base coordinates (unscaled), shell IDs, and unique radii.
    """
    if n_atoms in _FCC_CACHE:
        return _FCC_CACHE[n_atoms]
    
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 10
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    radii = np.linalg.norm(coords, axis=1)
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    
    _FCC_CACHE[n_atoms] = (coords, shell_ids, unique_r)
    return coords, shell_ids, unique_r

def _make_truncated_octahedron():
    """
    Build an explicit FCC truncated octahedron for 38 atoms.
    FCC truncated octahedron: 1 + 12 + 14 + 11 = 38
    Shell 1 (12): face centers of octahedron - FCC positions at (±1,±1,0) etc., scaled
    Shell 2 (14): edge centers - FCC positions at (±2,0,0) type, plus face centers
    Shell 3 (11): remaining
    
    Actually, the standard FCC truncated octahedron for 38 atoms has:
    - 1 center
    - 12 atoms at distance sqrt(2) (face centers of the octahedron)
    - 14 atoms at distance 2 (edge midpoints and some face centers)
    - 11 atoms at distance sqrt(6) or similar
    
    Let me use the actual FCC lattice points sorted by radius.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    # Scale to LJ equilibrium: nearest neighbor should be 2^(1/6)
    # In our base coords, NN distance is 1.0 (after dividing by sqrt(2))
    # Wait, let's check: FCC points at (1,1,0) have distance sqrt(2) from origin
    # After scaling by 1/sqrt(2), they're at distance 1.
    # So NN distance in base_coords is 1.0.
    # We want NN distance = 2^(1/6), so global scale = 2^(1/6)
    scale = 2.0 ** (1.0/6.0)
    return base_coords * scale

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient for full Cartesian coordinates."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def _lj_energy_only(pos_flat):
    """Compute LJ energy only (no gradient) - faster for basin hopping."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _basin_hop(x0, current_energy, rng, step_size, n_iter, deadline_margin=20, temp_start=0.15, temp_end=0.02):
    """
    Basin hopping with adaptive step size and annealing Metropolis acceptance.
    Returns (best_energy, best_x).
    """
    best_bh_energy = current_energy
    best_bh_x = x0.copy()
    
    # Adaptive step size
    base_step = step_size
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - deadline_margin:
            break
        
        # Annealing temperature
        progress = i / max(n_iter - 1, 1)
        temp = temp_start + (temp_end - temp_start) * progress
        
        # Adaptive step size: occasionally increase or decrease
        if i % 8 == 4:
            if rng.uniform() < 0.35:
                step_size = base_step * rng.uniform(1.5, 3.0)
            elif rng.uniform() < 0.3:
                step_size = base_step * rng.uniform(0.3, 0.7)
            else:
                step_size = base_step
        
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': 150, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy = _lj_energy_only(new_x)
            
            if new_energy < current_energy:
                x0 = new_x
                current_energy = new_energy
                if new_energy < best_bh_energy:
                    best_bh_energy = new_energy
                    best_bh_x = new_x.copy()
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    accept = math.exp(-delta / max(temp, 1e-10))
                    if rng.uniform() < accept:
                        x0 = new_x
                        current_energy = new_energy
        except Exception:
            continue
    
    return best_bh_energy, best_bh_x

def _make_icosahedral_structure():
    """
    Generate an icosahedral structure for 38 atoms.
    1 center + 12 inner shell + 25 outer shell
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    min_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
    
    scale = 1.0 / min_dist
    vertices = vertices * scale
    
    positions = np.zeros((38, 3))
    positions[0] = [0.0, 0.0, 0.0]
    positions[1:13] = vertices
    
    # Outer 25 atoms: place on a larger sphere with good angular distribution
    n_outer = 25
    r_outer = 1.8
    
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(1.0 - z*z)
        theta = math.pi * (1.0 + math.sqrt(5.0)) * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions[13 + i] = np.array([x, y, z]) * r_outer
    
    return positions

def _topology_loss_and_grad(positions):
    """
    Compute a smooth topology-forcing loss that penalizes deviation from the
    FCC truncated octahedron shell radii. Returns (loss, gradient).
    
    The loss is based on comparing each atom's radial distance to its expected
    shell radius in the ideal FCC structure.
    """
    n_atoms = N
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    # Ideal radii for each shell (scaled to LJ equilibrium distance)
    ideal_scale = 2.0 ** (1.0/6.0)
    ideal_radii = unique_r * ideal_scale
    
    # Compute current radial distances
    radii = np.linalg.norm(positions, axis=1)
    
    # For each atom, find its target radius based on its shell ID
    target_radii = np.zeros(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(ideal_radii):
            target_radii[i] = ideal_radii[s]
        else:
            target_radii[i] = ideal_radii[-1]
    
    # Smooth loss: sum of squared deviations from target radii
    # Weight by shell to prioritize inner shells
    shell_weights = np.ones(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        shell_weights[i] = 1.0 + 0.5 * (n_shells - s)  # Higher weight for inner shells
    
    diff = radii - target_radii
    loss = np.sum(shell_weights * diff ** 2)
    
    # Gradient: d(loss)/d(positions)
    # d(loss)/d(r_i) = 2 * shell_weights[i] * (radii[i] - target_radii[i])
    # d(r_i)/d(positions[i]) = positions[i] / r_i
    grad = np.zeros_like(positions)
    for i in range(n_atoms):
        r = radii[i]
        if r > 1e-8:
            grad[i] = 2.0 * shell_weights[i] * diff[i] * positions[i] / r
    
    return loss, grad

def _topology_forcing_optimize(initial_positions, max_iter=500, gtol=1e-6):
    """
    Optimize positions to match FCC topology using the topology loss.
    This helps navigate toward the FCC truncated octahedron structure.
    """
    x0 = initial_positions.flatten()
    
    def topology_obj_and_grad(x_flat):
        pos = x_flat.reshape(N, 3)
        loss, grad = _topology_loss_and_grad(pos)
        return loss, grad.flatten()
    
    try:
        result = minimize(topology_obj_and_grad, x0, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return initial_positions

def _optimize_fcc_with_shell_breathing():
    """
    Optimize FCC structure with per-shell breathing parameters.
    Returns (positions, energy) or (None, inf) on failure.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    initial_scale = 2.0 ** (1.0/6.0)
    
    def _shell_breath_energy_and_grad(x_flat):
        """Energy and gradient with respect to [global_scale, shell_breath_0, ..., shell_breath_{n-1}]."""
        gscale = x_flat[0]
        breaths = 1.0 + x_flat[1:]  # breath params are deviations from 1
        
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(breaths):
                positions[i] = base_coords[i] * breaths[s] * gscale
            else:
                positions[i] = base_coords[i] * gscale
        
        # Compute LJ energy
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
        
        # Gradient with respect to positions
        inv_r14 = inv_r12 * inv_r2
        inv_r8 = inv_r6 * inv_r2
        coeff = -24.0 * inv_r14 + 12.0 * inv_r8
        grad_pos = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Chain rule: dE/dx_flat
        grad_flat = np.zeros_like(x_flat)
        # dE/d(gscale) = sum_i grad_pos[i] . (base_coords[i] * breaths[s_i])
        for i in range(N):
            s = shell_ids[i]
            b = breaths[s] if s < len(breaths) else 1.0
            grad_flat[0] += np.dot(grad_pos[i], base_coords[i] * b)
            # dE/d(breath_s) = sum_{i: shell_i=s} grad_pos[i] . (base_coords[i] * gscale)
            if s < len(breaths):
                grad_flat[1 + s] += np.dot(grad_pos[i], base_coords[i] * gscale)
        
        return energy, grad_flat
    
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        res = minimize(_shell_breath_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 2000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-16})
        
        global_scale = res.x[0]
        shell_params = res.x[1:]
        
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(shell_params):
                breath = 1.0 + shell_params[s]
            else:
                breath = 1.0
            positions[i] = base_coords[i] * breath * global_scale
        
        if check_validity(positions):
            energy = lj_energy_vectorized(positions)
            return positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _two_stage_topology_optimize(start_positions):
    """
    Two-stage optimization:
    1. Use topology-forcing loss to guide toward FCC structure
    2. Then do full LJ energy optimization
    
    Returns (positions, energy) or (None, inf) on failure.
    """
    # Stage 1: Topology forcing - guide positions toward FCC shell radii
    try:
        topo_positions = _topology_forcing_optimize(start_positions, max_iter=300, gtol=1e-5)
        
        # Stage 2: Full LJ energy relaxation from topology-guided positions
        lj_positions = relax(topo_positions, maxiter=1000, gtol=1e-10)
        
        if check_validity(lj_positions):
            energy = lj_energy_vectorized(lj_positions)
            return lj_positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _rigid_body_fcc_optimize():
    """
    Rigid-body optimization: parameterize the 38-atom FCC structure as two
    interlocking 19-atom sub-clusters (FCC A and B sublattices) with a shared
    rotation and relative translation. This reduces DOF from 114 to ~12,
    explicitly enforcing FCC stacking and preventing icosahedral collapse.
    
    Parameters: [qx, qy, qz, qw] (rotation quat, 4) + [tx, ty, tz] (rel trans, 3)
               + [scale, breath_A, breath_B] (3) = 10 params
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    scale = 2.0 ** (1.0/6.0)
    
    # Identify A and B sublattices by parity of (i+j+k) in FCC integer coords
    # In our scaled coords, A sublattice has even sum, B has odd
    # Reconstruct integer indices
    inv_scale = 1.0 / scale
    int_coords = []
    for c in base_coords:
        # base_coords = (i/sqrt2, j/sqrt2, k/sqrt2) where (i+j+k)%2==0
        # So integer coords are i = c[0]*sqrt2, etc.
        ic = np.round(c * math.sqrt(2.0))
        int_coords.append(ic)
    int_coords = np.array(int_coords)
    
    # FCC condition: (i+j+k) % 2 == 0
    parity = np.sum(int_coords, axis=1) % 2
    # In FCC, all points have even parity by construction
    # Split into two interpenetrating simple cubic sublattices:
    # A: all even coordinates (i,j,k all even)
    # B: mixed parity but sum even -> e.g., (odd, odd, even) permutations
    all_even = np.all(int_coords % 2 == 0, axis=1)
    
    if np.sum(all_even) >= 10 and np.sum(~all_even) >= 10:
        idx_A = np.where(all_even)[0]
        idx_B = np.where(~all_even)[0]
    else:
        # Fallback: split by first coordinate parity
        idx_A = np.where(int_coords[:, 0] % 2 == 0)[0]
        idx_B = np.where(int_coords[:, 0] % 2 != 0)[0]
    
    A_base = base_coords[idx_A].copy()
    B_base = base_coords[idx_B].copy()
    nA = len(idx_A)
    nB = len(idx_B)
    
    def _rigid_energy_and_grad(x_flat):
        """x_flat = [qx, qy, qz, qw, tx, ty, tz, scale, breath_A, breath_B]"""
        qx, qy, qz, qw = x_flat[0:4]
        tx, ty, tz = x_flat[4:7]
        sc = x_flat[7]
        bA = 1.0 + x_flat[8]
        bB = 1.0 + x_flat[9]
        
        R = _rotation_matrix_from_quat([qw, qx, qy, qz])
        
        # Apply rotation and scaling
        A_pos = (A_base * bA) @ R.T * sc
        B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
        
        # Assemble full positions
        positions = np.zeros((N, 3))
        positions[idx_A] = A_pos
        positions[idx_B] = B_pos
        
        # Compute LJ energy
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
        
        # Gradient w.r.t. positions
        inv_r14 = inv_r12 * inv_r2
        inv_r8 = inv_r6 * inv_r2
        coeff = -24.0 * inv_r14 + 12.0 * inv_r8
        grad_pos = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Chain rule
        grad_flat = np.zeros(10)
        
        # dE/d(tx,ty,tz) = sum_{i in B} grad_pos[i]
        grad_flat[4:7] = np.sum(grad_pos[idx_B], axis=0)
        
        # dE/d(scale) = sum_i grad_pos[i] . (R @ (base_i * b_s))
        for i in range(N):
            if i in idx_A:
                pos_unit = (A_base[np.where(idx_A == i)[0][0]] * bA) @ R.T
            else:
                pos_unit = (B_base[np.where(idx_B == i)[0][0]] * bB) @ R.T
            grad_flat[7] += np.dot(grad_pos[i], pos_unit)
        
        # dE/d(breath_A) = sum_{i in A} grad_pos[i] . (R @ A_base_i * scale)
        for i in idx_A:
            ai = np.where(idx_A == i)[0][0]
            pos_unit = (A_base[ai]) @ R.T * sc
            grad_flat[8] += np.dot(grad_pos[i], pos_unit)
        
        # dE/d(breath_B) = sum_{i in B} grad_pos[i] . (R @ B_base_i * scale)
        for i in idx_B:
            bi = np.where(idx_B == i)[0][0]
            pos_unit = (B_base[bi]) @ R.T * sc
            grad_flat[9] += np.dot(grad_pos[i], pos_unit)
        
        # dE/d(quaternion) - approximate via finite difference for stability
        # (Analytic quaternion gradient is complex; use central differences)
        eps = 1e-6
        for k in range(4):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[k] += eps
            x_minus[k] -= eps
            
            def _eval_energy(x):
                qx, qy, qz, qw = x[0:4]
                tx, ty, tz = x[4:7]
                sc = x[7]
                bA = 1.0 + x[8]
                bB = 1.0 + x[9]
                R = _rotation_matrix_from_quat([qw, qx, qy, qz])
                A_pos = (A_base * bA) @ R.T * sc
                B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
                positions = np.zeros((N, 3))
                positions[idx_A] = A_pos
                positions[idx_B] = B_pos
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                r2 = np.sum(diff * diff, axis=-1)
                np.fill_diagonal(r2, 1.0)
                r2 = np.maximum(r2, 1e-12)
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3
                inv_r12 = inv_r6 ** 2
                return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
            
            grad_flat[k] = (_eval_energy(x_plus) - _eval_energy(x_minus)) / (2 * eps)
        
        return energy, grad_flat
    
    # Initial guess: identity rotation, zero translation, scale=2^(1/6), no breathing
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0])
    
    try:
        res = minimize(_rigid_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16})
        
        qx, qy, qz, qw = res.x[0:4]
        tx, ty, tz = res.x[4:7]
        sc = res.x[7]
        bA = 1.0 + res.x[8]
        bB = 1.0 + res.x[9]
        
        R = _rotation_matrix_from_quat([qw, qx, qy, qz])
        A_pos = (A_base * bA) @ R.T * sc
        B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
        
        positions = np.zeros((N, 3))
        positions[idx_A] = A_pos
        positions[idx_B] = B_pos
        
        if check_validity(positions):
            energy = lj_energy_vectorized(positions)
            return positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _high_precision_polish(positions):
    """
    Final high-precision local optimization with very tight tolerances.
    """
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': 10000, 'gtol': 1e-14, 'maxls': 200, 'ftol': 1e-18})
        polished = result.x.reshape(N, 3)
        if check_validity(polished):
            energy = lj_energy_vectorized(polished)
            return polished, energy
    except Exception:
        pass
    
    return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1a: Rigid-body FCC sub-cluster optimization (reduced DOF, enforces FCC)
    try:
        rb_pos, rb_energy = _rigid_body_fcc_optimize()
        if rb_pos is not None:
            best_energy = rb_energy
            best_positions = rb_pos
    except Exception:
        pass
    
    # Phase 1b: Shell-breathing FCC optimization
    try:
        fcc_pos, fcc_energy = _optimize_fcc_with_shell_breathing()
        if fcc_pos is not None:
            if fcc_energy < best_energy:
                best_energy = fcc_energy
                best_positions = fcc_pos
    except Exception:
        pass
    
    # Phase 2: Two-stage topology-forcing optimization from multiple starting points
    if time.time() - START_TIME < DEADLINE - 60:
        # Try from FCC structure
        if best_positions is not None:
            try:
                topo_pos, topo_energy = _two_stage_topology_optimize(best_positions)
                if topo_pos is not None and topo_energy < best_energy:
                    best_energy = topo_energy
                    best_positions = topo_pos
            except Exception:
                pass
        
        # Try from icosahedral structure - this may escape the icosahedral funnel
        try:
            ico_positions = _make_icosahedral_structure()
            ico_relaxed = relax(ico_positions, maxiter=200, gtol=1e-6)
            
            if check_validity(ico_relaxed):
                topo_pos, topo_energy = _two_stage_topology_optimize(ico_relaxed)
                if topo_pos is not None:
                    if topo_energy < best_energy:
                        best_energy = topo_energy
                        best_positions = topo_pos
        except Exception:
            pass
        
        # Try from a perturbed FCC structure
        try:
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            scale = 2.0 ** (1.0/6.0)
            perturbed = base_coords * scale
            rng = np.random.default_rng(777)
            perturbed = perturbed + rng.normal(0, 0.1, perturbed.shape)
            
            topo_pos, topo_energy = _two_stage_topology_optimize(perturbed)
            if topo_pos is not None:
                if topo_energy < best_energy:
                    best_energy = topo_energy
                    best_positions = topo_pos
        except Exception:
            pass
    
    # Phase 3: High-precision polish of the current best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Phase 4: Basin hopping from the current best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 50:
        x0 = best_positions.flatten()
        current_energy = _lj_energy_only(x0)
        
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
            if time.time() - START_TIME > DEADLINE - 25:
                break
            
            rng = np.random.default_rng(seed)
            step_size = 0.15 + 0.05 * (seed % 4)
            
            bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 60, deadline_margin=20, 
                                         temp_start=0.20, temp_end=0.03)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_x.reshape(N, 3)
                # Update x0 for next iteration
                x0 = bh_x
                current_energy = bh_energy
    
    # Phase 4b: Rigid-body basin hopping (perturb in reduced 10-D space)
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            scale = 2.0 ** (1.0/6.0)
            inv_scale = 1.0 / scale
            int_coords = np.round(base_coords * math.sqrt(2.0))
            all_even = np.all(int_coords % 2 == 0, axis=1)
            idx_A = np.where(all_even)[0]
            idx_B = np.where(~all_even)[0]
            A_base = base_coords[idx_A].copy()
            B_base = base_coords[idx_B].copy()
            
            def _rb_energy(x_flat):
                qx, qy, qz, qw = x_flat[0:4]
                tx, ty, tz = x_flat[4:7]
                sc = x_flat[7]
                bA = 1.0 + x_flat[8]
                bB = 1.0 + x_flat[9]
                R = _rotation_matrix_from_quat([qw, qx, qy, qz])
                A_pos = (A_base * bA) @ R.T * sc
                B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
                positions = np.zeros((N, 3))
                positions[idx_A] = A_pos
                positions[idx_B] = B_pos
                return lj_energy_vectorized(positions)
            
            def _rb_grad(x_flat):
                eps = 1e-6
                grad = np.zeros(10)
                for k in range(10):
                    x_plus = x_flat.copy()
                    x_minus = x_flat.copy()
                    x_plus[k] += eps
                    x_minus[k] -= eps
                    grad[k] = (_rb_energy(x_plus) - _rb_energy(x_minus)) / (2 * eps)
                return grad
            
            def _rb_energy_grad(x_flat):
                return _rb_energy(x_flat), _rb_grad(x_flat)
            
            x0_rb = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0])
            # Initialize from current best if possible
            if best_positions is not None:
                # Try to extract approximate parameters from best_positions
                # For now, just use identity + scale as starting point
                pass
            
            for seed in [555, 666, 777]:
                if time.time() - START_TIME > DEADLINE - 15:
                    break
                
                rng = np.random.default_rng(seed)
                x_start = x0_rb + rng.normal(0, 0.05, 10)
                # Keep quaternion normalized
                x_start[0:4] = x_start[0:4] / np.linalg.norm(x_start[0:4])
                
                try:
                    res = minimize(_rb_energy_grad, x_start, method='L-BFGS-B',
                                   jac=True,
                                   options={'maxiter': 1000, 'gtol': 1e-10, 'maxls': 100})
                    qx, qy, qz, qw = res.x[0:4]
                    tx, ty, tz = res.x[4:7]
                    sc = res.x[7]
                    bA = 1.0 + res.x[8]
                    bB = 1.0 + res.x[9]
                    R = _rotation_matrix_from_quat([qw, qx, qy, qz])
                    A_pos = (A_base * bA) @ R.T * sc
                    B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
                    positions = np.zeros((N, 3))
                    positions[idx_A] = A_pos
                    positions[idx_B] = B_pos
                    if check_validity(positions):
                        energy = lj_energy_vectorized(positions)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                except Exception:
                    continue
        except Exception:
            pass
    
    # Phase 5: Basin hopping from icosahedral structure
    if time.time() - START_TIME < DEADLINE - 40:
        try:
            ico_positions = _make_icosahedral_structure()
            ico_positions = relax(ico_positions, maxiter=300, gtol=1e-7)
            
            if check_validity(ico_positions):
                ico_energy = lj_energy_vectorized(ico_positions)
                
                if ico_energy < best_energy:
                    best_energy = ico_energy
                    best_positions = ico_positions
                
                x0 = ico_positions.flatten()
                current_energy = _lj_energy_only(x0)
                
                for seed in [11, 22, 33, 44, 55]:
                    if time.time() - START_TIME > DEADLINE - 20:
                        break
                    
                    rng = np.random.default_rng(seed)
                    step_size = 0.18 + 0.04 * (seed % 3)
                    
                    bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 50, deadline_margin=18,
                                                 temp_start=0.25, temp_end=0.04)
                    
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_positions = bh_x.reshape(N, 3)
                        x0 = bh_x
                        current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 6: More basin hopping from the current best with larger perturbations
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            x0 = best_positions.flatten()
            current_energy = _lj_energy_only(x0)
            
            for seed in [999, 888, 777]:
                if time.time() - START_TIME > DEADLINE - 15:
                    break
                
                rng = np.random.default_rng(seed)
                # Larger step size to escape local minima
                bh_energy, bh_x = _basin_hop(x0, current_energy, rng, 0.30, 40, deadline_margin=12,
                                             temp_start=0.30, temp_end=0.05)
                
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_x.reshape(N, 3)
                    x0 = bh_x
                    current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 7: Final ultra-high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 8:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Phase 8: One more rigid-body optimization from the polished result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            # Use the polished positions to initialize a refined rigid-body search
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            scale = 2.0 ** (1.0/6.0)
            inv_scale = 1.0 / scale
            int_coords = np.round(base_coords * math.sqrt(2.0))
            all_even = np.all(int_coords % 2 == 0, axis=1)
            idx_A = np.where(all_even)[0]
            idx_B = np.where(~all_even)[0]
            A_base = base_coords[idx_A].copy()
            B_base = base_coords[idx_B].copy()
            
            # Estimate rotation from best_positions vs base
            # Simple approach: use SVD to find best-fit rotation
            if len(idx_A) > 3 and len(idx_B) > 3:
                A_est = best_positions[idx_A]
                B_est = best_positions[idx_B]
                
                # Center
                A_c = A_est - np.mean(A_est, axis=0)
                B_c = B_est - np.mean(B_est, axis=0)
                A_b_c = A_base - np.mean(A_base, axis=0)
                B_b_c = B_base - np.mean(B_base, axis=0)
                
                # SVD for rotation
                H = A_b_c.T @ A_c
                U, S, Vt = np.linalg.svd(H)
                R_est = Vt.T @ U.T
                if np.linalg.det(R_est) < 0:
                    Vt[2, :] *= -1
                    R_est = Vt.T @ U.T
                
                # Convert R to quaternion
                tr = np.trace(R_est)
                if tr > 0:
                    s = math.sqrt(tr + 1.0) * 2
                    qw = 0.25 * s
                    qx = (R_est[2,1] - R_est[1,2]) / s
                    qy = (R_est[0,2] - R_est[2,0]) / s
                    qz = (R_est[1,0] - R_est[0,1]) / s
                elif (R_est[0,0] > R_est[1,1]) and (R_est[0,0] > R_est[2,2]):
                    s = math.sqrt(1.0 + R_est[0,0] - R_est[1,1] - R_est[2,2]) * 2
                    qw = (R_est[2,1] - R_est[1,2]) / s
                    qx = 0.25 * s
                    qy = (R_est[0,1] + R_est[1,0]) / s
                    qz = (R_est[0,2] + R_est[2,0]) / s
                elif R_est[1,1] > R_est[2,2]:
                    s = math.sqrt(1.0 + R_est[1,1] - R_est[0,0] - R_est[2,2]) * 2
                    qw = (R_est[0,2] - R_est[2,0]) / s
                    qx = (R_est[0,1] + R_est[1,0]) / s
                    qy = 0.25 * s
                    qz = (R_est[1,2] + R_est[2,1]) / s
                else:
                    s = math.sqrt(1.0 + R_est[2,2] - R_est[0,0] - R_est[1,1]) * 2
                    qw = (R_est[1,0] - R_est[0,1]) / s
                    qx = (R_est[0,2] + R_est[2,0]) / s
                    qy = (R_est[1,2] + R_est[2,1]) / s
                    qz = 0.25 * s
                
                q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm
                
                # Estimate translation
                t_est = np.mean(B_est - B_c, axis=0) - R_est @ np.mean(B_b_c, axis=0)
                
                # Estimate scale and breathing
                dists_A = np.linalg.norm(A_est - np.mean(A_est, axis=0), axis=1)
                dists_A_base = np.linalg.norm(A_b_c, axis=1)
                scale_est_A = np.median(dists_A / np.maximum(dists_A_base, 1e-8))
                
                dists_B = np.linalg.norm(B_est - np.mean(B_est, axis=0), axis=1)
                dists_B_base = np.linalg.norm(B_b_c, axis=1)
                scale_est_B = np.median(dists_B / np.maximum(dists_B_base, 1e-8))
                
                x0_rb = np.array([qx, qy, qz, qw, t_est[0], t_est[1], t_est[2],
                                  scale_est_A, scale_est_A - 1.0, scale_est_B - 1.0])
                
                def _rb_eg(x_flat):
                    qx, qy, qz, qw = x_flat[0:4]
                    tx, ty, tz = x_flat[4:7]
                    sc = x_flat[7]
                    bA = 1.0 + x_flat[8]
                    bB = 1.0 + x_flat[9]
                    R = _rotation_matrix_from_quat([qw, qx, qy, qz])
                    A_pos = (A_base * bA) @ R.T * sc
                    B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
                    positions = np.zeros((N, 3))
                    positions[idx_A] = A_pos
                    positions[idx_B] = B_pos
                    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                    r2 = np.sum(diff * diff, axis=-1)
                    np.fill_diagonal(r2, 1.0)
                    r2 = np.maximum(r2, 1e-12)
                    inv_r2 = 1.0 / r2
                    inv_r6 = inv_r2 ** 3
                    inv_r12 = inv_r6 ** 2
                    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
                    inv_r14 = inv_r12 * inv_r2
                    inv_r8 = inv_r6 * inv_r2
                    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
                    grad_pos = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
                    grad_flat = np.zeros(10)
                    grad_flat[4:7] = np.sum(grad_pos[idx_B], axis=0)
                    for i in range(N):
                        if i in idx_A:
                            ai = np.where(idx_A == i)[0][0]
                            grad_flat[7] += np.dot(grad_pos[i], (A_base[ai] * bA) @ R.T)
                        else:
                            bi = np.where(idx_B == i)[0][0]
                            grad_flat[7] += np.dot(grad_pos[i], (B_base[bi] * bB) @ R.T)
                    for i in idx_A:
                        ai = np.where(idx_A == i)[0][0]
                        grad_flat[8] += np.dot(grad_pos[i], (A_base[ai]) @ R.T * sc)
                    for i in idx_B:
                        bi = np.where(idx_B == i)[0][0]
                        grad_flat[9] += np.dot(grad_pos[i], (B_base[bi]) @ R.T * sc)
                    # Finite diff for quaternion
                    eps = 1e-6
                    for k in range(4):
                        xp = x_flat.copy(); xp[k] += eps
                        xm = x_flat.copy(); xm[k] -= eps
                        def _e(x):
                            qx_, qy_, qz_, qw_ = x[0:4]
                            tx_, ty_, tz_ = x[4:7]
                            sc_ = x[7]
                            bA_ = 1.0 + x[8]
                            bB_ = 1.0 + x[9]
                            R_ = _rotation_matrix_from_quat([qw_, qx_, qy_, qz_])
                            Ap = (A_base * bA_) @ R_.T * sc_
                            Bp = (B_base * bB_) @ R_.T * sc_ + np.array([tx_, ty_, tz_])
                            pos = np.zeros((N, 3))
                            pos[idx_A] = Ap
                            pos[idx_B] = Bp
                            d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                            r2_ = np.sum(d*d, axis=-1)
                            np.fill_diagonal(r2_, 1.0)
                            r2_ = np.maximum(r2_, 1e-12)
                            ir2 = 1.0/r2_; ir6 = ir2**3; ir12 = ir6**2
                            return 4.0*np.sum(ir12-ir6)/2.0
                        grad_flat[k] = (_e(xp) - _e(xm)) / (2*eps)
                    return energy, grad_flat
                
                res = minimize(_rb_eg, x0_rb, method='L-BFGS-B', jac=True,
                              options={'maxiter': 3000, 'gtol': 1e-12, 'maxls': 100})
                qx, qy, qz, qw = res.x[0:4]
                tx, ty, tz = res.x[4:7]
                sc = res.x[7]
                bA = 1.0 + res.x[8]
                bB = 1.0 + res.x[9]
                R = _rotation_matrix_from_quat([qw, qx, qy, qz])
                A_pos = (A_base * bA) @ R.T * sc
                B_pos = (B_base * bB) @ R.T * sc + np.array([tx, ty, tz])
                positions = np.zeros((N, 3))
                positions[idx_A] = A_pos
                positions[idx_B] = B_pos
                if check_validity(positions):
                    energy = lj_energy_vectorized(positions)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        try:
            base_coords, _, _ = _get_fcc_shell_structure(N)
            positions = base_coords * (2.0 ** (1.0/6.0))
            if check_validity(positions):
                best_positions = positions
            else:
                best_positions = np.zeros((N, 3))
                for i in range(N):
                    best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    t1 = time.time()
    e = lj_energy(pos)
    print(f"Energy: {e}, Time: {t1-t0:.2f}s")
    print(f"Valid: {check_validity(pos)}")
