"""Morse cluster global minima: Fragment-based annealing with rigid-body superclusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # Safety margin for 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute all pair differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    # Avoid division by zero
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    # dE/dr = exp_term * (2*exp_term - 2) * (-rho)
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    # Gradient accumulation
    grad = np.zeros_like(pos)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad

def _is_valid(pos, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    return bool(np.min(r) >= min_dist)

def _rotation_matrix_from_quaternion(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R

def _generate_icosahedral_fragment(n_atoms=12, spacing=1.0):
    """Generate a rigid icosahedral fragment (12 vertices of regular icosahedron)."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of regular icosahedron
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(vertices, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    # Scale to desired spacing
    verts = verts * spacing * 0.8
    # Center
    verts -= np.mean(verts, axis=0)
    return verts[:n_atoms]

def _generate_tetrahedral_fragment(n_atoms=4, spacing=1.0):
    """Generate a rigid tetrahedral fragment (4 vertices)."""
    vertices = [
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]
    verts = np.array(vertices, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    verts = verts * spacing * 0.7
    verts -= np.mean(verts, axis=0)
    return verts[:n_atoms]

def _generate_octahedral_fragment(n_atoms=6, spacing=1.0):
    """Generate a rigid octahedral fragment (6 vertices)."""
    vertices = [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ]
    verts = np.array(vertices, dtype=float)
    verts = verts * spacing * 0.8
    verts -= np.mean(verts, axis=0)
    return verts[:n_atoms]

def _assemble_from_fragments(fragments, positions, rotations, rng):
    """Assemble full atomic configuration from rigid fragments."""
    all_atoms = []
    for frag, pos, rot in zip(fragments, positions, rotations):
        R = _rotation_matrix_from_quaternion(rot)
        rotated = (R @ frag.T).T
        translated = rotated + pos
        all_atoms.append(translated)
    return np.vstack(all_atoms)

def _fragment_annealing(n_atoms, rho, rng, max_steps=200, T_start=1.0, T_end=0.01):
    """Kirkpatrick-Strauss style annealing on rigid-body fragment positions."""
    # Choose fragment size based on total atoms
    if n_atoms <= 15:
        frag_size = min(4, n_atoms)
        n_frags = n_atoms // frag_size
        if n_atoms % frag_size != 0:
            n_frags += 1
            frag_size = max(2, n_atoms // n_frags)
    elif n_atoms <= 30:
        frag_size = 6
        n_frags = n_atoms // frag_size
        if n_atoms % frag_size != 0:
            n_frags += 1
    else:
        frag_size = 12
        n_frags = n_atoms // frag_size
        if n_atoms % frag_size != 0:
            n_frags += 1
    
    # Generate fragments
    fragments = []
    for i in range(n_frags):
        if i == n_frags - 1 and n_atoms % frag_size != 0:
            actual_size = n_atoms % frag_size
        else:
            actual_size = frag_size
        
        # Choose fragment type based on index for diversity
        if i % 3 == 0:
            frag = _generate_icosahedral_fragment(actual_size)
        elif i % 3 == 1:
            frag = _generate_tetrahedral_fragment(actual_size)
        else:
            frag = _generate_octahedral_fragment(actual_size)
        
        # If fragment is larger than actual_size, truncate
        if len(frag) > actual_size:
            frag = frag[:actual_size]
        # If fragment is smaller, pad with random points
        while len(frag) < actual_size:
            frag = np.vstack([frag, rng.normal(0, 0.3, 3)])
        
        fragments.append(frag)
    
    # Initial positions: spread fragments on a sphere
    radius = n_atoms ** (1/3) * 0.6
    positions = []
    rotations = []
    for i in range(n_frags):
        # Fibonacci sphere for initial positions
        z = 1 - 2 * (i + 0.5) / n_frags
        r_xy = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / 1.6180339887
        pos = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * radius
        positions.append(pos)
        # Random initial rotation (quaternion)
        q = rng.normal(0, 1, 4)
        q /= np.linalg.norm(q)
        rotations.append(q)
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    
    # Annealing schedule
    best_energy = np.inf
    best_positions = positions.copy()
    best_rotations = rotations.copy()
    
    for step in range(max_steps):
        if _check_deadline():
            break
        
        # Temperature schedule: geometric cooling
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Current configuration
        current_config = _assemble_from_fragments(fragments, positions, rotations, rng)
        current_energy = _morse_energy_and_grad(current_config, rho)[0]
        
        # If current is best, update
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = positions.copy()
            best_rotations = rotations.copy()
        
        # Propose move: pick random fragment, perturb its position and rotation
        frag_idx = rng.integers(0, n_frags)
        
        # New position: Gaussian perturbation
        new_pos = positions[frag_idx].copy()
        new_pos += rng.normal(0, T * 0.5, 3)
        
        # New rotation: small angular perturbation
        new_rot = rotations[frag_idx].copy()
        # Add small random rotation
        delta_q = rng.normal(0, T * 0.3, 4)
        delta_q /= np.linalg.norm(delta_q)
        # Quaternion multiplication: q_new = q_delta * q_old
        w1, x1, y1, z1 = delta_q
        w2, x2, y2, z2 = new_rot
        new_rot = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        new_rot /= np.linalg.norm(new_rot)
        
        # Evaluate new configuration
        new_positions = positions.copy()
        new_rotations = rotations.copy()
        new_positions[frag_idx] = new_pos
        new_rotations[frag_idx] = new_rot
        
        new_config = _assemble_from_fragments(fragments, new_positions, new_rotations, rng)
        new_energy = _morse_energy_and_grad(new_config, rho)[0]
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
            positions = new_positions
            rotations = new_rotations
    
    # Return best configuration found
    best_config = _assemble_from_fragments(fragments, best_positions, best_rotations, rng)
    best_energy = _morse_energy_and_grad(best_config, rho)[0]
    return best_energy, best_config

def _local_minimize(pos, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    try:
        result = minimize(
            lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy_and_grad(new_pos, rho)[0]
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy_and_grad(pos, rho)[0], pos

def _fragment_annealing_worker(args):
    """Worker for parallel fragment annealing."""
    n_atoms, rho, seed, max_steps = args
    if _check_deadline():
        return None
    try:
        rng = np.random.default_rng(seed)
        e, p = _fragment_annealing(n_atoms, rho, rng, max_steps=max_steps)
        if _is_valid(p):
            return e, p
    except Exception:
        pass
    return None

def _basin_hop(pos, rho, rng, max_steps=50, perturb_scale=0.3):
    """Basin hopping: random perturbations + local relaxation."""
    current_energy, current_pos = _local_minimize(pos, rho, max_iter=200)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(max_steps):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, perturb_scale, pos.shape)
        perturbed = current_pos + perturbation
        perturbed_energy, perturbed_pos = _local_minimize(perturbed, rho, max_iter=100)
        
        if _is_valid(perturbed_pos) and perturbed_energy < best_energy:
            best_energy = perturbed_energy
            best_pos = perturbed_pos.copy()
        
        if perturbed_energy < current_energy:
            current_energy = perturbed_energy
            current_pos = perturbed_pos
        else:
            delta_e = perturbed_energy - current_energy
            if rng.random() < np.exp(-delta_e / max(0.01, abs(current_energy) * 0.1)):
                current_energy = perturbed_energy
                current_pos = perturbed_pos
    
    return best_energy, best_pos

def construct(instance):
    """Find global minimum of Morse cluster using fragment-based annealing."""
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Fragment-based annealing (primary)
    # Run multiple independent fragment annealing chains in parallel
    n_chains = 3
    max_steps = 150
    
    if time.time() < _DEADLINE - 30:
        tasks = [(size, rho, size + i, max_steps) for i in range(n_chains)]
        with ThreadPoolExecutor(max_workers=min(4, n_chains)) as executor:
            results = list(executor.map(_fragment_annealing_worker, tasks))
            for res in results:
                if res is not None:
                    e, p = res
                    if e < best_energy:
                        best_energy = e
                        best_pos = p.copy()
    
    # Strategy 2: Local minimization from fragment annealing results
    if best_pos is not None and time.time() < _DEADLINE - 15:
        try:
            e, p = _local_minimize(best_pos, rho, max_iter=300)
            if _is_valid(p) and e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    # Strategy 3: Basin hopping from best result
    if best_pos is not None and time.time() < _DEADLINE - 10:
        try:
            e_bh, pos_bh = _basin_hop(best_pos, rho, rng, max_steps=20, perturb_scale=0.2)
            if e_bh < best_energy and _is_valid(pos_bh):
                best_energy = e_bh
                best_pos = pos_bh.copy()
        except Exception:
            pass
    
    # Fallback: if fragment annealing failed, use crystalline starts
    if best_pos is None:
        # Generate FCC start
        try:
            points = []
            a = 1.0
            max_coord = int(np.ceil(size ** 0.5)) + 2
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    for k in range(-max_coord, max_coord + 1):
                        points.append(np.array([i * a, j * a, k * a]))
                        points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                        points.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                        points.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
            points = np.array(points)
            com = np.mean(points, axis=0)
            points -= com
            dists = np.linalg.norm(points, axis=1)
            order = np.argsort(dists)
            points = points[order[:size]]
            if _is_valid(points):
                e, p = _local_minimize(points, rho, max_iter=300)
                if _is_valid(p):
                    best_energy = e
                    best_pos = p.copy()
        except Exception:
            pass
    
    # Final refinement
    if best_pos is not None and time.time() < _DEADLINE - 2:
        try:
            e_final, pos_final = _local_minimize(best_pos, rho, max_iter=500)
            if e_final < best_energy and _is_valid(pos_final):
                best_energy = e_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    # Ultimate fallback
    if best_pos is None:
        best_pos = rng.uniform(-2, 2, (size, 3))
    
    return best_pos
# EVOLVE-BLOCK-END
