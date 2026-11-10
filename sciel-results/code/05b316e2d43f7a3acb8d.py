"""Morse cluster global minima: Fragment-based annealing with rigid-body superclusters and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285  # Safety margin for 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
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
    """Generate a rigid icosahedral fragment."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(vertices, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    verts = verts * spacing * 0.85
    verts -= np.mean(verts, axis=0)
    if n_atoms <= len(verts):
        return verts[:n_atoms]
    extra = np.random.default_rng(7).normal(0, 0.2, (n_atoms - len(verts), 3))
    return np.vstack([verts, extra])

def _generate_tetrahedral_fragment(n_atoms=4, spacing=1.0):
    """Generate a rigid tetrahedral fragment."""
    vertices = [
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]
    verts = np.array(vertices, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    verts = verts * spacing * 0.75
    verts -= np.mean(verts, axis=0)
    if n_atoms <= len(verts):
        return verts[:n_atoms]
    extra = np.random.default_rng(7).normal(0, 0.2, (n_atoms - len(verts), 3))
    return np.vstack([verts, extra])

def _generate_fcc(n, spacing=1.0):
    """Generate FCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.35)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i * a, j * a, k * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                points.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                points.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    
    points = np.array(points, dtype=float)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_bcc(n, spacing=1.0):
    """Generate BCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.35)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i * a, j * a, k * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    
    points = np.array(points, dtype=float)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_hcp(n, spacing=1.0):
    """Generate HCP lattice points, clipped to sphere."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_coord = int(np.ceil(n ** 0.35)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                points.append([x, y, z])
                x_b = a * (i + 0.5 * (j % 2)) + a / 2.0
                y_b = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z_b = c * (k + 0.5)
                points.append([x_b, y_b, z_b])
    
    points = np.array(points, dtype=float)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _generate_icosahedral_shell(n, spacing=1.0):
    """Generate an icosahedral-like shell structure."""
    points = []
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * (n ** (1.0/3.0)) * 0.6
    
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points, dtype=float)

def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(42)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.65 * spacing
    attempts = 0
    max_attempts = 20000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.sqrt(np.sum((arr - p) ** 2, axis=1))
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _local_minimize(pos, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    try:
        result = minimize(
            lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy_and_grad(new_pos, rho)[0]
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy_and_grad(pos, rho)[0], pos

def _assemble_from_fragments(fragments, centers, rotations):
    """Assemble full configuration from rigid fragments."""
    pos = []
    for frag, center, R in zip(fragments, centers, rotations):
        pos.append(frag @ R.T + center)
    return np.vstack(pos)

def _fragment_annealing(n, rho, rng, n_frags=12, frag_size=10, max_iter=200):
    """Kirkpatrick-Strauss annealing on rigid-body fragment lattice."""
    # Generate fragments
    fragments = []
    for i in range(n_frags):
        if i == 0:
            frag = _generate_icosahedral_fragment(min(frag_size, 12))
        elif i % 3 == 0:
            frag = _generate_tetrahedral_fragment(min(frag_size, 4))
        else:
            frag = _generate_icosahedral_fragment(min(frag_size, 12))
        fragments.append(frag)
    
    # Initial positions: random sphere
    radius = n ** (1.0/3.0) * 0.8
    centers = rng.uniform(-radius, radius, (n_frags, 3))
    # Random rotations as quaternions
    quats = []
    for i in range(n_frags):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        quats.append(q)
    
    def _energy_and_grad(params):
        """Compute energy and gradient for fragment parameters."""
        centers = params[:n_frags*3].reshape(n_frags, 3)
        quats = params[n_frags*3:].reshape(n_frags, 4)
        # Normalize quaternions
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / norms
        
        rotations = [_rotation_matrix_from_quaternion(q) for q in quats]
        pos = _assemble_from_fragments(fragments, centers, rotations)
        
        # Check validity
        if not _is_valid(pos, min_dist=0.45):
            return 1e6, np.zeros_like(params)
        
        e, grad_pos = _morse_energy_and_grad(pos, rho)
        
        # Back-propagate gradient to fragment parameters
        grad_params = np.zeros_like(params)
        for i in range(n_frags):
            grad_params[:n_frags*3][i*3:(i+1)*3] = grad_pos[i*frag_size:(i+1)*frag_size].sum(axis=0)
            # Rotation gradient (approximate via finite difference)
            eps = 1e-4
            for k in range(4):
                q_plus = quats[i].copy()
                q_minus = quats[i].copy()
                q_plus[k] += eps
                q_minus[k] -= eps
                q_plus /= np.linalg.norm(q_plus)
                q_minus /= np.linalg.norm(q_minus)
                
                R_plus = _rotation_matrix_from_quaternion(q_plus)
                R_minus = _rotation_matrix_from_quaternion(q_minus)
                
                pos_plus = _assemble_from_fragments(fragments, centers, rotations)
                pos_plus[i*frag_size:(i+1)*frag_size] = fragments[i] @ R_plus.T + centers[i]
                
                pos_minus = _assemble_from_fragments(fragments, centers, rotations)
                pos_minus[i*frag_size:(i+1)*frag_size] = fragments[i] @ R_minus.T + centers[i]
                
                e_plus = _morse_energy_and_grad(pos_plus, rho)[0]
                e_minus = _morse_energy_and_grad(pos_minus, rho)[0]
                
                grad_params[n_frags*3 + i*4 + k] = (e_plus - e_minus) / (2 * eps)
        
        return e, grad_params
    
    # Simulated annealing
    T = 1.0
    cooling = 0.95
    current_params = np.concatenate([centers.flatten(), np.array(quats).flatten()])
    current_energy, _ = _energy_and_grad(current_params)
    
    best_params = current_params.copy()
    best_energy = current_energy
    
    for step in range(max_iter):
        if _check_deadline():
            break
        
        # Perturb
        new_params = current_params + rng.normal(0, 0.1, size=current_params.shape)
        new_energy, _ = _energy_and_grad(new_params)
        
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
            current_params = new_params
            current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_params = current_params.copy()
        
        T *= cooling
    
    # Assemble and relax
    centers = best_params[:n_frags*3].reshape(n_frags, 3)
    quats = best_params[n_frags*3:].reshape(n_frags, 4)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms
    rotations = [_rotation_matrix_from_quaternion(q) for q in quats]
    pos = _assemble_from_fragments(fragments, centers, rotations)
    
    e, p = _local_minimize(pos, rho, max_iter=300)
    if _is_valid(p):
        return e, p
    return best_energy, pos

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
    """Find global minimum of Morse cluster using fragment annealing and basin hopping."""
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Fragment-based annealing
    if size >= 20:
        n_frags = min(12, size // 8)
        frag_size = max(4, size // n_frags)
        try:
            e_frag, pos_frag = _fragment_annealing(size, rho, rng, n_frags=n_frags, frag_size=frag_size, max_iter=150)
            if _is_valid(pos_frag) and e_frag < best_energy:
                best_energy = e_frag
                best_pos = pos_frag.copy()
        except Exception:
            pass
    
    # Strategy 2: Diverse crystalline and shell starts
    initial_configs = []
    
    try:
        fcc = _generate_fcc(size)
        if _is_valid(fcc):
            initial_configs.append(fcc)
    except Exception:
        pass
    
    try:
        bcc = _generate_bcc(size)
        if _is_valid(bcc):
            initial_configs.append(bcc)
    except Exception:
        pass
        
    try:
        hcp = _generate_hcp(size)
        if _is_valid(hcp):
            initial_configs.append(hcp)
    except Exception:
        pass
        
    try:
        shell = _generate_icosahedral_shell(size)
        if _is_valid(shell):
            initial_configs.append(shell)
    except Exception:
        pass
        
    for _ in range(2):
        try:
            rand = _generate_random_compact(size, rng=rng)
            if _is_valid(rand):
                initial_configs.append(rand)
        except Exception:
            pass
            
    if not initial_configs:
        pos = rng.uniform(-2, 2, (size, 3))
        initial_configs.append(pos)
        
    # Parallel relaxation of initial configs
    max_workers = min(4, len(initial_configs))
    
    if max_workers > 1 and len(initial_configs) > 1:
        tasks = [(config, rho, 300) for config in initial_configs]
        
        def _relax_worker(args):
            pos, rho, max_iter = args
            if _check_deadline():
                return None
            try:
                e, p = _local_minimize(pos, rho, max_iter)
                if _is_valid(p):
                    return e, p
            except Exception:
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_relax_worker, tasks)
            for res in results:
                if res is not None:
                    e, p = res
                    if e < best_energy:
                        best_energy = e
                        best_pos = p.copy()
    else:
        for config in initial_configs:
            if _check_deadline():
                break
            try:
                e, p = _local_minimize(config, rho, max_iter=300)
                if _is_valid(p) and e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                pass

    if best_pos is None:
        best_pos = initial_configs[0]
        best_energy = _morse_energy_and_grad(best_pos, rho)[0]
        
    # Basin hopping on the best result if time permits
    if time.time() < _DEADLINE - 15:
        try:
            e_bh, pos_bh = _basin_hop(best_pos, rho, rng, max_steps=40, perturb_scale=0.25)
            if e_bh < best_energy and _is_valid(pos_bh):
                best_energy = e_bh
                best_pos = pos_bh.copy()
        except Exception:
            pass
            
    # Final refinement
    if time.time() < _DEADLINE - 2:
        try:
            e_final, pos_final = _local_minimize(best_pos, rho, max_iter=500)
            if e_final < best_energy and _is_valid(pos_final):
                best_energy = e_final
                best_pos = pos_final
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
