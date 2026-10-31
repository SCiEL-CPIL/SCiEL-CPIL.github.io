import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
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
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def lj_gradient(positions):
    """Compute gradient of LJ energy with respect to positions."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6)
            grad[i] -= coeff * rij
            grad[j] += coeff * rij
    return grad

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def _get_fcc_shell_structure():
    """
    Get the FCC truncated octahedron shell structure for 38 atoms.
    Returns:
        shell_centers: list of lists of (i,j,k) FCC lattice indices for each shell
        The FCC lattice has points where i+j+k is even, scaled by 1/sqrt(2).
    """
    a = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((r2, (i, j, k)))
    
    points.sort(key=lambda p: p[0])
    
    # Group by distance (shell)
    shells = []
    current_shell = []
    prev_r2 = -1
    for r2, idx in points:
        if current_shell and abs(r2 - prev_r2) > 1e-10:
            shells.append(current_shell)
            current_shell = []
        current_shell.append(idx)
        prev_r2 = r2
    if current_shell:
        shells.append(current_shell)
    
    # Take shells until we have 38 atoms
    selected_shells = []
    count = 0
    for shell in shells:
        if count + len(shell) <= 38:
            selected_shells.append(shell)
            count += len(shell)
        else:
            # Take only what we need from this shell
            selected_shells.append(shell[:38 - count])
            break
    
    return selected_shells

def _make_fcc_truncated_octahedron(params):
    """
    Build FCC truncated octahedron from symmetry-adapted parameters.
    params: array of shell radii [r0, r1, r2, ...] for each shell.
    The FCC lattice structure is fixed; only radial distances are optimized.
    """
    shells = _get_fcc_shell_structure()
    a = 1.0 / math.sqrt(2.0)
    positions = []
    for s, shell in enumerate(shells):
        r_shell = params[s]
        for (i, j, k) in shell:
            x, y, z = i * a, j * a, k * a
            r = math.sqrt(x*x + y*y + z*z)
            if r > 1e-10:
                scale = r_shell / r
            else:
                scale = 1.0
            positions.append([x * scale, y * scale, z * scale])
    return np.array(positions, dtype=float)

def _symmetry_adapted_energy(params):
    """Energy as function of shell radii."""
    positions = _make_fcc_truncated_octahedron(params)
    return lj_energy_vectorized(positions)

def _symmetry_adapted_energy_and_grad(params):
    """Energy and gradient with respect to shell radii using finite differences."""
    e0 = _symmetry_adapted_energy(params)
    n_params = len(params)
    grad = np.zeros(n_params)
    h = 1e-7
    for i in range(n_params):
        params_p = params.copy()
        params_p[i] += h
        e_p = _symmetry_adapted_energy(params_p)
        params_m = params.copy()
        params_m[i] -= h
        e_m = _symmetry_adapted_energy(params_m)
        grad[i] = (e_p - e_m) / (2 * h)
    return e0, grad

def _optimize_symmetry_adapted(initial_params, maxiter=2000):
    """Optimize shell radii using L-BFGS-B."""
    def func(p):
        return _symmetry_adapted_energy(p)
    
    def func_and_grad(p):
        e, g = _symmetry_adapted_energy_and_grad(p)
        return e, g
    
    try:
        result = minimize(func_and_grad, initial_params, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 100})
        return result.x
    except Exception:
        return initial_params

def _make_initial_clusters():
    """Create diverse initial cluster configurations."""
    clusters = []
    
    # 1. FCC truncated octahedron (exact lattice)
    a = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((r2, x, y, z))
    points.sort(key=lambda p: p[0])
    fcc = np.array([[p[1], p[2], p[3]] for p in points[:N]])
    if check_validity(fcc):
        clusters.append(fcc)
    
    # 2. Perturbed FCC
    rng = np.random.default_rng(42)
    fcc_pert = fcc + rng.normal(0, 0.08, fcc.shape)
    if check_validity(fcc_pert):
        clusters.append(fcc_pert)
    
    # 3. Icosahedral
    phi = (1 + math.sqrt(5)) / 2.0
    r_ico = 1.0 / math.sqrt(1 + phi * phi)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r_ico
    ico_points = [np.array([0.0, 0.0, 0.0])]
    ico_points.extend([v * 1.12 for v in ico_verts])
    rng2 = np.random.default_rng(123)
    for _ in range(N - len(ico_points)):
        u = rng2.uniform(0, 1)
        v = rng2.uniform(0, 1)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(1 - z*z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        ico_points.append(np.array([x, y, z]) * 1.6)
    ico = np.array(ico_points, dtype=float)
    if check_validity(ico):
        clusters.append(ico)
    
    # 4. Random cluster
    rng3 = np.random.default_rng(999)
    random_cluster = rng3.uniform(-2, 2, (N, 3))
    if check_validity(random_cluster):
        clusters.append(random_cluster)
    
    return clusters

def _relax(positions, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with analytic gradient."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    def grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_gradient(pos).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(energy_func, x0, jac=grad_func, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _perturb(positions, magnitude, rng):
    """Add random perturbation to positions."""
    perturbed = positions + rng.normal(0, magnitude, positions.shape)
    return perturbed

def _basin_hopping(positions, n_iterations=20, perturb_scale=0.2, rng=None):
    """
    Basin hopping: perturb, relax, accept if lower energy.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = positions.copy()
    current_e = lj_energy_vectorized(current)
    best = current.copy()
    best_e = current_e
    
    for i in range(n_iterations):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Perturb
        perturbed = _perturb(current, perturb_scale, rng)
        
        # Relax
        relaxed = _relax(perturbed, maxiter=300, gtol=1e-8)
        
        if not check_validity(relaxed):
            continue
        
        e_new = lj_energy_vectorized(relaxed)
        
        # Metropolis acceptance (always accept if lower, sometimes accept if higher)
        if e_new < current_e:
            current = relaxed
            current_e = e_new
            if e_new < best_e:
                best = relaxed
                best_e = e_new
        else:
            # Metropolis criterion
            delta = e_new - current_e
            # Use a simple acceptance probability
            if rng.random() < math.exp(-delta / 0.5):
                current = relaxed
                current_e = e_new
    
    return best, best_e

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy 1: Symmetry-adapted FCC optimization
    try:
        shells = _get_fcc_shell_structure()
        n_shells = len(shells)
        # Initial guess: exact FCC lattice distances
        a = 1.0 / math.sqrt(2.0)
        initial_params = np.array([1.0] * n_shells)
        
        # Compute actual distances for initialization
        fcc_points = []
        for s, shell in enumerate(shells):
            for (i, j, k) in shell:
                x, y, z = i * a, j * a, k * a
                fcc_points.append((s, x, y, z))
        
        # Get unique distances per shell
        shell_dists = {}
        for s, x, y, z in fcc_points:
            r = math.sqrt(x*x + y*y + z*z)
            if s not in shell_dists:
                shell_dists[s] = r
        
        initial_params = np.array([shell_dists.get(s, 1.0) for s in range(n_shells)])
        
        optimized_params = _optimize_symmetry_adapted(initial_params, maxiter=2000)
        sym_positions = _make_fcc_truncated_octahedron(optimized_params)
        
        if check_validity(sym_positions):
            e_sym = lj_energy_vectorized(sym_positions)
            if e_sym < best_energy:
                best_energy = e_sym
                best_positions = sym_positions
    except Exception:
        pass
    
    # Strategy 2: Multi-start with basin hopping
    clusters = _make_initial_clusters()
    rng = np.random.default_rng(2024)
    
    for i, cluster in enumerate(clusters):
        if time.time() - START_TIME > DEADLINE - 20:
            break
        
        try:
            # First relax
            relaxed = _relax(cluster, maxiter=500, gtol=1e-10)
            if not check_validity(relaxed):
                continue
            
            e_relaxed = lj_energy_vectorized(relaxed)
            
            # Basin hopping from this starting point
            bh_positions, bh_energy = _basin_hopping(relaxed, n_iterations=15, 
                                                      perturb_scale=0.15, rng=rng)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions
            elif e_relaxed < best_energy:
                best_energy = e_relaxed
                best_positions = relaxed
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish on best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = _relax(best_positions, maxiter=2000, gtol=1e-12)
            if check_validity(polished):
                e_polished = lj_energy_vectorized(polished)
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_positions = polished
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        try:
            fcc = _make_initial_clusters()[0]
            if check_validity(fcc):
                best_positions = fcc
            else:
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
