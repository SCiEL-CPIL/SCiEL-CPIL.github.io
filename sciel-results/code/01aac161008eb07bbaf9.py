# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _energy_morse(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r)) (exp(rho(1-r))-2)."""
    n = pos.shape[0]
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask diagonal and lower triangle (only i<j)
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    e = np.exp(rho * (1.0 - r))
    return np.sum(e * (e - 2.0))


def _min_distance(pos):
    """Compute minimum interatomic distance."""
    if pos.shape[0] < 2:
        return np.inf
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def _gradient_morse(pos, rho):
    """Compute gradient of Morse energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    # dE/dr_ij = rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    e = np.exp(rho * (1.0 - dists))
    dE_dr = rho * e * (2.0 * e - 2.0)
    # grad_i = -sum_{j!=i} dE_dr_ij * (r_i - r_j)/r_ij
    # Using symmetry: force on i from j is -dE_dr * (ri-rj)/r
    grad = np.zeros_like(pos)
    # dE_dr has shape (n,n), diff has shape (n,n,3)
    # grad[i] += sum_j dE_dr[i,j] * (ri - rj)/r_ij  ... wait sign
    # E = sum_{i<j} f(r_ij), dE/dri = sum_{j!=i, j<i or j>i} ... 
    # Actually: dE/dri = -sum_{j != i} f'(r_ij) * (ri - rj)/r_ij
    # where f'(r) = rho*exp(rho(1-r))*(2*exp(rho(1-r))-2)
    # So grad[i] = -sum_j f'(r_ij) * (ri-rj)/r_ij
    # = -sum_j dE_dr[i,j] * diff[i,j] / r_ij
    mask = ~np.eye(n, dtype=bool)
    inv_r = np.where(mask, 1.0 / dists, 0.0)
    # factor[i,j] = -dE_dr[i,j] * inv_r[i,j]
    factor = -dE_dr * inv_r
    factor = factor * mask  # zero out diagonal
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _bfgs_local(pos, rho, max_iter=200, tol=1e-10):
    """Local optimization using L-BFGS-B or BFGS via scipy."""
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        return _energy_morse(p, rho)
    
    def jac(x):
        p = x.reshape(-1, 3)
        g = _gradient_morse(p, rho)
        return g.ravel()
    
    x0 = pos.ravel()
    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return res.fun, res.x.reshape(-1, 3)


def _make_initial_guesses(n, rho, rng):
    """Generate multiple initial configurations: FCC, BCC, HCP-like, random."""
    guesses = []
    spacing = 1.0
    
    # 1. FCC lattice
    a = spacing * 2 ** 0.5  # nearest neighbor = a/sqrt(2) = 1
    # Generate FCC points in a cube
    max_coord = a * (n ** (1/3) + 2)
    fcc_points = []
    L = int(2 * max_coord / a) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                # FCC basis
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= max_coord:
                        fcc_points.append(p)
    fcc_points = np.array(fcc_points)
    if len(fcc_points) >= n:
        # Select n points closest to center
        centers = np.linalg.norm(fcc_points, axis=1)
        idx = np.argsort(centers)[:n]
        guesses.append(fcc_points[idx].copy())
    
    # 2. BCC lattice
    b = spacing / (np.sqrt(3) / 2)  # nearest neighbor in BCC is b*sqrt(3)/2 = 1
    bcc_points = []
    max_coord = b * (n ** (1/3) + 2)
    L = int(2 * max_coord / b) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = b * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= max_coord:
                        bcc_points.append(p)
    bcc_points = np.array(bcc_points)
    if len(bcc_points) >= n:
        centers = np.linalg.norm(bcc_points, axis=1)
        idx = np.argsort(centers)[:n]
        guesses.append(bcc_points[idx].copy())
    
    # 3. Simple cubic
    sc_points = []
    max_coord = spacing * (n ** (1/3) + 2)
    L = int(2 * max_coord / spacing) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                p = spacing * np.array([i, j, k])
                if np.linalg.norm(p) <= max_coord:
                    sc_points.append(p)
    sc_points = np.array(sc_points)
    if len(sc_points) >= n:
        centers = np.linalg.norm(sc_points, axis=1)
        idx = np.argsort(centers)[:n]
        guesses.append(sc_points[idx].copy())
    
    # 4. HCP-like lattice (hexagonal close packed)
    hcp_points = []
    # HCP: layers of hexagonal packing, ABAB stacking
    layer_spacing = spacing * np.sqrt(2/3)  # distance between layers
    hex_spacing = spacing
    max_layers = int(n ** (1/3) * 2) + 3
    for layer in range(-max_layers, max_layers + 1):
        z = layer * layer_spacing
        # Hexagonal grid in this layer
        for i in range(-max_layers, max_layers + 1):
            for j in range(-max_layers, max_layers + 1):
                # Hexagonal offset
                x = hex_spacing * (i + 0.5 * (j % 2))
                y = hex_spacing * np.sqrt(3) / 2 * j
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_coord:
                    hcp_points.append(p)
    hcp_points = np.array(hcp_points)
    if len(hcp_points) >= n:
        centers = np.linalg.norm(hcp_points, axis=1)
        idx = np.argsort(centers)[:n]
        guesses.append(hcp_points[idx].copy())
    
    # 5. Random compact (multiple attempts)
    radius = spacing * n ** (1/3) * 0.8
    for _ in range(5):
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 200000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.51 * spacing for q in pts):
                    pts.append(p)
            attempts += 1
        if len(pts) >= n:
            guesses.append(np.array(pts[:n]))
    
    # 6. Shell-like: place atoms on concentric shells
    # First atom at origin, then shells
    shell_pts = [np.zeros(3)]
    shell_r = 1.0
    while len(shell_pts) < n:
        # Add points on a sphere of radius shell_r
        num_on_shell = min(n - len(shell_pts), int(12 * shell_r**2))
        if num_on_shell < 2:
            num_on_shell = 2
        # Fibonacci sphere
        golden = (1 + np.sqrt(5)) / 2
        for m in range(num_on_shell):
            z = 1 - 2 * (m + 0.5) / num_on_shell
            r_z = np.sqrt(1 - z*z)
            theta = golden * m * 2 * np.pi
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            shell_pts.append(p)
        shell_r += 1.8  # spacing between shells
        if len(shell_pts) >= n:
            break
    if len(shell_pts) >= n:
        guesses.append(np.array(shell_pts[:n]))
    
    # 7. Random with varying density
    for density_factor in [0.6, 1.0, 1.4]:
        radius = spacing * n ** (1/3) * density_factor
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 100000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.51 * spacing for q in pts):
                    pts.append(p)
            attempts += 1
        if len(pts) >= n:
            guesses.append(np.array(pts[:n]))
    
    return guesses


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = 330.0  # must return before 330s wall-clock
    safety_margin = 5.0
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 7919)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial guesses
    guesses = _make_initial_guesses(size, rho, rng)
    
    # Relax each guess
    for i, guess in enumerate(guesses):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        try:
            # Use the provided relax function first
            energy, positions = relax(guess, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
        
        # Also try our own BFGS for potentially better local min
        if time.time() - start_time > deadline - safety_margin:
            break
        try:
            e2, p2 = _bfgs_local(guess, rho, max_iter=300)
            # Check minimum distance constraint
            min_d = _min_distance(p2)
            if min_d >= 0.5 - 1e-6:
                if e2 < best_energy:
                    best_energy = e2
                    best_pos = p2.copy()
        except Exception:
            pass
    
    # If we have time, do simulated annealing / perturbation-based search from best
    if best_pos is not None and time.time() - start_time < deadline - safety_margin - 10:
        # Simulated annealing style search
        current_pos = best_pos.copy()
        current_energy = best_energy
        T = 0.5  # initial temperature
        T_min = 0.01
        cooling = 0.95
        max_sa_steps = 500
        
        for step in range(max_sa_steps):
            if time.time() - start_time > deadline - safety_margin:
                break
            if T < T_min:
                break
            
            # Perturb: move a random subset of atoms
            n_perturb = max(1, size // 4)
            idx = rng.choice(size, n_perturb, replace=False)
            noise_scale = 0.2 * (1 + T)
            perturbed = current_pos.copy()
            perturbed[idx] += rng.normal(0, noise_scale, (n_perturb, 3))
            
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d < 0.5 - 1e-6:
                    continue
                
                delta_e = energy - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    current_pos = positions.copy()
                    current_energy = energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
            except Exception:
                pass
            
            T *= cooling
        
        # Try several perturbations from best configuration with varying scales
        for trial in range(15):
            if time.time() - start_time > deadline - safety_margin:
                break
            # Vary noise scale
            noise_scale = 0.2 + 0.1 * trial
            perturbed = best_pos + rng.normal(0, noise_scale, best_pos.shape)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
        
        # Try larger perturbations
        for trial in range(10):
            if time.time() - start_time > deadline - safety_margin:
                break
            noise_scale = 0.8 + 0.2 * trial
            perturbed = best_pos + rng.normal(0, noise_scale, best_pos.shape)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
        
        # Try kick-out: remove one atom and re-add
        for trial in range(8):
            if time.time() - start_time > deadline - safety_margin:
                break
            idx = rng.integers(0, size)
            # Move this atom far away
            perturbed = best_pos.copy()
            perturbed[idx] = best_pos[idx] + rng.normal(0, 1.5, 3)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
    
    if best_pos is None:
        # Fallback: return a valid configuration
        # Place atoms on a line with spacing 1.0
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i, 0] = i * 1.0
        best_pos -= best_pos.mean(axis=0)
    
    return best_pos
# EVOLVE-BLOCK-END
