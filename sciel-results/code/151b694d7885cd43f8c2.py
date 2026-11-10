"""Morse cluster global minima: Basin Hopping with crystalline initializations."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

try:
    from autoresearch_suite.chem_clusters import relax as _cc_relax, morse_energy_gradient
    HAS_CHEM_CLUSTERS = True
except ImportError:
    HAS_CHEM_CLUSTERS = False

# Deadline guard: evaluator kills at 330s, use 300s as hard limit with margin
TOTAL_DEADLINE = 295.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster using vectorized ops."""
    n = positions.shape[0]
    # Compute all pair distances
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)  # avoid self-interaction
    r = np.where(d > 0, d, 1.0)
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 3)
    dvdr = rho * ex * (2.0 * ex - 3.0)
    
    # Vectorized gradient computation
    # factor[i,j] = -dvdr[i,j] / r[i,j]
    factor = np.zeros_like(r)
    mask = r > 1e-10
    factor[mask] = -dvdr[mask] / r[mask]
    
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    # = sum_j factor[i,j] * pos[i] - sum_j factor[i,j] * pos[j]
    # = pos[i] * sum_j factor[i,j] - (factor @ pos)[i]
    row_sum = factor.sum(axis=1)  # shape (n,)
    grad = positions * row_sum[:, np.newaxis] - factor @ positions
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for speed in some contexts)."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    return 0.5 * np.sum(v)


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    # BCC lattice points
    a = 1.0  # lattice parameter
    positions = []
    # Generate BCC points in a sphere
    max_coord = int(np.ceil((n * (a/2)**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # Corner atom
                p1 = np.array([ix * a, iy * a, iz * a])
                # Body center atom
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a])
                candidates.append(p1)
                candidates.append(p2)
    
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    
    # Center the cluster
    positions -= positions.mean(axis=0)
    return positions


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    a = 1.0
    max_coord = int(np.ceil((n * (a/2)**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC: corner + face centers
                p1 = np.array([ix * a, iy * a, iz * a])
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a])
                p3 = np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a])
                p4 = np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a])
                candidates.extend([p1, p2, p3, p4])
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_hcp(n):
    """Generate HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    
    candidates = []
    max_coord = int(np.ceil((n * a**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                if layer % 2 == 0:
                    x = ix * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                else:
                    x = (ix + 0.5) * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                candidates.append(np.array([x, y, z]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_icosahedral(n):
    """Generate icosahedral shell structure - known ground state for short-range Morse."""
    # Icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    ico_vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices[0])  # normalize to unit sphere
    
    positions = [np.zeros(3)]  # center atom
    
    # Shell 1: icosahedron (12 atoms)
    shell1_radius = 1.0
    for v in ico_vertices:
        if len(positions) >= n:
            break
        positions.append(shell1_radius * v)
    
    # Shell 2: 30 atoms at edge midpoints of icosahedron
    if len(positions) < n:
        # 30 edges of icosahedron
        ico_edges = [
            (0,1),(0,2),(0,4),(0,6),(1,3),(1,5),(1,7),
            (2,3),(2,4),(2,8),(3,5),(3,9),(4,6),(4,8),
            (4,10),(5,7),(5,9),(5,11),(6,8),(6,10),
            (7,9),(7,11),(8,10),(8,12),(9,11),(9,13),
            (10,12),(10,14),(11,13),(11,14)
        ]
        # Correct icosahedron edges (30 total)
        ico_edges = [
            (0,1),(0,2),(0,4),(0,6),(1,3),(1,5),(1,7),
            (2,3),(2,4),(2,8),(3,5),(3,9),(4,6),(4,8),
            (4,10),(5,7),(5,9),(5,11),(6,8),(6,10),
            (7,9),(7,11),(8,10),(8,12),(9,11),(9,13),
            (10,12),(10,14),(11,13),(11,14)
        ]
        # Use proper icosahedron edge list
        ico_edges = [
            (0,1),(0,2),(0,4),(0,6),(1,3),(1,5),(1,7),
            (2,3),(2,4),(2,8),(3,5),(3,9),(4,6),(4,8),
            (4,10),(5,7),(5,9),(5,11),(6,8),(6,10),
            (7,9),(7,11),(8,10),(8,12),(9,11),(9,13),
            (10,12),(10,14),(11,13),(11,14)
        ]
        # Actually let me just compute edges from the vertex list
        # Two vertices are connected if their dot product is close to -1/t
        target_dot = -1.0 / t
        shell2_radius = np.sqrt(2.0 * (1.0 + 1.0/t))  # edge midpoint distance from center
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                dot = np.dot(ico_vertices[i], ico_vertices[j])
                if abs(dot - target_dot) < 1e-6:
                    mid = ico_vertices[i] + ico_vertices[j]
                    mid /= np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        for v in edge_mids:
            if len(positions) >= n:
                break
            positions.append(shell2_radius * v)
    
    # Shell 3: 60 atoms at face centers
    if len(positions) < n:
        # 20 faces of icosahedron
        # Find faces (triangles where all 3 vertices are mutually connected)
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d_ij = np.dot(ico_vertices[i], ico_vertices[j])
                    d_jk = np.dot(ico_vertices[j], ico_vertices[k])
                    d_ik = np.dot(ico_vertices[i], ico_vertices[k])
                    if (abs(d_ij - target_dot) < 1e-6 and 
                        abs(d_jk - target_dot) < 1e-6 and
                        abs(d_ik - target_dot) < 1e-6):
                        center = ico_vertices[i] + ico_vertices[j] + ico_vertices[k]
                        center /= np.linalg.norm(center)
                        face_centers.append(center)
        
        shell3_radius = 1.0 / np.cos(np.arcsin(1.0/np.sqrt(5.0)))  # approximate
        # Better: face center distance = 1/cos(theta) where theta is half-angle
        # For regular icosahedron, face center is at distance ~1.051 from center
        shell3_radius = 1.0515
        for v in face_centers:
            if len(positions) >= n:
                break
            positions.append(shell3_radius * v)
    
    # If still need more atoms, add random points on outer shell
    if len(positions) < n:
        rng = np.random.default_rng(42)
        for _ in range(n - len(positions)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = 1.8 * np.array([np.sin(phi) * np.cos(theta),
                                np.sin(phi) * np.sin(theta),
                                np.cos(phi)])
            positions.append(p)
    
    result = np.array(positions[:n])
    result -= result.mean(axis=0)
    return result


def _generate_shell(n):
    """Generate concentric shell structure."""
    positions = [np.zeros(3)]
    shell_radii = [1.0, 1.8, 2.5, 3.2, 4.0, 4.8, 5.5]
    rng = np.random.default_rng(42)
    
    for radius in shell_radii:
        if len(positions) >= n:
            break
        # Fibonacci sphere for even distribution
        remaining = n - len(positions)
        # Place points on sphere surface
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = radius * np.array([np.sin(phi) * np.cos(theta),
                                    np.sin(phi) * np.sin(theta),
                                    np.cos(phi)])
            positions.append(p)
    
    return np.array(positions[:n])


def _relax_lbfgs(positions, rho, maxiter=200):
    """Relax positions using L-BFGS-B."""
    if HAS_CHEM_CLUSTERS:
        try:
            result = _cc_relax(positions, 'morse', rho)
            if result is not None and isinstance(result, np.ndarray) and result.shape == positions.shape:
                e = _morse_energy_only(result, rho)
                return result, e
        except Exception:
            pass
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    result = minimize(fun_and_grad, positions.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def _basin_hopping(positions, rho, n_iter=50, step_size=0.3, seed=0, time_limit=None):
    """Basin hopping with Metropolis acceptance: local minimize + random perturbation."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_energy = _morse_energy_only(best_pos, rho)
    
    # First, relax the initial position
    best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=100)
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Temperature schedule for Metropolis
    T0 = 0.5
    T_final = 0.01
    n_total = max(n_iter, 1)
    
    for i in range(n_iter):
        # Check time limit
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Temperature cooling
        T = T0 * (T_final / T0) ** (i / n_total)
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Local minimization
        new_pos, new_energy = _relax_lbfgs(new_pos, rho, maxiter=50)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
        
        # Update best
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        # Slowly reduce step size
        if i % 10 == 0 and i > 0:
            step_size *= 0.92
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check time budget
    elapsed = time.time() - START_TIME
    if elapsed > TOTAL_DEADLINE - 5:
        # Return a reasonable default if out of time
        return _generate_bcc(size)
    
    # Generate diverse initial structures - icosahedral first for short-range
    generators = [
        ('ico', _generate_icosahedral),
        ('fcc', _generate_fcc),
        ('bcc', _generate_bcc),
        ('hcp', _generate_hcp),
        ('shell', _generate_shell),
    ]
    
    best_pos = None
    best_energy = np.inf
    
    # Allocate time per generator based on remaining budget
    deadline = START_TIME + TOTAL_DEADLINE
    
    for idx, (name, gen_func) in enumerate(generators):
        elapsed = time.time() - START_TIME
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        # Allocate more iterations to early generators (especially icosahedral)
        if name == 'ico':
            n_iter = 35
            step_size = 0.25
        elif idx == 0:
            n_iter = 25
            step_size = 0.25
        else:
            n_iter = 15
            step_size = 0.3
        
        try:
            init_pos = gen_func(size)
            # Quick relax first
            relaxed_pos, relaxed_energy = _relax_lbfgs(init_pos, rho, maxiter=80)
            
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
            
            # Basin hopping from this starting point with time limit
            time_limit = deadline - 10
            bh_pos, bh_energy = _basin_hopping(relaxed_pos, rho, n_iter=n_iter, 
                                                step_size=step_size, seed=size + idx * 100,
                                                time_limit=time_limit)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # If we have time, do more basin hopping from the best found
    elapsed = time.time() - START_TIME
    remaining = deadline - time.time()
    if remaining > 20 and best_pos is not None:
        try:
            time_limit = deadline - 5
            bh_pos, bh_energy = _basin_hopping(best_pos, rho, n_iter=40, step_size=0.3, 
                                                seed=size + 1000, time_limit=time_limit)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # Final relaxation
    if best_pos is not None:
        elapsed = time.time() - START_TIME
        if time.time() < deadline - 5:
            try:
                best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=150)
            except Exception:
                pass
    
    # Ensure minimum distance constraint (0.5 apart)
    if best_pos is not None:
        # Vectorized check and fix
        d = squareform(pdist(best_pos))
        np.fill_diagonal(d, 1.0)
        for i in range(size):
            for j in range(i + 1, size):
                if d[i, j] < 0.5 and d[i, j] > 1e-10:
                    direction = (best_pos[i] - best_pos[j]) / d[i, j]
                    shift = (0.5 - d[i, j]) / 2.0 + 0.01
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
    
    if best_pos is None:
        best_pos = _generate_bcc(size)
    
    return best_pos
