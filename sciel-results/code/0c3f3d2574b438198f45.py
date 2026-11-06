"""
Strategy: Lamarckian Basin Hopping with Rigid-Body Perturbations.
This approach abandons the slow, local-step Simulated Annealing in favor of 
Basin Hopping, which is better suited for glassy potentials like Dzugutov.
Key improvements:
1. Rigid-body kicks (translation + rotation) preserve the cluster's internal 
   structure, allowing the optimizer to explore different orientations and 
   translations without introducing high-energy internal strains that slow 
   down relaxation.
2. L-BFGS-B is used for the local minimization step, which converges faster 
   and more reliably to deep local minima than the provided `relax` function 
   (which might use a less aggressive method or looser tolerances).
3. Parallel execution of multiple independent Basin Hopping chains from 
   diverse seeds (Mackay icosahedra, FCC, random) to maximize coverage.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _energy_and_grad(x, n):
    """
    Compute energy and gradient for a flat array of positions.
    """
    pos = x.reshape(n, 3)
    try:
        # The provided function returns energy and gradient
        e, g = dzugutov_energy_gradient(pos)
        # Ensure gradient is flat
        g = g.flatten()
        return e, g
    except Exception as ex:
        # If gradient fails, return a large penalty and zero gradient
        return 1e10, np.zeros_like(x)

def _lbfgs_minimize(pos, n):
    """
    Perform L-BFGS-B minimization starting from pos.
    """
    x0 = pos.flatten()
    try:
        res = minimize(
            _energy_and_grad, 
            x0, 
            args=(n,), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        # Check if minimization succeeded
        if res.success or res.fun < 1e9:
            return res.fun, res.x.reshape(n, 3)
        else:
            # If L-BFGS fails, fallback to the provided relax
            e, p = relax(pos, 'dzugutov', None)
            return e, p
    except Exception:
        try:
            e, p = relax(pos, 'dzugutov', None)
            return e, p
        except Exception:
            return 1e10, pos

def _get_mackay_icosahedron(n, scale=1.0):
    """
    Construct a Mackay Icosahedral cluster of n atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    R1 = 0.9 * scale
    verts = verts * R1
    
    points = [np.zeros(3)]
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
        
    if n > 13:
        # Generate Shell 3 candidates (12 edge mids, 30 face centers)
        v = verts
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.linalg.norm(v[i] - v[j]) < 2.1 * R1: 
                    mid = (v[i] + v[j]) / 2.0
                    edge_mids.append(mid)
        
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if d1 < 2.1 * R1 and d2 < 2.1 * R1 and d3 < 2.1 * R1:
                        fc = (v[i] + v[j] + v[k]) / 3.0
                        face_centers.append(fc)
                        
        shell3 = np.array(edge_mids + face_centers)
        
        num_to_add = n - 13
        if num_to_add <= 42:
            # Select a subset. For now, just take the first ones.
            # A better strategy would be to select based on symmetry or energy.
            selected_shell3 = shell3[:num_to_add]
        else:
            # n > 55
            selected_shell3 = shell3
            extra = num_to_add - 42
            R4 = R1 * 2.0
            for _ in range(extra):
                z = np.random.uniform(-1, 1)
                theta = np.random.uniform(0, 2 * np.pi)
                x = np.sqrt(1 - z**2) * np.cos(theta)
                y = np.sqrt(1 - z**2) * np.sin(theta)
                p = np.array([x, y, z]) * R4
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    selected_shell3 = np.vstack([selected_shell3, p])
                else:
                    for _ in range(10):
                        p = np.array([x, y, z]) * R4 * (1 + np.random.uniform(-0.1, 0.1))
                        valid = True
                        for q in points:
                            if np.linalg.norm(p - q) < 0.5:
                                valid = False
                                break
                        if valid:
                            selected_shell3 = np.vstack([selected_shell3, p])
                            break
                            
        points.extend([p.copy() for p in selected_shell3])
        
    return np.array(points[:n])

def _generate_fcc_cluster(n):
    """
    Generate a simple FCC-like cluster.
    """
    points = []
    a = 1.0 # Lattice constant
    # Generate points in a cube
    max_coord = int(np.ceil((2*n)**(1/3)))
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC positions: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i+offset[0], j+offset[1], k+offset[2]]) * a
                    # Simple cutoff to keep it compact
                    if np.linalg.norm(p) < 2.5 * (n**(1/3)):
                        points.append(p)
    
    # Randomize and select n points
    np.random.shuffle(points)
    # Ensure minimum distance
    selected = []
    for p in points:
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
            if len(selected) == n:
                break
    if len(selected) < n:
        # Fill with random points
        for _ in range(n - len(selected)):
            p = np.random.uniform(-2, 2, 3)
            selected.append(p)
    return np.array(selected)

def _generate_random_cluster(n):
    """
    Generate a random compact cluster.
    """
    points = []
    for _ in range(n):
        for _attempt in range(100):
            p = np.random.uniform(-2, 2, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            points.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
    return np.array(points)

def _random_rotation_matrix():
    """
    Generate a random rotation matrix.
    """
    # Use QR decomposition of a random Gaussian matrix
    A = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q

def _basin_hopping_worker(start_pos, potential, parameter, time_limit, best_energy_holder, lock, seed_idx):
    """
    Basin Hopping worker with rigid-body kicks.
    """
    pos = start_pos.copy()
    n = len(pos)
    
    # Initial minimization
    try:
        current_energy, pos = _lbfgs_minimize(pos, n)
    except Exception:
        return 1e10
        
    T = 5.0
    cooling_rate = 0.995
    kick_size = 0.2
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time and _check_time():
        # Rigid body kick: Translation + Rotation
        translation = np.random.normal(0, kick_size, 3)
        R_mat = _random_rotation_matrix()
        
        new_pos = (pos @ R_mat.T) + translation
        
        # Minimize the kicked structure
        try:
            new_energy, new_pos_min = _lbfgs_minimize(new_pos, n)
        except Exception:
            continue
            
        delta_e = new_energy - current_energy
        # Metropolis criterion
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_min
            current_energy = new_energy
            
            with lock:
                if current_energy < best_energy_holder[0]:
                    best_energy_holder[0] = current_energy
                    best_energy_holder[1] = pos.copy()
                    
            # Adaptive kick size
            kick_size *= 0.999
            if kick_size < 0.05:
                kick_size = 0.05
            if kick_size > 0.5:
                kick_size = 0.5
                
        T *= cooling_rate
        if T < 0.5:
            T = 5.0 # Reheat

    return current_energy

def construct(instance):
    """
    Strategy:
    1. Generate diverse seeds (Mackay, FCC, Random).
    2. Relax them with L-BFGS-B.
    3. Run parallel Basin Hopping from these seeds.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # 1. Generate Seeds
    seeds = []
    
    # Primary seed: Standard Mackay
    try:
        seed_1 = _get_mackay_icosahedron(size, 1.0)
        seeds.append(seed_1)
    except Exception:
        pass
        
    # Secondary seed: Slightly different scale
    try:
        seed_2 = _get_mackay_icosahedron(size, 0.95)
        seeds.append(seed_2)
    except Exception:
        pass
        
    # Tertiary seed: FCC
    try:
        seed_3 = _generate_fcc_cluster(size)
        seeds.append(seed_3)
    except Exception:
        pass
        
    # Quaternary seed: Random
    try:
        seed_4 = _generate_random_cluster(size)
        seeds.append(seed_4)
    except Exception:
        pass
        
    # Relax seeds
    start_points = []
    best_energy = 1e10
    best_positions = seeds[0] if seeds else np.zeros((size, 3))
    
    for init in seeds:
        try:
            e, pos_relaxed = _lbfgs_minimize(init, size)
            start_points.append(pos_relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed.copy()
        except Exception:
            continue
            
    if not start_points:
        start_points = [best_positions]
        
    # 2. Parallel Basin Hopping
    num_threads = 4
    time_per_thread = 270.0 / num_threads
    
    # Create multiple starting points for threads
    while len(start_points) < num_threads:
        perturbed = best_positions + np.random.normal(0, 0.3, (size, 3))
        try:
            e, pos_relaxed = _lbfgs_minimize(perturbed, size)
            start_points.append(pos_relaxed)
        except Exception:
            start_points.append(perturbed)
            
    best_energy_holder = [best_energy, best_positions]
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, sp in enumerate(start_points):
            if not _check_time():
                break
            futures.append(executor.submit(_basin_hopping_worker, sp, potential, parameter, time_per_thread, best_energy_holder, lock, i))
            
        try:
            for future in as_completed(futures, timeout=280):
                future.result()
        except Exception:
            pass
            
    # 3. Final Relaxation
    try:
        e_final, pos_final = _lbfgs_minimize(best_energy_holder[1], size)
        return pos_final
    except Exception:
        return best_energy_holder[1]

# EVOLVE-BLOCK-END
