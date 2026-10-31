"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def _minimize_from_start(x0, max_iter=200, time_limit=2.0):
    """
    Run L-BFGS-B minimization from a starting point.
    Returns (energy, positions) or (inf, None) if failed/timeout.
    """
    def energy_func(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint strictly
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # r < 0.5
            return 1e10
        return _lj_energy(pos)

    start_time = time.time()
    
    # We cannot easily pass a time limit to scipy minimize directly without a wrapper that checks time.
    # However, L-BFGS-B is usually fast. We will limit max_iter to prevent runaway.
    # To be safe against the 120s total limit, we will do a quick check before running.
    
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        # Check if it converged or hit max iter
        # If it hit max iter, it might still be good, but we check the final energy.
        
        pos = result.x.reshape(-1, 3)
        
        # Final constraint check
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r = np.sqrt(np.min(r2))
        
        if min_r < 0.5:
            # If constraint violated, scale up or return invalid
            # Scaling up might increase energy significantly, so we mark as invalid for this start
            return 1e10, None
            
        energy = _lj_energy(pos)
        
        if time.time() - start_time > time_limit:
             # If it took too long, we might want to discard, but if it found a good energy, keep it.
             # For simplicity, we accept it if it's valid.
             pass
             
        return energy, pos
    except Exception:
        return 1e10, None


def _generate_fcc_start():
    """Generate FCC truncated octahedron geometry."""
    a = 1.0
    v1 = np.array([a/2, a/2, 0.0])
    v2 = np.array([a/2, 0.0, a/2])
    v3 = np.array([0.0, a/2, a/2])
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    
    points = np.array(points)
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order][:N]
    
    # Scale to approximate LJ equilibrium distance ~1.1225
    # Nearest neighbor in this FCC gen is a/sqrt(2) = 0.7071
    scale = 1.1225 / 0.7071
    return points * scale


def _generate_random_start(radius=5.0):
    """Generate random points in a sphere."""
    # Uniform in sphere
    u = np.random.rand(N, 3)
    theta = 2 * np.pi * u[:, 0]
    phi = np.arccos(2 * u[:, 1] - 1)
    r = radius * np.cbrt(u[:, 2])
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return np.column_stack((x, y, z))


def _generate_shell_start():
    """
    Generate a start with shell-like structure.
    1 center, 12 first shell, 15 second, 10 third? 
    1+12+15+10 = 38.
    This mimics icosahedral/FCC mixed packing.
    """
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms (icosahedral vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    # Normalize and scale to r=1.12
    for v in ico_verts[:12]:
        v = np.array(v)
        v = v / np.linalg.norm(v) * 1.12
        points.append(v)
        
    # Shell 2: 15 atoms
    # Just place them on a sphere of radius ~2.2, random angles but spaced
    # Or use a cubic face centers?
    # Let's just use random points on a sphere for simplicity in this start, 
    # relying on relaxation to fix it.
    for _ in range(15):
        u = np.random.rand(3)
        theta = 2 * np.pi * u[0]
        phi = np.arccos(2 * u[1] - 1)
        r = 2.2
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points.append(np.array([x, y, z]))
        
    # Shell 3: 10 atoms
    for _ in range(10):
        u = np.random.rand(3)
        theta = 2 * np.pi * u[0]
        phi = np.arccos(2 * u[1] - 1)
        r = 3.2
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points.append(np.array([x, y, z]))
        
    return np.array(points)


def construct():
    """
    Multi-start local optimization.
    Tries several diverse initial configurations and returns the best one.
    """
    start_time = time.time()
    time_limit_total = 110.0 # Leave 10s buffer
    
    # List of initial configurations
    starts = []
    
    # 1. FCC Truncated Octahedron (High quality start)
    starts.append(_generate_fcc_start())
    
    # 2. FCC with jitter
    fcc_jittered = _generate_fcc_start() + np.random.normal(0, 0.1, (N, 3))
    starts.append(fcc_jittered)
    
    # 3. Random Sphere
    starts.append(_generate_random_start(radius=5.0))
    
    # 4. Random Sphere (tighter)
    starts.append(_generate_random_start(radius=4.0))
    
    # 5. Shell Structure
    starts.append(_generate_shell_start())
    
    # 6. Another Random
    starts.append(_generate_random_start(radius=6.0))
    
    # 7. FCC with different scale (slightly compressed/expanded)
    fcc_comp = _generate_fcc_start() * 0.95
    starts.append(fcc_comp)
    fcc_exp = _generate_fcc_start() * 1.05
    starts.append(fcc_exp)
    
    best_energy = float('inf')
    best_positions = None
    
    for i, x0 in enumerate(starts):
        if time.time() - start_time > time_limit_total:
            break
            
        x0_flat = x0.flatten()
        energy, pos = _minimize_from_start(x0_flat)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            
        # Optional: If we find a very good energy (< -173.9), we can stop early?
        # The theoretical min is -173.928427.
        if best_energy < -173.928:
             break

    if best_positions is None:
        # Fallback to the first start's raw result if all failed
        # This shouldn't happen
        return _generate_fcc_start()
        
    return best_positions
# EVOLVE-BLOCK-END
