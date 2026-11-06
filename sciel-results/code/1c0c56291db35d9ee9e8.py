import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
CUBE_HALF = 0.5
DEADLINE = 110  # seconds, safety margin before 120s limit

def compute_radius(centres: np.ndarray) -> float:
    """Compute the common radius r for given centres."""
    if len(centres) == 0:
        return 0.0
    max_c = np.abs(centres).max(axis=1)
    wall_min = CUBE_HALF - max_c
    r_wall = wall_min.min()
    
    n = len(centres)
    if n < 2:
        return r_wall
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(r_wall, min_pair)

def initial_grid(n: int = N) -> np.ndarray:
    """Create initial grid arrangement."""
    # 5x5x4 grid = 100 spheres
    # Cube side 1, centers in [-0.5, 0.5]
    # x, y: 5 points, z: 4 points
    nx, ny, nz = 5, 5, 4
    x = np.linspace(-0.4, 0.4, nx)
    y = np.linspace(-0.4, 0.4, ny)
    z = np.linspace(-0.375, 0.375, nz) # Adjust z spacing to be slightly smaller to fit 4 in 1.0? 
    # Actually, let's just use a simple grid and let optimizer fix it.
    # Better: use a jittered grid to avoid symmetry traps.
    
    # Simple grid
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - CUBE_HALF
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:n].astype(float)
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    c += rng.normal(0, 0.01, c.shape)
    c = np.clip(c, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    
    return c

def repulsive_energy(centres: np.ndarray) -> float:
    """
    Calculate the repulsive energy.
    We want to minimize this to maximize the minimum distance.
    """
    c = centres.reshape(-1, 3)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    
    # Inverse square or inverse fourth power
    # Inverse fourth power is steeper, helps resolve close pairs
    pair_energy = np.sum(1.0 / (dists ** 4))
    
    # Wall repulsion
    # Distance to nearest wall for each sphere
    max_c = np.abs(c).max(axis=1)
    wall_dists = CUBE_HALF - max_c
    # Avoid division by zero
    wall_dists = np.maximum(wall_dists, 1e-6)
    wall_energy = np.sum(1.0 / (wall_dists ** 4))
    
    return pair_energy + wall_energy

def gradient_repulsive_energy(centres: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of the repulsive energy.
    E = sum_{i<j} d_ij^-4 + sum_i w_i^-4
    dE/dd_ij = -4 d_ij^-5
    d(d_ij)/dc_i = (c_i - c_j) / d_ij
    dE/dc_i = sum_{j!=i} -4 d_ij^-5 * (c_i - c_j)/d_ij = -4 sum_{j!=i} (c_i - c_j) / d_ij^6
    """
    c = centres.reshape(-1, 3)
    n = len(c)
    grad = np.zeros_like(c)
    
    if n < 2:
        return grad.flatten()
    
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    
    # d_ij^6
    d6 = dists ** 6
    
    # Gradient of pair energy
    # grad_i = -4 * sum_j (c_i - c_j) / d_ij^6
    # Vectorized:
    # diff[i,j] = c_i - c_j
    # This is memory heavy for N=100 (100x100x3 = 30k floats, fine)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    d6_masked = np.where(mask, d6, 1.0)
    
    # term = diff / d6_masked[..., np.newaxis]
    term = diff / d6_masked[..., np.newaxis]
    
    # Sum over j (axis 1)
    grad_pair = -4.0 * np.sum(term, axis=1)
    
    # Wall gradient
    # w_i = min_k (0.5 - |c_ik|)
    # Actually, we used max_c = max(|c_i|). So w_i = 0.5 - max(|c_i|).
    # d(w_i)/dc_ik = -sign(c_ik) if k is argmax(|c_i|), else 0.
    # dE/dw_i = -4 w_i^-5
    # dE/dc_ik = dE/dw_i * dw_i/dc_ik = 4 w_i^-5 * sign(c_ik)
    
    max_c = np.abs(c).max(axis=1)
    wall_dists = CUBE_HALF - max_c
    wall_dists_safe = np.maximum(wall_dists, 1e-6)
    
    # dE/dw = -4 * w^-5
    dE_dw = -4.0 / (wall_dists_safe ** 5)
    
    # dw/dc_ik
    # For each sphere i, find the axis k that gives max |c_ik|
    max_axes = np.argmax(np.abs(c), axis=1)
    
    grad_wall = np.zeros_like(c)
    for i in range(n):
        k = max_axes[i]
        sign_val = np.sign(c[i, k])
        if sign_val == 0:
            sign_val = 1.0
        # dw/dc_ik = -sign(c_ik)
        # dE/dc_ik = dE_dw * (-sign) = 4 * w^-5 * sign
        grad_wall[i, k] = dE_dw[i] * (-sign_val) * -1.0 # Wait.
        # Let's re-derive.
        # w = 0.5 - |x| (assuming x is max coord)
        # dw/dx = -sign(x)
        # E_wall = w^-4
        # dE_wall/dw = -4 w^-5
        # dE_wall/dx = -4 w^-5 * (-sign(x)) = 4 w^-5 sign(x)
        grad_wall[i, k] = 4.0 / (wall_dists_safe[i] ** 5) * sign_val
        
    grad = grad_pair + grad_wall
    
    return grad.flatten()

def fea_optimize(centres: np.ndarray, target_r: float, 
                 iters: int = 2000, lr: float = 0.01) -> np.ndarray:
    """
    Force-exclusion algorithm to push spheres apart while staying in cube.
    """
    c = centres.copy().astype(float)
    n = len(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        # Check if we're already at target
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
        if min(min_pair, wall_min) >= target_r:
            break
        
        # Build force vectors
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (only for pairs closer than 2*target_r)
        # Vectorized pairwise forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists_mat, np.inf)
        
        # Only consider pairs closer than 2*target_r
        mask = dists_mat < 2.0 * target_r
        # Force magnitude: (2*r - d) / 2
        # Direction: diff / d
        # Force on i from j: (2*r - d_ij)/2 * (c_i - c_j)/d_ij
        
        # To avoid division by zero, use safe dists
        safe_dists = np.where(dists_mat > 1e-10, dists_mat, 1.0)
        
        # Magnitude
        mag = np.where(mask, (2.0 * target_r - dists_mat) / 2.0, 0.0)
        mag = np.where(mag < 0, 0.0, mag)
        
        # Unit vector
        unit_vec = diff / safe_dists[..., np.newaxis]
        
        # Force contribution
        force_contrib = mag[..., np.newaxis] * unit_vec
        
        # Sum forces for each i
        forces = np.sum(force_contrib, axis=1)
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                # Positive wall
                if c[i, k] + target_r > CUBE_HALF:
                    forces[i, k] -= (c[i, k] + target_r - CUBE_HALF)
                # Negative wall
                if c[i, k] - target_r < -CUBE_HALF:
                    forces[i, k] += (-CUBE_HALF - (c[i, k] - target_r))
        
        c += lr * forces
        
        # Clip to cube
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c

def construct() -> np.ndarray:
    """
    Construct centres for 100 equal spheres in unit cube.
    Uses grid initialization + L-BFGS-B optimization + FEA refinement.
    """
    t_start = time.time()
    
    # Start with grid
    c = initial_grid(N)
    
    # Phase 1: L-BFGS-B to minimize repulsive energy
    # This spreads the spheres out
    if time.time() < DEADLINE - 20:
        res = minimize(
            repulsive_energy, 
            c.flatten(), 
            method='L-BFGS-B',
            jac=gradient_repulsive_energy,
            bounds=[(-CUBE_HALF + 0.01, CUBE_HALF - 0.01)] * (N * 3),
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-6}
        )
        c = res.x.reshape(N, 3)
    
    # Ensure c is valid
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    r_current = compute_radius(c)
    
    # Phase 2: FEA to tighten the packing
    # Try to increase the radius
    # We can do a binary search or incremental increase
    # Let's try incremental increase
    r_target = r_current + 0.001
    max_r = 0.112 # Best known is ~0.11138, try to beat it
    
    while r_target < max_r and time.time() < DEADLINE - 5:
        c_new = fea_optimize(c, r_target, iters=500, lr=0.005)
        r_new = compute_radius(c_new)
        
        if r_new >= r_target * 0.99: # Accept if we got close
            c = c_new
            r_current = r_new
            r_target += 0.0005
        else:
            # Failed to reach target, maybe try smaller step or break
            r_target += 0.0001
            # If we can't even make small progress, break
            if r_target - r_current > 0.005:
                break
                
    # Final check
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c

# Keep the original function names and signatures as required
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program had: compute_radius, initial_grid, fea_optimize, local_search, construct
# I have redefined compute_radius, initial_grid, fea_optimize, construct.
# I should add a dummy local_search to match the interface if strictly required, 
# but the prompt says "Keep the function names and signatures of the seed program" 
# and "do not rename functions". It implies I should provide the same set of functions.
# However, the "construct" function is the entry point. The other functions are helpers.
# The prompt says "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program above".
# The entry point is construct().
# The other functions are internal. I can modify them.
# But to be safe, I will keep the names.

def local_search(centres: np.ndarray, iters: int = 50, step_size: float = 0.02) -> np.ndarray:
    """
    Simple local search: try small random perturbations and keep if radius improves.
    (Kept for interface compatibility, but not used in main flow if time is tight)
    """
    c = centres.copy().astype(float)
    n = len(c)
    
    def radius_of(arr):
        dists = cdist(arr, arr)
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(arr).max(axis=1)).min()
        return min(min_pair, wall_min)
    
    best_r = radius_of(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        i = np.random.randint(n)
        original = c[i].copy()
        
        for _ in range(5):
            delta = np.random.randn(3) * step_size
            c[i] = original + delta
            c[i] = np.clip(c[i], -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
            
            r = radius_of(c)
            if r > best_r:
                best_r = r
            else:
                c[i] = original
                break
    
    return c
