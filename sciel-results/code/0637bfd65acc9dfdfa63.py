"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    n = len(centres)
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair:
                min_pair = d
    
    return min(min_pair / 2.0, min_wall)


def _soft_min_objective(flat, beta=50.0):
    """
    Soft-min objective: we want to MAXIMIZE the minimum of:
    - pairwise distances / 2
    - wall distances
    
    So we MINIMIZE the negative soft-min.
    Using log-sum-exp for numerical stability.
    """
    centres = flat.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    
    # Pairwise distances / 2
    pair_vals = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centres[i] - centres[j])
            pair_vals.append(d / 2.0)
    pair_vals = np.array(pair_vals)
    
    # Combine all values
    all_vals = np.concatenate([wall_dists.ravel(), pair_vals])
    
    # Soft-min via log-sum-exp: softmin(x) = -logsumexp(-beta*x)
    # = -(-1/beta) * log(sum(exp(-beta*x)))
    # For numerical stability:
    max_neg_beta_x = np.max(-beta * all_vals)
    log_sum_exp = max_neg_beta_x + np.log(np.sum(np.exp(-beta * all_vals - max_neg_beta_x)))
    soft_min = -log_sum_exp / beta
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min


def _make_bcc_init():
    """
    Create a body-centered cubic (BCC) inspired initialization.
    BCC has better packing density than simple cubic.
    For 100 spheres, we use a 5x5x4 arrangement with BCC offsets.
    """
    centres = []
    
    # Use a 5x5x4 grid with BCC-like offsets
    # Simple approach: start with a cubic grid and add small random perturbations
    m = 5  # 5x5x4 = 100
    spacing = 1.0 / m
    offset = spacing / 2.0
    
    for iz in range(4):
        for iy in range(5):
            for ix in range(5):
                x = -0.5 + offset + ix * spacing
                y = -0.5 + offset + iy * spacing
                z = -0.5 + (iz + 0.5) * (1.0/4)
                centres.append([x, y, z])
    
    centres = np.array(centres[:N])
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centres += rng.uniform(-0.01, 0.01, size=centres.shape)
    
    # Clip to stay inside cube
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_grid_init():
    """Create a simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _local_optimize(centres, beta=50.0, max_iter=200):
    """
    Local optimization using L-BFGS-B on the soft-min objective.
    """
    flat = centres.flatten()
    
    def objective(flat_x):
        return _soft_min_objective(flat_x, beta)
    
    def gradient(flat_x):
        # Numerical gradient (expensive but simple)
        eps = 1e-6
        grad = np.zeros_like(flat_x)
        for i in range(len(flat_x)):
            flat_plus = flat_x.copy()
            flat_minus = flat_x.copy()
            flat_plus[i] += eps
            flat_minus[i] -= eps
            grad[i] = (objective(flat_plus) - objective(flat_minus)) / (2 * eps)
        return grad
    
    result = minimize(
        objective,
        flat,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return result.x.reshape(N, 3)


def _simulated_annealing(centres, n_iterations=5000, T_start=0.05, T_end=0.001):
    """
    Simulated annealing to escape local minima.
    """
    rng = np.random.default_rng(123)
    current = centres.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for i in range(n_iterations):
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        # Random perturbation
        idx = rng.integers(0, N)
        perturbation = rng.normal(0, T, size=3)
        new_centre = current[idx].copy()
        new_centre += perturbation
        
        # Keep inside cube
        new_centre = np.clip(new_centre, -0.499, 0.499)
        
        new_centres = current.copy()
        new_centres[idx] = new_centre
        
        new_r = _compute_radius(new_centres)
        
        # Accept if better or with probability exp((new_r - current_r)/T)
        if new_r >= current_r or rng.random() < math.exp((new_r - current_r) / max(T, 1e-10)):
            current = new_centres
            current_r = new_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with BCC-inspired grid
    centres = _make_bcc_init()
    current_r = _compute_radius(centres)
    best_centres = centres.copy()
    best_r = current_r
    
    # Phase 1: Simulated annealing to find a good basin
    if time.time() < deadline - 60.0:
        sa_centres = _simulated_annealing(centres, n_iterations=3000, T_start=0.03, T_end=0.0005)
        sa_r = _compute_radius(sa_centres)
        if sa_r > best_r:
            best_centres = sa_centres
            best_r = sa_r
    
    # Phase 2: Local optimization with L-BFGS-B
    if time.time() < deadline - 20.0:
        opt_centres = _local_optimize(best_centres, beta=50.0, max_iter=100)
        opt_r = _compute_radius(opt_centres)
        if opt_r > best_r:
            best_centres = opt_centres
            best_r = opt_r
    
    # Phase 3: Another round of SA + polish if time permits
    if time.time() < deadline - 10.0:
        sa2_centres = _simulated_annealing(best_centres, n_iterations=2000, T_start=0.01, T_end=0.0001)
        sa2_r = _compute_radius(sa2_centres)
        if sa2_r > best_r:
            best_centres = sa2_centres
            best_r = sa2_r
    
    # Phase 4: Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        opt2_centres = _local_optimize(best_centres, beta=100.0, max_iter=50)
        opt2_r = _compute_radius(opt2_centres)
        if opt2_r > best_r:
            best_centres = opt2_centres
            best_r = opt2_r
    
    # Ensure all centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    
    return best_centres
# EVOLVE-BLOCK-END
