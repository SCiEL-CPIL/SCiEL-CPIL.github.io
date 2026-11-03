"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, cdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses a smooth least-squares formulation with a fixed target radius r,
    iteratively increasing r until infeasible, then binary searching.
    """
    n = N
    
    def get_initial_guess():
        """Generate a good initial hexagonal packing."""
        # For 300 circles, use a hexagonal lattice
        # Optimal spacing for hexagonal packing in unit disk
        # Try different spacings and pick the one that fits most points
        best_points = None
        best_count = 0
        
        for spacing in np.linspace(0.05, 0.12, 20):
            k = int(1.0 / spacing) + 2
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 0.95:
                        rows.append((x, y))
            points = np.array(rows) if rows else np.empty((0, 2))
            if len(points) > best_count:
                best_count = len(points)
                best_points = points
        
        if best_points is None or len(best_points) < n:
            # Fallback: random points
            angles = np.random.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n)) * 0.9
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Shuffle and take first n
        idx = np.random.permutation(len(best_points))[:n]
        return best_points[idx]
    
    x0 = get_initial_guess().flatten()
    
    def compute_radius(centres):
        """Compute the achievable radius for a given configuration."""
        dists = pdist(centres)
        min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
        boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    def residuals_fixed_r(x, r_target, smoothing=10.0):
        """
        Smooth residuals for fixed target radius r_target.
        We want d_ij >= 2*r_target and |c_i| <= 1 - r_target.
        
        Residual for pair (i,j): softplus-like penalty when d_ij < 2*r_target
        Residual for point i: softplus-like penalty when |c_i| > 1 - r_target
        
        Using a smooth approximation:
        For pair: max(0, 2*r - d) approximated by (1/smoothing)*log(1 + exp(smoothing*(2*r - d)))
        For boundary: max(0, |c| - (1-r)) approximated similarly
        """
        centres = x.reshape(n, 2)
        
        # Pairwise distances
        dists = pdist(centres)
        
        # Boundary distances
        norms = np.linalg.norm(centres, axis=1)
        boundary_viol = norms - (1.0 - r_target)  # want this <= 0
        
        # Smooth max(0, violation)
        pair_viol = 2.0 * r_target - dists  # want this <= 0
        
        # Smooth approximation of max(0, v) = (1/s) * log(1 + exp(s*v))
        # Clip to avoid overflow
        pair_viol_clipped = np.clip(pair_viol, -50, 50)
        boundary_viol_clipped = np.clip(boundary_viol, -50, 50)
        
        pair_resid = (1.0 / smoothing) * np.log(1.0 + np.exp(smoothing * pair_viol_clipped))
        boundary_resid = (1.0 / smoothing) * np.log(1.0 + np.exp(smoothing * boundary_viol_clipped))
        
        return np.concatenate([pair_resid, boundary_resid])
    
    def check_feasibility(r_target, x_init, max_nfev=2000):
        """Check if configuration is feasible for given r_target."""
        try:
            result = least_squares(
                lambda x: residuals_fixed_r(x, r_target, smoothing=15.0),
                x_init,
                method='trf',
                max_nfev=max_nfev,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            centres = result.x.reshape(n, 2)
            r_achieved = compute_radius(centres)
            feasible = r_achieved >= r_target - 1e-9
            return feasible, centres, r_achieved
        except Exception:
            return False, x_init.reshape(n, 2), 0.0
    
    # Start with a reasonable initial radius estimate
    r_init = compute_radius(x0.reshape(n, 2))
    print(f"Initial radius: {r_init:.10f}")
    
    # First, try to improve the initial configuration at its current radius
    feasible, centres, r_achieved = check_feasibility(r_init, x0, max_nfev=3000)
    if feasible:
        x0 = centres.flatten()
        r_init = r_achieved
        print(f"Improved initial radius: {r_init:.10f}")
    
    # Binary search for the maximum feasible radius
    r_low = r_init
    r_high = r_init * 1.5  # Upper bound guess
    
    # First, find a feasible upper bound
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        feasible, centres, r_ach = check_feasibility(r_mid, x0, max_nfev=1500)
        if feasible:
            r_low = r_mid
            x0 = centres.flatten()
        else:
            r_high = r_mid
        if r_high - r_low < 1e-7:
            break
    
    print(f"Binary search range: [{r_low:.10f}, {r_high:.10f}]")
    
    # Fine-tune: try to push r_low up a bit more
    best_r = r_low
    best_x = x0.copy()
    
    for _ in range(5):
        r_try = best_r * 1.001
        feasible, centres, r_ach = check_feasibility(r_try, best_x, max_nfev=2000)
        if feasible:
            best_r = r_ach
            best_x = centres.flatten()
        else:
            break
    
    # Final refinement at best_r
    feasible, centres_final, r_final = check_feasibility(best_r, best_x, max_nfev=5000)
    if feasible:
        return centres_final
    
    # Fallback: return best configuration we have
    return best_x.reshape(n, 2)

# EVOLVE-BLOCK-END
