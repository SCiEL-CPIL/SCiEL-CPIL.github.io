"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _normalize(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms

def _covering_radius(points):
    """Compute the covering radius in degrees using a dense spherical grid."""
    # Use a reasonably dense grid to approximate the max angular distance
    # We use a grid that's fine enough to capture the covering radius well
    # For 50 points, the covering radius is ~18.3 degrees, so we need grid spacing < ~5 degrees
    # That's about 36/5 = 7.2 per 36 degrees in each direction
    # Let's use 72x36 grid (5 degree spacing)
    
    n_theta = 72  # 5 degree spacing in theta (0 to 360)
    n_phi = 36    # 5 degree spacing in phi (0 to 180)
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)
    
    # Create meshgrid
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(PH) * np.cos(TH)
    y = np.sin(PH) * np.sin(TH)
    z = np.cos(PH)
    
    # Grid points
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute angular distances from each grid point to each site
    # Using dot product: cos(angle) = dot(p, s)
    # angle = arccos(dot)
    
    # For efficiency, compute in batches
    # points shape: (N, 3), grid shape: (M, 3)
    # We want min over sites for each grid point
    
    # Compute dot products: (M, N)
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Max dot product gives min angle
    max_dots = np.max(dots, axis=1)
    
    # Convert to angles in degrees
    angles = np.degrees(np.arccos(max_dots))
    
    return np.max(angles)

def _soft_covering_radius(points, temperature=0.5):
    """Smooth surrogate for covering radius using log-sum-exp on angular distances."""
    # Use a moderate grid
    n_theta = 48
    n_phi = 24
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)
    
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(PH) * np.cos(TH)
    y = np.sin(PH) * np.sin(TH)
    z = np.cos(PH)
    
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute angular distances
    dots = grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Min angle = arccos(max dot)
    max_dots = np.max(dots, axis=1)
    angles_rad = np.arccos(max_dots)
    
    # Soft max using log-sum-exp
    # soft_max(x) = T * log(sum(exp(x_i / T)))
    # This approximates max(x) as T -> 0
    scaled = angles_rad / temperature
    # Shift for numerical stability
    scaled_max = np.max(scaled)
    log_sum_exp = scaled_max + np.log(np.sum(np.exp(scaled - scaled_max)))
    soft_max = temperature * log_sum_exp
    
    return soft_max * 180.0 / np.pi  # Convert to degrees

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_points = _normalize(best_points)
    best_cr = _covering_radius(best_points)
    
    # Try multiple random restarts with optimization
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial points
        if restart == 0:
            points = best_points.copy()
        else:
            # Random points on sphere
            points = np.random.randn(N, 3)
            points = _normalize(points)
        
        # Optimize using L-BFGS-B with soft covering radius
        try:
            def objective(flat):
                pts = _normalize(flat.reshape(N, 3))
                return _soft_covering_radius(pts, temperature=0.3)
            
            result = minimize(objective, points.ravel(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < best_cr:
                new_points = _normalize(result.x.reshape(N, 3))
                cr = _covering_radius(new_points)
                if cr < best_cr:
                    best_cr = cr
                    best_points = new_points
                    print(f"Restart {restart}: CR = {best_cr:.6f} deg")
        
        except Exception as e:
            continue
        
        # Lloyd's algorithm refinement
        if time.time() > deadline - 15:
            break
        
        try:
            points = best_points.copy()
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Assign each grid point to nearest site
                n_theta = 72
                n_phi = 36
                theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
                phi_grid = np.linspace(0, np.pi, n_phi)
                TH, PH = np.meshgrid(theta, phi_grid, indexing='ij')
                x = np.sin(PH) * np.cos(TH)
                y = np.sin(PH) * np.sin(TH)
                z = np.cos(PH)
                grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                
                dots = grid @ points.T
                dots = np.clip(dots, -1.0, 1.0)
                max_dots = np.max(dots, axis=1)
                nearest = np.argmax(dots, axis=1)
                
                # Move each site to centroid of its Voronoi cell
                new_points = np.zeros_like(points)
                counts = np.zeros(N)
                for j in range(N):
                    mask = nearest == j
                    if np.any(mask):
                        new_points[j] = np.sum(grid[mask], axis=0)
                        counts[j] = np.sum(mask)
                    else:
                        new_points[j] = points[j]
                
                new_points = _normalize(new_points)
                
                cr = _covering_radius(new_points)
                if cr < best_cr:
                    best_cr = cr
                    best_points = new_points
                    points = new_points
                else:
                    # No improvement, stop Lloyd
                    if iteration > 5:
                        break
                    points = new_points
        
        except Exception as e:
            continue
    
    return _normalize(best_points)

# EVOLVE-BLOCK-END
