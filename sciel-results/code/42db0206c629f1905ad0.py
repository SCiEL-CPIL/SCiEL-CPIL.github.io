"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute a dense spherical grid for fast covering radius evaluation
def _make_grid(n_theta=200, n_phi=400):
    """Create a dense spherical grid for evaluating covering radius."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

_GRID = _make_grid()

def _covering_radius(pts):
    """Compute the covering radius in degrees using vectorized grid evaluation."""
    # For each grid point, find the minimum angular distance to any point in pts
    # angular distance = arccos(dot product)
    # We want max over grid of min over pts of arccos(dot)
    
    # Compute all dot products: grid (M,3) @ pts.T (3,N) -> (M,N)
    dots = _GRID @ pts.T  # (M, N)
    # Clip for numerical safety
    dots = np.clip(dots, -1.0, 1.0)
    # Min angular distance for each grid point
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    # Covering radius is max angle = arccos(min of max dots)
    # Actually: max over grid of arccos(min_dots[grid_point])
    # = arccos(min over grid of min_dots)
    # Since arccos is decreasing
    max_radius_rad = np.arccos(np.min(min_dots))
    return np.degrees(max_radius_rad)


def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_local(pts, start_time, max_time=120.0):
    """Local optimization using gradient-free moves with fast radius evaluation."""
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Adaptive step size
    step_size = 0.1
    
    while time.time() - start_time < max_time - 5.0:
        improved = False
        # Try moving each point
        for idx in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Try several random directions
            for attempt in range(5):
                # Random direction on sphere
                delta = np.random.randn(3) * step_size
                new_pt = best_pts[idx] + delta
                norm = np.linalg.norm(new_pt)
                if norm < 1e-10:
                    continue
                new_pt = new_pt / norm
                
                new_pts = best_pts.copy()
                new_pts[idx] = new_pt
                
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius - 1e-12:
                    best_pts = new_pts
                    best_radius = new_radius
                    improved = True
                    break
        
        # Adapt step size
        if improved:
            step_size = max(step_size * 0.9, 0.001)
        else:
            step_size = max(step_size * 0.5, 0.001)
            if step_size <= 0.001:
                # Try a few more rounds with smaller steps
                for _ in range(3):
                    if time.time() - start_time > max_time - 5.0:
                        break
                    improved = False
                    for idx in range(N):
                        if time.time() - start_time > max_time - 5.0:
                            break
                        delta = np.random.randn(3) * step_size
                        new_pt = best_pts[idx] + delta
                        norm = np.linalg.norm(new_pt)
                        if norm < 1e-10:
                            continue
                        new_pt = new_pt / norm
                        new_pts = best_pts.copy()
                        new_pts[idx] = new_pt
                        new_radius = _covering_radius(new_pts)
                        if new_radius < best_radius - 1e-12:
                            best_pts = new_pts
                            best_radius = new_radius
                            improved = True
                            break
                    if not improved:
                        break
    
    return best_pts, best_radius


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Local optimization from Fibonacci start
    improved_pts, improved_radius = _optimize_local(pts, start_time, max_time)
    if improved_radius < best_radius:
        best_pts = improved_pts
        best_radius = improved_radius
    
    # Try a few random restarts if time permits
    remaining = max_time - (time.time() - start_time)
    if remaining > 20.0:
        for restart in range(5):
            if time.time() - start_time > max_time - 10.0:
                break
            
            # Random configuration
            np.random.seed(42 + restart)
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Quick local optimization
            improved_pts, improved_radius = _optimize_local(pts, start_time, max_time)
            
            if improved_radius < best_radius:
                best_pts = improved_pts
                best_radius = improved_radius
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
