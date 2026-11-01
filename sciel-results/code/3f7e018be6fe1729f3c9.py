"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy(points):
    """Calculate the logarithmic energy."""
    if not np.all(np.isfinite(points)):
        return float('inf')
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    return energy

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _compute_grad(points):
    """
    Compute the gradient of the logarithmic energy on the sphere.
    E = sum_{i<j} log(1/||p_i - p_j||) = - sum_{i<j} log(||p_i - p_j||)
    
    The gradient in the ambient space is:
    grad_i = - sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    
    We return the Riemannian gradient (projected onto the tangent space).
    """
    if not np.all(np.isfinite(points)):
        return None
    
    # Compute pairwise differences and distances
    # diffs[i, j] = p_i - p_j
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists[i, j] = ||p_i - p_j||
    dists = np.linalg.norm(diffs, axis=-1)
    
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    if np.min(dists) < 1e-9:
        return None

    # We want sum_{j != i} (p_i - p_j) / d_ij^2
    # This is equivalent to summing over all j, excluding i.
    # Since diffs[i, i] = 0, we can sum over all j.
    # coeff[i, j] = 1 / d_ij^2
    coeff = 1.0 / (dists ** 2)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_ambient[i] = sum_j coeff[i, j] * (p_i - p_j)
    # Note: diffs[i, j] is p_i - p_j.
    grad_ambient = np.sum(coeff[:, :, np.newaxis] * diffs, axis=1)
    
    # The energy is -sum log(d), so the gradient of energy is -grad_ambient?
    # Let's check: d/dp_i log(1/d) = d/dp_i (-log d) = - (1/d) (p_i - p_j)/d = -(p_i - p_j)/d^2.
    # So grad_E = - sum (p_i - p_j)/d^2.
    # My grad_ambient above is sum (p_i - p_j)/d^2.
    # So Riemannian gradient of E is -grad_ambient projected.
    
    # Project to tangent space: grad_tan = grad - (grad . p) p
    # We want to minimize E, so we move in direction -grad_tan.
    # Let's define the descent direction.
    
    # grad_E_ambient = -grad_ambient
    # grad_E_tan = grad_E_ambient - (grad_E_ambient . p_i) p_i
    
    # Actually, let's just compute the vector field for descent.
    # v_i = - (grad_E_tan)_i
    # grad_E_ambient_i = - sum_j (p_i - p_j)/d_ij^2
    # grad_E_tan_i = grad_E_ambient_i - (grad_E_ambient_i . p_i) p_i
    # v_i = - grad_E_tan_i = sum_j (p_i - p_j)/d_ij^2 - (sum_j (p_i - p_j)/d_ij^2 . p_i) p_i
    
    # Wait, grad_ambient computed above is sum (p_i - p_j)/d^2.
    # So grad_E_ambient = - grad_ambient.
    # grad_E_tan = - grad_ambient - (- grad_ambient . p_i) p_i = - (grad_ambient - (grad_ambient . p_i) p_i).
    # Descent direction v = - grad_E_tan = grad_ambient - (grad_ambient . p_i) p_i.
    
    # So v_i = proj_tangent(grad_ambient_i)
    
    dots = np.sum(grad_ambient * points, axis=1, keepdims=True)
    grad_tan = grad_ambient - dots * points
    
    return grad_tan

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_energy = float('inf')
    
    # Strategy: Geometric Flow (Riemannian Gradient Descent) with Cooling
    # We run multiple "simulations" (starts) with a cooling schedule.
    # The step size starts large (to escape local minima) and decays exponentially.
    
    starts = []
    # 1. Fibonacci
    starts.append(_fibonacci_sphere(N))
    # 2. Random starts
    rng = np.random.default_rng(42)
    for i in range(10):
        starts.append(_random_sphere(N, seed=42 + i))
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        pts = start_pts.copy()
        
        # Cooling schedule parameters
        # T0: Initial temperature (step size)
        # T_end: Final temperature
        # alpha: Cooling rate
        # We want to run for a fixed number of steps, but monitor time.
        
        # A good strategy for log energy is to start with a step size that allows
        # points to move significantly, then refine.
        
        # Let's use a fixed number of iterations per start, adjusted by time.
        # Approx 5000 steps per start if we have time.
        
        T = 0.1  # Initial step size
        T_end = 1e-6
        # Number of steps to decay from T to T_end
        # T_k = T0 * (T_end/T0)^(k/K)
        # Let's just use exponential decay: T_{k+1} = T_k * decay
        decay = 0.999
        # Or linear in log space?
        
        # Let's try a specific schedule:
        # Phase 1: Large steps, many iterations
        # Phase 2: Small steps, fewer iterations
        
        # Total steps: let's aim for ~2000-5000 steps depending on time left.
        # Each step is O(N^2) ~ 10^4 ops. 5000 steps ~ 5*10^7 ops.
        # NumPy can do ~10^8-10^9 simple ops/sec. So 5000 steps should take < 1s.
        # Wait, matrix ops are slower. Let's benchmark mentally.
        # N=100. diffs is 100x100x3. ~30k floats.
        # 5000 iterations * 30k floats = 1.5e8 float ops.
        # Should be fast (< 1-2 seconds).
        
        # We can afford more steps. Let's do 10,000 steps per start if time permits.
        
        max_steps = 10000
        current_T = 0.05 # Start with moderate step size
        
        # We will use a simple Riemannian Gradient Descent with adaptive step size (backtracking)
        # to ensure we don't diverge, but guided by the cooling schedule.
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient
            grad_tan = _compute_grad(pts)
            if grad_tan is None:
                break
            
            # Compute energy
            e_curr = _energy(pts)
            
            # Update step size based on cooling
            # We want to decay T.
            # Simple exponential decay
            current_T *= 0.9995
            
            # Try to take a step
            # New points: p_new = p + T * v
            # Then project to sphere.
            
            # To be safe, use backtracking line search if the energy increases significantly
            # but capped by current_T.
            
            step_size = current_T
            moved = False
            
            # Backtracking: reduce step size if energy increases
            # Max 10 backtrackings
            for bt in range(10):
                # Move
                pts_new = pts + step_size * grad_tan
                
                # Project to sphere
                norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                pts_new = pts_new / norms
                
                e_new = _energy(pts_new)
                
                if e_new < e_curr:
                    pts = pts_new
                    moved = True
                    break
                else:
                    step_size *= 0.5
            
            if not moved:
                # If we can't move even with tiny steps, we might be at a minimum or stuck
                # Break if step size is very small
                if step_size < 1e-8:
                    break
            
        # End of steps for this start
        
        # Final polish with very small steps
        # (Already covered by the decay)
        
        e_final = _energy(pts)
        if e_final < best_energy:
            best_energy = e_final
            best_points = pts.copy()
            
    # If we still have time, do a final polish on the best point found
    if best_points is not None and time.time() < deadline - 5.0:
        pts = best_points.copy()
        # Run a few more steps with very small T
        T = 1e-5
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
            grad_tan = _compute_grad(pts)
            if grad_tan is None:
                break
            e_curr = _energy(pts)
            pts_new = pts + T * grad_tan
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts_new = pts_new / norms
            e_new = _energy(pts_new)
            if e_new < e_curr:
                pts = pts_new
            T *= 0.999
            
        e_final = _energy(pts)
        if e_final < best_energy:
            best_energy = e_final
            best_points = pts

    if best_points is None:
        best_points = _fibonacci_sphere(N)
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
        
    # Normalize
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
