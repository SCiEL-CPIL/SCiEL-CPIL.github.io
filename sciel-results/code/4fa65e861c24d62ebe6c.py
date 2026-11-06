"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Calculate squared distances
    # diff is (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0) # Avoid log(0)
    
    # We only need upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2_upper = d2[mask]
    
    # Check for very small distances
    if np.any(d2_upper < 1e-18):
        return 1e18 # Return large positive energy to indicate invalid/collision
        
    d_upper = np.sqrt(d2_upper)
    
    # Energy is sum log(1/d) = - sum log(d)
    # Using math.fsum for precision as requested in task contract
    return -math.fsum(np.log(d_upper).tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2_upper = d2[mask]
    
    if np.any(d2_upper < 1e-18):
        return 1e18, np.zeros_like(pts)
        
    d_upper = np.sqrt(d2_upper)
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient of E = - sum_{i<j} log(d_ij)
    # dE/dx_i = - sum_{j!=i} (1/d_ij) * d(d_ij)/d(x_i)
    # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
    # So dE/dx_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    # Let W_ij = 1/d_ij^2 for i!=j, 0 for i=j
    # dE/dx_i = - sum_j W_ij (x_i - x_j) = - (sum_j W_ij) x_i + sum_j W_ij x_j
    
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    
    sum_weights = np.sum(weights, axis=1) # shape (n,)
    weighted_sum = weights @ pts # shape (n, 3)
    
    grad = -pts * sum_weights[:, np.newaxis] + weighted_sum
    
    # Project gradient to tangent space
    # The constraint is |x_i|^2 = 1.
    # Tangent projection: g_t = g - (g . x_i) x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_minimize(pts0, deadline):
    """
    Perform local minimization using L-BFGS-B.
    We optimize the points directly, but we need to ensure they stay on the sphere.
    Standard L-BFGS-B works on Euclidean space.
    Strategy: 
    1. Unconstrained minimization in R^3, then project? No, that's not differentiable at projection.
    2. Use a penalty method? 
    3. Better: Use the fact that we can minimize on the manifold. 
       scipy doesn't have native manifold support in minimize.
       
    Alternative: 
    Since the gradient is already projected to the tangent space, we can use 
    a custom line search or just use L-BFGS-B on the flattened vector and 
    periodically project? 
       
    Actually, a common trick is to minimize the energy of the points 
    without explicit constraint, but the energy function blows up if points 
    leave the sphere? No, the energy is defined for any non-zero vectors if 
    we normalize them inside the function.
    
    Let's define an objective function that takes a vector of 3N floats, 
    reshapes to (N,3), normalizes each point, and returns the energy.
    The gradient returned by L-BFGS-B will be with respect to the un-normalized 
    coordinates. We need to be careful.
    
    If f(x) = E(x/|x|), then grad_f(x) = (I - x x^T / |x|^2) grad_E(x/|x|) / |x| ?
    Actually, simpler: just use L-BFGS-B on the normalized points is hard.
    
    Let's stick to the manual gradient descent but make it MUCH better using 
    Nesterov momentum or just a robust line search?
    
    Wait, the inspiration says L-BFGS is good. How do they do it?
    Usually, people parameterize the sphere using spherical coordinates (theta, phi).
    N points -> 2N parameters.
    theta in [0, pi], phi in [0, 2*pi].
    This avoids the normalization step and keeps points on the sphere automatically.
    However, singularities at poles (theta=0, pi) can be problematic for gradients.
    
    Let's try the Spherical Coordinates approach with L-BFGS-B.
    """
    # Convert cartesian to spherical
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    # r=1
    pts = pts0.copy()
    # Avoid issues with atan2(0,0)
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    
    x0 = np.concatenate([theta, phi])
    
    def objective_and_grad(x):
        theta = x[:N]
        phi = x[N:]
        
        # Handle periodicity of phi? L-BFGS doesn't know about it.
        # We can wrap phi to [-pi, pi] or just let it be.
        # Gradient calculations need to be consistent.
        
        # Compute Cartesian
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        X = sin_t * cos_p
        Y = sin_t * sin_p
        Z = cos_t
        
        pts_cart = np.column_stack([X, Y, Z])
        
        # Calculate Energy and Gradient in Cartesian
        E, grad_cart = _log_energy_grad(pts_cart)
        
        if E > 1e17:
             return E, np.zeros_like(x)
             
        # Chain rule: dE/dx_param = sum_k (dE/dx_k_cart) * (dx_k_cart/dx_param)
        # For point i:
        # dx/dtheta = [cos(t)cos(p), cos(t)sin(p), -sin(t)]
        # dx/dphi = [-sin(p), cos(p), 0]
        
        grad_theta = np.zeros(N)
        grad_phi = np.zeros(N)
        
        for i in range(N):
            t = theta[i]
            p = phi[i]
            
            # Derivatives of Cartesian coords w.r.t theta
            dX_dt = np.cos(t) * np.cos(p)
            dY_dt = np.cos(t) * np.sin(p)
            dZ_dt = -np.sin(t)
            
            # Derivatives of Cartesian coords w.r.t phi
            dX_dp = -np.sin(p)
            dY_dp = np.cos(p)
            dZ_dp = 0.0
            
            grad_theta[i] = grad_cart[i, 0] * dX_dt + grad_cart[i, 1] * dY_dt + grad_cart[i, 2] * dZ_dt
            grad_phi[i] = grad_cart[i, 0] * dX_dp + grad_cart[i, 1] * dY_dp + grad_cart[i, 2] * dZ_dp
            
        grad = np.concatenate([grad_theta, grad_phi])
        return E, grad
        
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if res.success or res.fun < 1e17:
             return res.x
        else:
             return x0
    except Exception:
        return x0

def _spherical_to_cartesian(x):
    theta = x[:N]
    phi = x[N:]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    X = sin_t * cos_p
    Y = sin_t * sin_p
    Z = cos_t
    return np.column_stack([X, Y, Z])

def _cartesian_to_spherical(pts):
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return np.concatenate([theta, phi])

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_energy = np.inf
    best_pts = None
    
    # Strategy: Basin Hopping
    # 1. Start with a good initial guess (Fibonacci or Random)
    # 2. Local Minimize (L-BFGS on spherical coords)
    # 3. Perturb (Randomly rotate or jitter points)
    # 4. Local Minimize again
    # 5. Accept if energy is lower (or with Metropolis probability)
    
    # We will run multiple independent basin hopping chains
    # Each chain gets a time slice
    
    num_chains = 5
    time_per_chain = (deadline - start_time) / num_chains
    
    for chain in range(num_chains):
        chain_deadline = start_time + (chain + 1) * time_per_chain
        
        # Initialize this chain
        if chain == 0:
            pts0 = _fibonacci_sphere(N)
        else:
            pts0 = _random_sphere(N, seed=chain * 100)
            
        x_current = _cartesian_to_spherical(pts0)
        E_current = _log_energy(_spherical_to_cartesian(x_current))
        
        x_best = x_current.copy()
        E_best = E_current
        
        # Basin Hopping Loop
        # Temperature for Metropolis acceptance
        T = 10.0
        cooling = 0.95
        
        while time.time() < chain_deadline:
            # 1. Local Minimization
            x_min = _local_minimize(_spherical_to_cartesian(x_current), chain_deadline - 5)
            pts_min = _spherical_to_cartesian(x_min)
            E_min = _log_energy(pts_min)
            
            if E_min < E_best:
                E_best = E_min
                x_best = x_min.copy()
                
            if E_min > 1e17:
                # Failed, break
                break
                
            # 2. Perturbation
            # Jitter the spherical coordinates
            # Scale of jitter: needs to be large enough to escape local minima
            # but small enough to stay in the same basin often.
            # Typical angular separation is ~ 1/sqrt(N) ~ 0.1 rad
            # Let's try jitter scale 0.1 to 0.5
            jitter_scale = 0.2
            noise = np.random.randn(len(x_current)) * jitter_scale
            x_perturbed = x_min + noise
            
            # Wrap phi to avoid large jumps in parameter space?
            # Not strictly necessary if L-BFGS handles it, but good practice.
            # x_perturbed[N:] = np.arctan2(np.sin(x_perturbed[N:]), np.cos(x_perturbed[N:]))
            
            # 3. Acceptance
            delta_E = E_min - E_current # Wait, we compare new local min to old local min?
            # Standard Basin Hopping: 
            # Generate x_new from x_current.
            # Minimize x_new -> x_new_min.
            # Accept x_new_min as new x_current if E(x_new_min) < E(x_current) OR 
            # with prob exp(-(E(x_new_min) - E(x_current))/T).
            
            # Here x_current is the result of the previous iteration's minimization.
            # So we compare E_min (new) to E_current (old).
            
            if E_min < E_current:
                x_current = x_min
                E_current = E_min
            else:
                if T > 1e-6 and np.random.rand() < math.exp(-(E_min - E_current) / T):
                    x_current = x_min
                    E_current = E_min
                else:
                    # Keep x_current, but we already minimized it, so we are stuck?
                    # In standard BH, if rejected, we stay at x_current.
                    # But we already did the minimization work.
                    # To avoid getting stuck in a cycle, we might want to 
                    # perturb x_current more aggressively next time?
                    # For now, just stay.
                    pass
                    
            T *= cooling
            
            # If temperature is very low, we are just doing local minimization repeatedly.
            # We should break and let the next chain start?
            # Or just continue until deadline.
            
        # End of chain
        if E_best < best_energy:
            best_energy = E_best
            best_pts = _spherical_to_cartesian(x_best)
            
        # Check global time
        if time.time() > deadline:
            break
            
    # Final Polish
    if best_pts is not None:
        # Do one more high-precision polish on the best result
        # Using the local minimizer again with tighter tolerances?
        # The local minimizer already uses tight tolerances.
        # Let's just ensure it's on the sphere.
        best_pts = _project_to_sphere(best_pts)
        
    else:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
