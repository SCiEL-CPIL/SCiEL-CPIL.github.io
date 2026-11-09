import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    n = 14

    def sph_to_cart(sph):
        """Convert spherical coordinates (theta, phi) to Cartesian on unit sphere."""
        theta = sph[0::2]  # Polar angle [0, pi]
        phi = sph[1::2]    # Azimuthal angle [0, 2*pi]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def thomson_energy(sph):
        """Calculate Thomson energy: sum of 1/d^2 for all pairs."""
        cart = sph_to_cart(sph)
        d = pdist(cart)
        # Avoid division by zero if points coincide
        d[d < 1e-12] = 1e-12
        return np.sum(1.0 / (d ** 2))

    def metric_from_cart(cart):
        """Calculate the actual metric (d_min / d_max)^2."""
        d = pdist(cart)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial guess: Fibonacci sphere
    def fib_sphere(n):
        k = np.arange(n)
        phi = np.arccos(1 - 2 * (k + 0.5) / n)
        golden = np.pi * (1 + np.sqrt(5))
        theta = phi
        phi_az = golden * k
        # Interleave theta and phi for the flat array expected by sph_to_cart
        sph = np.empty(2 * n)
        sph[0::2] = theta
        sph[1::2] = phi_az
        return sph

    x0 = fib_sphere(n)

    # Local minimizer for basinhopping
    def local_minimize(x):
        # Use L-BFGS-B for smooth convergence on Thomson Energy
        res = minimize(thomson_energy, x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        return {'fun': res.fun, 'x': res.x, 'nit': res.nit}

    # Global search using Basin Hopping
    best_result = None
    best_energy = np.inf
    best_metric = 0.0

    # Run basin hopping with multiple restarts or long iterations
    # Basin hopping is stochastic, so we run it for a while
    try:
        # We will run basin hopping for a specific number of steps
        # The step size is important. A step size of 0.1-0.5 radians is typical.
        bh = basinhopping(thomson_energy, x0, niter=200, stepsize=0.5, 
                          minimizer_kwargs={'method': 'L-BFGS-B'}, 
                          interval=50, disp=False)
        
        if bh.fun < best_energy:
            best_energy = bh.fun
            best_result = bh.x
            cart = sph_to_cart(best_result)
            best_metric = metric_from_cart(cart)
            
    except Exception:
        pass

    # If basin hopping didn't run or failed, try a few manual restarts with local minimization
    if best_result is None:
        best_result = x0
        cart = sph_to_cart(best_result)
        best_metric = metric_from_cart(cart)

    # Refine the best result found by basin hopping
    # Do a final local optimization on the Thomson energy to ensure we are at a local minimum
    try:
        res_final = minimize(thomson_energy, best_result, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14})
        best_result = res_final.x
        cart = sph_to_cart(best_result)
        current_metric = metric_from_cart(cart)
        if current_metric > best_metric:
            best_metric = current_metric
    except Exception:
        pass

    # Now, try to improve the actual metric directly using Nelder-Mead
    # The Thomson energy minimum might not be the exact metric maximum
    def neg_metric_sph(sph):
        cart = sph_to_cart(sph)
        return -metric_from_cart(cart)

    try:
        if time.time() < deadline - 30:
            res_metric = minimize(neg_metric_sph, best_result, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            current_metric = -res_metric.fun
            if np.isfinite(current_metric) and current_metric > best_metric:
                best_metric = current_metric
                best_result = res_metric.x
    except Exception:
        pass

    # Final conversion and validation
    result = sph_to_cart(best_result)
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        result[mask, 0] = 1.0
        result[mask, 1:] = 0.0
    result = result / norms
    
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
