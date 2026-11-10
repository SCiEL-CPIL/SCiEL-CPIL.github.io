import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length 2*k + 1 = 21.

    Strategy:
    1. Precompute the spherical harmonic matrix Y at all quadrature points.
    2. Define the objective function R^4 = (integral f^4) / (integral f^2)^2 
       on the unit sphere of coefficients (scale invariance).
    3. Use Differential Evolution (global, derivative-free) to find a good basin.
    4. Polish the result using Riemannian Gradient Descent on the unit sphere 
       to achieve high precision.
    5. Return the coefficients.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # Time management
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # --- 1. Precompute Spherical Harmonics at Quadrature Points ---
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    w_flat = np.repeat(w_gl, n_phi)
    
    N = np.zeros(n_coeffs)
    P = np.zeros((n_coeffs, len(x_flat)))
    
    for m in range(0, k + 1):
        norm_factor = 21.0 / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        P[m] = lpmv(m, k, x_flat)
        if m > 0:
            N[-m] = N[m]
            P[-m] = P[m]
            
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            Y_matrix[:, m_idx] = N[0] * P[0]
        elif m > 0:
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_flat)
        else:
            abs_m = -m
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P[abs_m] * np.sin(abs_m * phi_flat)
            
    W_eff = w_flat * (2.0 * np.pi / n_phi)
    
    # --- 2. Define Objective Function ---
    # We optimize on the unit sphere ||c||_2 = 1 to handle scale invariance.
    # R^4 = I4 / I2^2
    # Since ||c||=1, I2 is constant? No, I2 = c^T G c where G is the Gram matrix.
    # But the harmonics are orthonormal, so G = Identity.
    # Thus if ||c||_2 = 1, then I2 = 1.
    # So minimizing R^4 is equivalent to minimizing I4 = sum(W_eff * f^4) subject to ||c||=1.
    
    def objective(c):
        # c is assumed to be normalized, but we normalize for safety in DE
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10
        c_norm = c / norm_c
        f = Y_matrix @ c_norm
        f2 = f * f
        f4 = f2 * f2
        # I2 is 1 if c is unit norm and Y is orthonormal.
        # Let's verify orthonormality of Y columns under the quadrature.
        # The quadrature is exact for polynomials up to degree 2*10 = 20.
        # Y_i Y_j is a polynomial of degree 20. So the discrete inner product 
        # should be exactly delta_ij.
        # Thus I2 = 1.
        # We just minimize I4.
        return np.sum(W_eff * f4)

    def gradient(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return np.zeros_like(c)
        c_norm = c / norm_c
        f = Y_matrix @ c_norm
        f2 = f * f
        # d(I4)/d(c) = d(sum W f^4)/d(c) = 4 * sum W f^3 * (df/dc)
        # df/dc = Y^T f (vector of derivatives wrt coeffs)
        # Actually, f = Y c. df/dc_i = Y_i.
        # grad_I4 = 4 * Y^T (W * f^3)
        grad_I4 = 4.0 * (Y_matrix.T @ (W_eff * f**3))
        # Riemannian gradient projection:
        # The manifold is ||c||=1.
        # Riemannian Grad = Euclidean Grad - (Euclidean Grad . c) c
        # But we are using Euclidean gradient for the line search, 
        # and will project the step.
        return grad_I4

    # --- 3. Global Optimization: Differential Evolution ---
    # Bounds: coefficients are roughly in [-1, 1] for unit norm vectors.
    # We optimize on the unit sphere, so bounds are loose.
    bounds = [(-1.0, 1.0) for _ in range(n_coeffs)]
    
    best_result = None
    best_score = np.inf
    
    # Use DE to find a good starting point
    # maxiter is limited by time.
    # popsize=15 is standard.
    # We will run DE for a portion of the time, then polish.
    
    # Time budget split: 60% for DE, 40% for Polishing
    de_deadline = start_time + 300
    
    try:
        # DE is stochastic. We might run it multiple times if time permits, 
        # but one run with enough iterations is usually enough to find the basin.
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-6,
            init='sobol'
        )
        
        if res_de.success or res_de.fun < best_score:
            best_score = res_de.fun
            best_result = res_de.x.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random if DE fails
        best_result = np.random.randn(n_coeffs)
        best_score = objective(best_result)

    if time.time() > deadline:
        # Return best found so far
        if best_result is None:
            return np.ones(n_coeffs)
        best_result /= np.linalg.norm(best_result)
        return best_result

    # --- 4. Riemannian Polishing ---
    # Take the best result from DE and polish it using Riemannian Gradient Descent.
    # This exploits the local structure for fast convergence.
    
    if best_result is None:
        # Fallback if DE didn't return anything useful
        inits = []
        for m in range(-k, k + 1):
            init = np.zeros(n_coeffs)
            init[m + k] = 1.0
            inits.append(init)
        # Pick the best single mode
        best_init_score = np.inf
        best_init = None
        for init in inits:
            sc = objective(init)
            if sc < best_init_score:
                best_init_score = sc
                best_init = init
        best_result = best_init
        best_score = best_init_score

    # Normalize
    best_result /= np.linalg.norm(best_result)
    
    # Riemannian Gradient Descent with Armijo Line Search
    # Step size schedule: start large, decay if stuck.
    current_x = best_result.copy()
    current_f = best_score
    
    # Precompute Y^T W Y? No, we need Y^T (W * f^3).
    # We can precompute W * Y^T? No, f^3 depends on x.
    
    # Armijo parameters
    alpha = 1.0
    c_armijo = 1e-4
    rho = 0.5
    
    max_iter_polish = 500
    
    for i in range(max_iter_polish):
        if time.time() > deadline:
            break
            
        # Compute Euclidean Gradient
        grad_eucl = gradient(current_x)
        
        # Project to Tangent Space: grad_tan = grad_eucl - (grad_eucl . current_x) * current_x
        proj_coeff = np.dot(grad_eucl, current_x)
        grad_tan = grad_eucl - proj_coeff * current_x
        
        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-10:
            break
            
        # Direction: negative gradient
        direction = -grad_tan
        
        # Normalize direction? Not strictly necessary for Armijo, but helps with scale.
        # Let's keep it as is.
        
        # Line Search: Find step size t such that f(x + t * direction) < f(x) + c * t * grad^T direction
        # Since we are on a sphere, the step should be: x_new = normalize(x + t * direction)
        # Or use exponential map: x_new = cos(t*||dir||) x + sin(t*||dir||) (dir/||dir||)
        # For small steps, normalize(x + t*dir) is sufficient.
        
        # Initial step size estimate
        t = 1.0 / (1.0 + np.linalg.norm(grad_tan))
        
        for _ in range(30): # Max line search iterations
            # Proposed point
            x_trial = current_x + t * direction
            x_trial_norm = np.linalg.norm(x_trial)
            if x_trial_norm < 1e-12:
                t *= 0.5
                continue
            x_trial /= x_trial_norm
            
            f_trial = objective(x_trial)
            
            # Armijo condition
            # Note: grad_tan . direction = -||grad_tan||^2
            slope = np.dot(grad_tan, direction)
            if slope > 0:
                # Should not happen if direction is -grad_tan
                break
                
            if f_trial <= current_f + c_armijo * t * slope:
                # Accept step
                current_x = x_trial
                current_f = f_trial
                break
            else:
                t *= rho
        
    # Final validation and scaling
    if np.any(np.isnan(current_x)) or np.any(np.isinf(current_x)):
        return np.ones(n_coeffs)
        
    # The problem asks for coefficients c. The ratio is scale invariant.
    # The validity check requires max|c| between 1e-3 and 1e5.
    # Our current_x is unit norm, so max|c| is around 1/sqrt(21) ~ 0.2, which is valid.
    # We can return it as is.
    
    return current_x

# Keep the original function name and signature
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had rudin_problem10().
# We have defined it above.

# To ensure no naming conflicts or issues, we just expose rudin_problem10
