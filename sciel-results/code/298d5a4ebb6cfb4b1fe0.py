import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize, basinhopping

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_{m=-10}^{10} c_m Y_{10}^m that MINIMIZES the norm ratio
    R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients c_m,
        ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    k = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin before 600s limit

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the full matrix Y_flat of shape (70*130, 21)
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    
    # Weights for each quadrature point
    w_flat = np.zeros(N_q, dtype=np.float64)
    
    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi
        
        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]
        
        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            # c_m (index 10+m) corresponds to cos(m*phi)
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            # c_-m (index 10-m) corresponds to sin(m*phi)
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    # Precompute Y^T for faster gradient calculation
    # We need Y^T (w * f) and Y^T (w * f^3)
    # Y_flat.T @ (w_flat * f) can be computed as (Y_flat.T * w_flat[:, None]) @ f
    # Let's precompute Y_w = Y_flat.T * w_flat[:, None]
    Y_w = Y_flat.T * w_flat[:, np.newaxis] # Shape (21, N_q)

    def objective_J(c_vec):
        """
        Compute J = S4 / S2^2 where S2 = sum(w f^2), S4 = sum(w f^4).
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    def jac_J(c_vec):
        """
        Gradient of J = S4 / S2^2.
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        
        if S2 <= 1e-300:
            return np.zeros(21)
            
        # dS2/dc = 2 * Y^T (w f)
        # dS4/dc = 4 * Y^T (w f^3)
        wf = w_flat * f
        wf3 = w_flat * f * f2
        
        dS2 = 2.0 * (Y_w @ wf)
        dS4 = 4.0 * (Y_w @ wf3)
        
        grad = (S2 * dS4 - 2.0 * S4 * dS2) / (S2 * S2 * S2)
        return grad

    # Penalty function for unconstrained minimization (L-BFGS-B)
    # We minimize J(c) + lambda * (||c||^2 - 1)^2
    penalty_weight = 1000.0
    
    def penalized_objective(c_vec):
        J = objective_J(c_vec)
        norm_sq = np.dot(c_vec, c_vec)
        penalty = penalty_weight * (norm_sq - 1.0)**2
        return J + penalty

    def penalized_jac(c_vec):
        J_grad = jac_J(c_vec)
        norm_sq = np.dot(c_vec, c_vec)
        penalty_grad = 4.0 * penalty_weight * (norm_sq - 1.0) * c_vec
        return J_grad + penalty_grad

    # Local minimizer for basinhopping
    def local_minimizer(x0):
        res = minimize(penalized_objective, x0, method='L-BFGS-B', jac=penalized_jac, 
                       options={'maxiter': 100, 'ftol': 1e-12})
        # Normalize the result to be on the sphere for the next step
        x_opt = res.x
        x_opt = x_opt / np.linalg.norm(x_opt)
        return {'x': x_opt, 'fun': objective_J(x_opt), 'nfev': res.nfev}

    # Step function for basinhopping
    def step(x):
        # Generate a random perturbation
        # We want to move on the sphere. A simple way is to add a random vector and renormalize.
        # Or use a small random rotation.
        # Let's use a random vector with small magnitude.
        delta = np.random.randn(21) * 0.5
        x_new = x + delta
        x_new = x_new / np.linalg.norm(x_new)
        return x_new

    # Initial guess
    # Start with a mix of m=0 and m=10
    c_start = np.zeros(21)
    c_start[10] = 1.0 / np.sqrt(2)
    c_start[20] = 1.0 / np.sqrt(2)
    
    best_result = None
    best_obj = np.inf

    # Run basinhopping
    # We limit the number of iterations to keep within time limit
    # basinhopping can be slow, so we monitor time
    try:
        res = basinhopping(local_minimizer, c_start, step=step, niter=20, 
                           minimizer_kwargs={'method': 'L-BFGS-B'})
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x.copy()
    except Exception:
        pass

    # If basinhopping didn't run or failed, do a few SLSQP restarts
    if best_result is None or time.time() < deadline - 30.0:
        starts = []
        # 1. m=0 only
        c_m0 = np.zeros(21)
        c_m0[10] = 1.0
        starts.append(c_m0)
        # 2. m=10 (cos) only
        c_m10 = np.zeros(21)
        c_m10[20] = 1.0
        starts.append(c_m10)
        # 3. m=-10 (sin) only
        c_mneg10 = np.zeros(21)
        c_mneg10[0] = 1.0
        starts.append(c_mneg10)
        # 4. Mix of m=0 and m=10
        c_mix1 = np.zeros(21)
        c_mix1[10] = 1.0 / np.sqrt(2)
        c_mix1[20] = 1.0 / np.sqrt(2)
        starts.append(c_mix1)
        # 5. Mix of m=0 and m=-10
        c_mix2 = np.zeros(21)
        c_mix2[10] = 1.0 / np.sqrt(2)
        c_mix2[0] = 1.0 / np.sqrt(2)
        starts.append(c_mix2)
        
        # Random starts
        for seed in range(20):
            np.random.seed(seed)
            c_rand = np.random.randn(21)
            c_rand = c_rand / np.linalg.norm(c_rand)
            starts.append(c_rand)
            
        # Add the best result from basinhopping if available
        if best_result is not None:
            starts.append(best_result)

        for c_start in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    objective_J,
                    c_start,
                    method='SLSQP',
                    jac=jac_J,
                    constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 
                                  'jac': lambda c: 2.0 * c}],
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x.copy()
            except Exception:
                pass

    if best_result is None:
        best_result = np.ones(21) / np.sqrt(21)
    
    c = best_result
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
        
    return c
