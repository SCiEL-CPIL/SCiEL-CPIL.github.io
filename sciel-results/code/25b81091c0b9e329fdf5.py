import numpy as np
import scipy.optimize as opt
import time
import math

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # Higher precision quadrature to ensure the integrals are accurate
    n_gauss = 30
    nodes, weights = np.polynomial.hermite.hermgauss(n_gauss)
    
    x_nodes = nodes
    x2_at_nodes = x_nodes**2
    
    # Precompute Hermite polynomials and their derivatives at nodes
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    def H0_vals(x):
        return np.ones_like(x, dtype=float)
    
    def H4_vals(x):
        return x**4 - 6*x**2 + 3
    
    def H8_vals(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12_vals(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    H0_at_nodes = H0_vals(x_nodes)
    H4_at_nodes = H4_vals(x_nodes)
    H8_at_nodes = H8_vals(x_nodes)
    H12_at_nodes = H12_vals(x_nodes)
    
    dH0_at_nodes = np.zeros_like(x_nodes)
    dH4_at_nodes = 4*x_nodes**3 - 12*x_nodes
    dH8_at_nodes = 8*x_nodes**7 - 168*x_nodes**5 + 840*x_nodes**3 - 840*x_nodes
    dH12_at_nodes = 12*x_nodes**11 - 660*x_nodes**9 + 10080*x_nodes**7 - 55440*x_nodes**5 + 92400*x_nodes**3 - 55440*x_nodes
    
    # Constraint: P(0) = 0
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    c12_dc0 = -1.0/10395.0
    c12_dc4 = -3.0/10395.0
    c12_dc8 = -105.0/10395.0
    
    # Effective basis for P in terms of c0,c4,c8:
    P_basis0 = H0_at_nodes + c12_dc0 * H12_at_nodes
    P_basis4 = H4_at_nodes + c12_dc4 * H12_at_nodes
    P_basis8 = H8_at_nodes + c12_dc8 * H12_at_nodes
    
    # Q is related to the derivative of P.
    # The variance of the derivative is related to the integral of (P')^2.
    # However, the task description mentions "A(f)*A(Fourier(f))".
    # For g(x) = P(x) exp(-x^2/2), the Fourier transform is related to the derivative.
    # Specifically, if g is a test function, the uncertainty principle bound involves 
    # the variance of x and the variance of the frequency domain.
    # For Gaussian-weighted Hermite polynomials, the "frequency" variance is proportional 
    # to the integral of (dP/dx * exp(-x^2/2))^2? No, it's usually related to the 
    # generator of the Hermite process.
    
    # Let's look at the current program's logic for Q:
    # Q_basis = dH/dx - x*H.
    # This comes from the identity: d/dx (H_n(x) e^{-x^2/2}) = -x H_n e^{-x^2/2} + H_n' e^{-x^2/2} = (H_n' - x H_n) e^{-x^2/2}.
    # So Q is the coefficient of the derivative term.
    # The variance in frequency space for g(x) = P(x) e^{-x^2/2} is proportional to 
    # \int (g'(x))^2 dx / \int g(x)^2 dx.
    # Since g'(x) = Q(x) e^{-x^2/2}, \int (g')^2 = \int Q^2 e^{-x^2} = \int Q^2 e^{-x^2} e^{x^2} e^{-x^2} ?
    # Gauss-Hermite quadrature approximates \int f(x) e^{-x^2} dx.
    # So \int g^2 = \int P^2 e^{-x^2} dx.
    # \int (g')^2 = \int Q^2 e^{-x^2} dx.
    # Thus A_f^2 = \int x^2 P^2 e^{-x^2} / \int P^2 e^{-x^2}.
    # A_gf^2 = (1/(4\pi^2)) * \int Q^2 e^{-x^2} / \int P^2 e^{-x^2}.
    # This matches the current program's compute_uncertainty_product.
    
    Q_basis0 = dH0_at_nodes - x_nodes * H0_at_nodes + c12_dc0 * (dH12_at_nodes - x_nodes * H12_at_nodes)
    Q_basis4 = dH4_at_nodes - x_nodes * H4_at_nodes + c12_dc4 * (dH12_at_nodes - x_nodes * H12_at_nodes)
    Q_basis8 = dH8_at_nodes - x_nodes * H8_at_nodes + c12_dc8 * (dH12_at_nodes - x_nodes * H12_at_nodes)
    
    # Precompute weighted vectors for faster dot products
    # Weights in Gauss-Hermite quadrature for \int f(x) e^{-x^2} dx
    w = weights
    
    def compute_objective_and_grad(coeffs):
        """
        Computes the objective function and its gradient.
        coeffs = [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        
        # P and Q at nodes
        P = c0 * P_basis0 + c4 * P_basis4 + c8 * P_basis8
        Q = c0 * Q_basis0 + c4 * Q_basis4 + c8 * Q_basis8
        
        I0 = np.sum(w * P**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf, np.array([0.0, 0.0, 0.0])
            
        Ix2 = np.sum(w * x2_at_nodes * P**2)
        Ideriv = np.sum(w * Q**2)
        
        if not np.isfinite(Ix2) or not np.isfinite(Ideriv) or Ix2 < 0 or Ideriv < 0:
            return np.inf, np.array([0.0, 0.0, 0.0])
            
        A = Ix2 / I0
        B = Ideriv / I0
        f = np.sqrt(A * B / (4.0 * np.pi**2))
        
        if f <= 1e-30 or not np.isfinite(f):
            return np.inf, np.array([0.0, 0.0, 0.0])
            
        grad = np.zeros(3)
        
        # dI0/dc_i = 2 * sum(w * P * P_basis_i)
        # dIx2/dc_i = 2 * sum(w * x^2 * P * P_basis_i)
        # dIderiv/dc_i = 2 * sum(w * Q * Q_basis_i)
        
        # Precompute dot products with bases
        # To save time, we compute these once per function call
        
        dI0 = np.array([
            2.0 * np.sum(w * P * P_basis0),
            2.0 * np.sum(w * P * P_basis4),
            2.0 * np.sum(w * P * P_basis8)
        ])
        
        dIx2 = np.array([
            2.0 * np.sum(w * x2_at_nodes * P * P_basis0),
            2.0 * np.sum(w * x2_at_nodes * P * P_basis4),
            2.0 * np.sum(w * x2_at_nodes * P * P_basis8)
        ])
        
        dIderiv = np.array([
            2.0 * np.sum(w * Q * Q_basis0),
            2.0 * np.sum(w * Q * Q_basis4),
            2.0 * np.sum(w * Q * Q_basis8)
        ])
        
        # dA/dc = (dIx2 * I0 - Ix2 * dI0) / I0^2
        # dB/dc = (dIderiv * I0 - Ideriv * dI0) / I0^2
        
        dA = (dIx2 * I0 - Ix2 * dI0) / (I0 * I0)
        dB = (dIderiv * I0 - Ideriv * dI0) / (I0 * I0)
        
        # df/dc = (1/(2*f)) * (B/(4*pi^2) * dA + A/(4*pi^2) * dB)
        const = 1.0 / (2.0 * f * 4.0 * np.pi**2)
        grad = const * (B * dA + A * dB)
        
        if not np.all(np.isfinite(grad)):
            return f, np.zeros(3)
            
        return f, grad

    # Initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score, _ = compute_objective_and_grad(x0)
    best_coeffs = x0.copy()
    
    if not np.isfinite(best_score):
        best_score = 1.0
        best_coeffs = x0.copy()
        
    # Strategy: Gradient Descent with Adaptive Step Size and Line Search
    # This is different from Nelder-Mead as it exploits the gradient structure.
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Gradient Descent from seed
    if time.time() < deadline - 30:
        current_coeffs = x0.copy()
        current_score, current_grad = compute_objective_and_grad(current_coeffs)
        
        if np.isfinite(current_score) and current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
            
        step_size = 0.01
        max_iter = 1000
        min_step = 1e-10
        
        for i in range(max_iter):
            if time.time() > deadline - 30:
                break
                
            _, grad = compute_objective_and_grad(current_coeffs)
            
            if not np.all(np.isfinite(grad)) or np.linalg.norm(grad) < 1e-12:
                break
                
            # Gradient direction
            direction = -grad
            
            # Line search: Backtracking
            alpha = step_size
            improved = False
            for _ in range(20):
                new_coeffs = current_coeffs + alpha * direction
                new_score, _ = compute_objective_and_grad(new_coeffs)
                
                if np.isfinite(new_score) and new_score < current_score:
                    current_coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_coeffs = current_coeffs.copy()
                    break
                else:
                    alpha *= 0.5
                    
            if not improved:
                step_size *= 0.5
                if step_size < min_step:
                    break
            else:
                # If we made a good step, we can keep or slightly increase step size
                # step_size = min(step_size * 1.1, 0.1) # Conservative
                pass

    # Phase 2: Random Restarts with Gradient Descent
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate a random start near the best found or seed
        if i % 2 == 0:
            scale = 0.1
            x_start = best_coeffs + rng.normal(0, scale, 3)
        else:
            scale = 0.05
            x_start = x0 + rng.normal(0, scale, 3)
            
        current_coeffs = x_start
        current_score, _ = compute_objective_and_grad(current_coeffs)
        
        if not np.isfinite(current_score):
            continue
            
        step_size = 0.01
        max_iter = 500
        
        for j in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            _, grad = compute_objective_and_grad(current_coeffs)
            if not np.all(np.isfinite(grad)) or np.linalg.norm(grad) < 1e-12:
                break
                
            direction = -grad
            alpha = step_size
            improved = False
            for _ in range(10):
                new_coeffs = current_coeffs + alpha * direction
                new_score, _ = compute_objective_and_grad(new_coeffs)
                
                if np.isfinite(new_score) and new_score < current_score:
                    current_coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_coeffs = current_coeffs.copy()
                    break
                else:
                    alpha *= 0.5
                    
            if not improved:
                step_size *= 0.5
                if step_size < 1e-10:
                    break

    # Phase 3: Final Polish with L-BFGS-B (using gradient)
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(compute_objective_and_grad, best_coeffs, 
                               jac=None, # We return (f, grad), but scipy expects separate or a function returning grad.
                               # Actually, scipy.minimize expects the function to return f.
                               # We need to wrap it or use a different method.
                               # Let's use a wrapper for L-BFGS-B.
                               method='L-BFGS-B', 
                               options={'gtol': 1e-12, 'maxiter': 100})
        except Exception:
            pass
            
    # Since scipy.minimize doesn't directly accept a function returning (f, grad) in all methods without a wrapper,
    # and our custom gradient descent is already quite effective, let's do a final Nelder-Mead polish
    # on the best result found by gradient descent, as NM doesn't need gradients and is robust.
    
    if time.time() < deadline - 1:
        try:
            def nm_obj(x):
                f, _ = compute_objective_and_grad(x)
                return f
                
            res = opt.minimize(nm_obj, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
