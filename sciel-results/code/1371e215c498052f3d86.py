import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import time

def rudin_problem10() -> np.ndarray:
    k = 10
    
    # Quadrature setup
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is in [-1, 1], corresponding to cos(theta)
    # theta = arccos(x)
    theta = np.arccos(x)
    
    # Phi nodes
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create grid
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Flatten grid
    Theta_flat = Theta.flatten()
    Phi_flat = Phi.flatten()
    x_flat = np.cos(Theta_flat)
    
    # Weights for surface integral
    # dOmega = sin(theta) dtheta dphi
    # integral approx sum f(theta_i, phi_j) * w_i * (2*pi/n_phi)
    # Note: leggauss weights are for integral over [-1,1] dx
    # dx = -sin(theta) dtheta
    # So integral over theta is sum f w_i
    # Integral over phi is sum f * (2*pi/n_phi)
    weight_phi = 2 * np.pi / n_phi
    W = np.repeat(w, n_phi) * weight_phi
    
    # Compute Spherical Harmonics Y_m
    # m ranges from -10 to 10
    # c[i] corresponds to m = i - 10
    
    # Precompute P_10^m(x) for m = 0..10
    # lpmv(m, l, x) includes Condon-Shortley phase
    P = np.zeros((11, len(x_flat)))
    for m in range(11):
        P[m] = lpmv(m, k, x_flat)
        
    # Normalization constants N_m
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    N = np.zeros(11)
    for m in range(11):
        # Use log factorial for stability? 10! is small, no need.
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * np.math.factorial(10 - m) / np.math.factorial(10 + m))
        
    # Construct A matrix (21, N_points)
    # A[i, j] is Y_{m_i}(theta_j, phi_j)
    A = np.zeros((21, len(Theta_flat)))
    
    for i, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        if m == 0:
            A[i] = N[0] * P[0]
        elif m > 0:
            A[i] = np.sqrt(2.0) * N[m] * P[m] * np.cos(m * Phi_flat)
        else:
            # m < 0, use |m|
            A[i] = np.sqrt(2.0) * N[abs_m] * P[abs_m] * np.sin(abs_m * Phi_flat)
            
    # Objective function
    def objective(c):
        f = c @ A
        f4 = f ** 4
        S4 = np.sum(f4 * W)
        norm2 = np.sum(c ** 2)
        if S4 <= 0 or norm2 <= 0:
            return 1e10
        return 0.25 * np.log(S4) - np.log(np.sqrt(norm2))
        
    def gradient(c):
        f = c @ A
        f3 = f ** 3
        S4 = np.sum((f ** 4) * W)
        norm2 = np.sum(c ** 2)
        if S4 <= 0 or norm2 <= 0:
            return np.zeros_like(c)
        
        # d(S4)/dc = A^T (W * 4 * f^3)
        g4 = A.T @ (W * 4.0 * f3)
        # d(norm2)/dc = 2 * c
        # d(log(sqrt(norm2)))/dc = d(0.5 log norm2)/dc = c / norm2
        
        return 0.25 * g4 / S4 - c / norm2

    best_c = np.ones(21)
    best_val = objective(best_c)
    
    start_time = time.time()
    deadline = start_time + 500  # Leave 130s margin
    
    # Define initial guesses
    inits = [
        np.ones(21),
        np.random.randn(21),
        np.zeros(21), # will be replaced or handled
    ]
    # Add some structured inits
    for m_target in [0, 2, 4, 6, 8, 10]:
        c_init = np.zeros(21)
        c_init[10 + m_target] = 1.0
        inits.append(c_init)
        c_init[10 - m_target] = 1.0
        inits.append(c_init)
        
    # Shuffle inits
    np.random.shuffle(inits)
    
    for c0 in inits:
        if time.time() > deadline:
            break
            
        # Ensure c0 is not zero
        if np.allclose(c0, 0):
            c0 = np.random.randn(21)
            
        # Normalize c0 to unit norm
        c0 = c0 / np.linalg.norm(c0)
        
        try:
            res = minimize(
                objective, 
                c0, 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                c_opt = res.x
                # Normalize result
                c_opt = c_opt / np.linalg.norm(c_opt)
                val = objective(c_opt)
                if val < best_val:
                    best_val = val
                    best_c = c_opt
        except Exception as e:
            continue
            
    # Final check
    if not np.all(np.isfinite(best_c)):
        return np.ones(21)
        
    # Ensure validity constraints
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3 or max_c > 1e5:
        # Scale to be within range
        # Ideally normalize to 1
        best_c = best_c / np.linalg.norm(best_c)
        
    return best_c
