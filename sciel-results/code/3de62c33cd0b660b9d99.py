"""
Strategy: 
Reformulate as a smooth optimization problem using a Fourier basis representation 
to ensure smoothness and exploit the spectral structure. 
We minimize a soft-max surrogate of the autoconvolution peak, 
constrained by the sum of heights, using L-BFGS-B.
"""
N = 400
import numpy as np
import time
from scipy.optimize import minimize

def _value(h):
    """Compute the evaluator's value: abs(2 * N * max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    # Full autoconvolution
    ac = np.convolve(h, h, mode='full')
    peak = np.max(ac)
    return abs(2 * N * peak / (s ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Number of Fourier modes to use. 
    # Low K means low dimensionality, faster optimization.
    # High K means more flexibility.
    K_modes = 15
    
    # Precompute Fourier basis vectors for efficiency
    # We represent h as: h_i = c_0 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    # Total parameters: 1 + 2*K_modes
    n_params = 1 + 2 * K_modes
    
    i_idx = np.arange(N)
    cos_basis = np.cos(2 * np.pi * np.outer(np.arange(1, K_modes + 1), i_idx) / N)
    sin_basis = np.sin(2 * np.pi * np.outer(np.arange(1, K_modes + 1), i_idx) / N)
    
    def _reconstruct(params):
        """Reconstruct h from Fourier parameters."""
        c0 = params[0]
        a = params[1:K_modes+1]
        b = params[K_modes+1:]
        h = np.full(N, c0)
        h += cos_basis @ a
        h += sin_basis @ b
        return h
    
    def _objective(params):
        """
        Objective: Minimize 2 * N * soft_max(conv(h,h)) subject to sum(h) ~ 1.
        We use a penalty for sum(h) != 1 to handle the constraint softly,
        or we can normalize h inside the objective.
        Normalizing h inside is better for scale invariance.
        """
        h = _reconstruct(params)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        # Scale h so sum is 1
        h_norm = h / s
        
        # Compute autoconvolution
        ac = np.convolve(h_norm, h_norm, mode='full')
        
        # Soft max approximation of max(ac)
        # beta controls sharpness. Higher beta -> closer to true max.
        beta = 50.0
        # Shift ac to avoid overflow in exp
        max_ac = np.max(ac)
        if max_ac < -100:
            soft_max = max_ac
        else:
            soft_max = max_ac + np.log(np.sum(np.exp(beta * (ac - max_ac)))) / beta
            
        # The value is 2 * N * max(ac) / sum(h)^2. 
        # Since we normalized sum to 1, value = 2 * N * max(ac).
        # But we must be careful: the evaluator uses original h.
        # V(h) = | 2 N max(ac_h) / s^2 |.
        # Let h' = h/s. ac_{h'} = ac_h / s^2.
        # So V(h) = | 2 N max(ac_{h'}) |.
        # So minimizing 2 N max(ac_{h'}) is correct.
        
        val = 2.0 * N * soft_max
        
        # Add a small penalty if h has huge variance or is not well-behaved?
        # Not strictly necessary if L-BFGS-B stays bounded.
        
        return val

    def _objective_grad(params):
        """
        Numerical gradient is sufficient for this dimensionality (31 params) 
        and the smoothness of the soft-max.
        """
        return None # Let scipy handle numerical diff or use finite differences

    # Try multiple random restarts
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        # Start with small coefficients to avoid huge values
        if restart == 0:
            # Try a "flat" start with slight perturbation
            params0 = np.zeros(n_params)
            params0[0] = 1.0 / N # So sum is approx 1
            # Add small noise
            params0[1:] += np.random.normal(0, 0.1, n_params - 1)
        else:
            params0 = np.random.normal(0, 0.5, n_params)
            params0[0] += 1.0/N
            
        try:
            # Use L-BFGS-B
            res = minimize(
                _objective, 
                params0, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                h_opt = _reconstruct(res.x)
                s_opt = np.sum(h_opt)
                if s_opt != 0:
                    val_opt = _value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt
                        
        except Exception:
            pass

    # If we still have time, try a slightly higher dimensionality
    if time.time() < deadline - 30:
        K_modes_2 = 30
        n_params_2 = 1 + 2 * K_modes_2
        cos_basis_2 = np.cos(2 * np.pi * np.outer(np.arange(1, K_modes_2 + 1), i_idx) / N)
        sin_basis_2 = np.sin(2 * np.pi * np.outer(np.arange(1, K_modes_2 + 1), i_idx) / N)
        
        def _reconstruct_2(params):
            c0 = params[0]
            a = params[1:K_modes_2+1]
            b = params[K_modes_2+1:]
            h = np.full(N, c0)
            h += cos_basis_2 @ a
            h += sin_basis_2 @ b
            return h

        def _objective_2(params):
            h = _reconstruct_2(params)
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            h_norm = h / s
            ac = np.convolve(h_norm, h_norm, mode='full')
            beta = 50.0
            max_ac = np.max(ac)
            if max_ac < -100:
                soft_max = max_ac
            else:
                soft_max = max_ac + np.log(np.sum(np.exp(beta * (ac - max_ac)))) / beta
            return 2.0 * N * soft_max

        for restart in range(5):
            if time.time() > deadline - 10:
                break
            params0 = np.random.normal(0, 0.5, n_params_2)
            params0[0] += 1.0/N
            
            try:
                res = minimize(
                    _objective_2, 
                    params0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    h_opt = _reconstruct_2(res.x)
                    s_opt = np.sum(h_opt)
                    if s_opt != 0:
                        val_opt = _value(h_opt)
                        if val_opt < best_val:
                            best_val = val_opt
                            best_h = h_opt
            except Exception:
                pass

    return best_h
