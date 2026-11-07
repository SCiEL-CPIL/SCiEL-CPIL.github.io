"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Spectral parameterization strategy:
    Represent h_i as a linear combination of cosine basis functions with frequencies
    that are divisors of N. This captures periodic/anti-periodic structures that
    minimize cross-correlation. Optimize coefficients via SLSQP.
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    if remaining_time() < 10:
        return project(best_h)
    
    # Define cosine basis functions with frequencies that are divisors of N
    # Frequencies: 0, 1, 2, 4, 5, 10, 20, 25, 50 (divisors of 100)
    # We use frequencies f where the basis is cos(2*pi*f*i/N)
    # For anti-periodic behavior, we want high-frequency components
    
    # Select a set of basis functions
    # Include DC component (f=0), and various harmonics
    # To ensure we can represent diverse structures, use multiple frequencies
    freqs = [0, 1, 2, 4, 5, 10, 20, 25, 50]
    
    # Build the basis matrix B where B[i, j] = cos(2*pi*freqs[j]*i/N)
    i = np.arange(N)
    B = np.zeros((N, len(freqs)))
    for j, f in enumerate(freqs):
        if f == 0:
            B[:, j] = 1.0
        else:
            B[:, j] = np.cos(2 * np.pi * f * i / N)
    
    # h = B @ c, where c is the coefficient vector
    # We need to optimize c such that h = B@c is in [0,1]^N and sum(h) = N/2
    
    # The sum constraint: sum(h) = sum(B@c) = (sum(B) @ c) = N/2
    # For f=0, sum(B[:,0]) = N. For f>0, sum(cos(2*pi*f*i/N)) = 0 if f doesn't divide N in a way that creates full periods... 
    # Actually, for f that divides N, sum_{i=0}^{N-1} cos(2*pi*f*i/N) = 0 if f>0.
    # So sum(h) = N * c[0] (the DC component)
    # Therefore, to have sum(h) = N/2, we need c[0] = 0.5
    
    # So we fix c[0] = 0.5 and optimize the other coefficients
    # Let c = [0.5, c1, c2, ..., c_{m-1}]
    
    m = len(freqs)
    c0 = 0.5
    c_vars = np.zeros(m - 1)  # initial guess for other coefficients
    
    def obj_and_grad(c_vars):
        """Objective and gradient for SLSQP."""
        if not np.all(np.isfinite(c_vars)):
            return float('inf'), np.zeros(m - 1)
        
        # Reconstruct full coefficient vector
        c_full = np.concatenate([[c0], c_vars])
        h = B @ c_full
        
        # Project to [0,1] for feasibility check in objective
        # But SLSQP handles constraints, so we just compute objective on raw h
        # However, we need h to be in [0,1] for the problem to be valid
        # So we add penalty or use constraints
        
        # Compute objective
        if not np.all(np.isfinite(h)):
            return float('inf'), np.zeros(m - 1)
        
        c = np.correlate(h, 1.0 - h, mode="full")
        val = float(np.max(c)) / N * 2.0
        
        if not np.isfinite(val):
            return float('inf'), np.zeros(m - 1)
        
        # Gradient computation (numerical for simplicity, or analytical)
        # For now, use numerical gradient via finite differences
        grad = np.zeros(m - 1)
        eps = 1e-8
        for j in range(m - 1):
            c_plus = c_vars.copy()
            c_plus[j] += eps
            c_full_plus = np.concatenate([[c0], c_plus])
            h_plus = B @ c_full_plus
            if np.all(np.isfinite(h_plus)):
                c_plus_corr = np.correlate(h_plus, 1.0 - h_plus, mode="full")
                val_plus = float(np.max(c_plus_corr)) / N * 2.0
                if np.isfinite(val_plus):
                    grad[j] = (val_plus - val) / eps
                else:
                    grad[j] = 0
            else:
                grad[j] = 0
        
        return val, grad
    
    # Constraints for SLSQP:
    # 1. h_i >= 0 for all i
    # 2. h_i <= 1 for all i
    # 3. Sum constraint is automatically satisfied by fixing c[0]=0.5
    
    def make_bound_constraint(i_idx, lower=0.0, upper=1.0):
        """Create constraint function for h_i in [lower, upper]."""
        def constraint(c_vars):
            c_full = np.concatenate([[c0], c_vars])
            h = B @ c_full
            if lower == 0.0:
                return h[i_idx]  # h_i >= 0
            else:
                return upper - h[i_idx]  # h_i <= upper
        return constraint
    
    # Build constraint list
    constraints = []
    # Lower bound constraints: h_i >= 0
    for i_idx in range(N):
        constraints.append({'type': 'ineq', 'fun': make_bound_constraint(i_idx, lower=0.0)})
    # Upper bound constraints: h_i <= 1
    for i_idx in range(N):
        constraints.append({'type': 'ineq', 'fun': make_bound_constraint(i_idx, upper=1.0)})
    
    # Initial guess: small random perturbations
    np.random.seed(42)
    c_vars_init = np.random.uniform(-0.1, 0.1, m - 1)
    
    # Run SLSQP
    result = minimize(
        obj_and_grad,
        c_vars_init,
        method='SLSQP',
        jac=True,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
    )
    
    if result.success or np.isfinite(result.fun):
        c_vars_opt = result.x
        c_full_opt = np.concatenate([[c0], c_vars_opt])
        h_opt = B @ c_full_opt
        h_opt = project(h_opt)
        val_opt = objective(h_opt)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Try multiple random initializations to escape local minima
    for trial in range(20):
        if remaining_time() < 5:
            break
        
        np.random.seed(100 + trial)
        c_vars_init = np.random.uniform(-0.2, 0.2, m - 1)
        
        result = minimize(
            obj_and_grad,
            c_vars_init,
            method='SLSQP',
            jac=True,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or np.isfinite(result.fun):
            c_vars_opt = result.x
            c_full_opt = np.concatenate([[c0], c_vars_opt])
            h_opt = B @ c_full_opt
            h_opt = project(h_opt)
            val_opt = objective(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Final local refinement using Nelder-Mead on the best result
    if remaining_time() > 10:
        def obj_scipy(x):
            h_p = project(x)
            return objective(h_p)
        
        res = minimize(
            obj_scipy,
            best_h,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        h_res = project(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
    
    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
