"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using a derivative-free
    trust-region dogleg method applied to the scalar objective ||F(x)||^2.
    """
    x0 = np.asarray(x0, dtype=float)
    
    def objective(x):
        Fx = F(x)
        return float(np.dot(Fx, Fx))
    
    # Use dogleg trust-region method which is derivative-free and robust
    # We set a tight tolerance for the objective (||F||^2 <= (1e-8)^2 = 1e-16)
    # But scipy's minimize may not reach such tight tolerances easily, so we
    # run it and then check. If it doesn't converge tightly enough, we do
    # a few extra Newton-like steps or just return the best found.
    
    try:
        res = minimize(
            objective,
            x0,
            method='dogleg',
            options={
                'maxiter': 200,
                'gtol': 1e-10,
                'xtol': 1e-12,
                'ftol': 1e-16,
            }
        )
        x = res.x
    except Exception:
        x = x0
    
    # Check if we have a good enough root
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    # If dogleg didn't get us there, try a few steps of a simple
    # derivative-free line search or fall back to a more aggressive approach.
    # Try a simple iterative refinement using finite differences for Jacobian
    # but with a damped Newton step.
    
    for _ in range(50):
        Fx = F(x)
        norm_F = np.linalg.norm(Fx)
        if norm_F <= 1e-8:
            break
        
        # Compute finite-difference Jacobian
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            dx = np.linalg.lstsq(J, Fx, rcond=None)[0]
        
        # Damped Newton step with backtracking
        alpha = 1.0
        for _ in range(30):
            x_trial = x - alpha * dx
            F_trial = F(x_trial)
            norm_trial = np.linalg.norm(F_trial)
            if norm_trial < norm_F:
                x = x_trial
                break
            alpha *= 0.5
        else:
            break
    
    return x
# EVOLVE-BLOCK-END
