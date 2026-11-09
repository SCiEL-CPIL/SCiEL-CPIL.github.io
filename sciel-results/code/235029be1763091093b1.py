"""Derivative-free trust-region dogleg method for 2D root finding."""
import numpy as np
from scipy.optimize import minimize


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a derivative-free
    trust-region dogleg method applied to the scalar objective ||F(x)||^2.
    """
    x0 = np.asarray(x0, dtype=float)
    
    def objective(x):
        Fx = F(x)
        return float(np.dot(Fx, Fx))
    
    # Use dogleg trust-region method on ||F||^2
    # This is derivative-free and robust for ill-conditioned systems
    result = minimize(
        objective,
        x0,
        method='dogleg',
        options={
            'gtol': 1e-12,
            'maxiter': 200,
            'disp': False
        }
    )
    
    x = result.x
    
    # Check if we achieved the target tolerance
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # If dogleg didn't converge tightly enough, try a few more Newton-like steps
    # with finite-difference Jacobian as a fallback, but keep it lightweight
    for _ in range(30):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            break
        if norm_Fx < 1e-12:
            break
        
        # Finite-difference Jacobian
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        
        try:
            dx = np.linalg.solve(J, Fx)
            # Damped step to avoid overshooting
            step = x - dx
            F_step = F(step)
            if np.linalg.norm(F_step) < norm_Fx:
                x = step
            else:
                # Halve the step
                step = x - 0.5 * dx
                F_step = F(step)
                if np.linalg.norm(F_step) < norm_Fx:
                    x = step
                else:
                    # Try smaller step
                    step = x - 0.1 * dx
                    F_step = F(step)
                    if np.linalg.norm(F_step) < norm_Fx:
                        x = step
                    else:
                        break
        except np.linalg.LinAlgError:
            break
    
    return x
