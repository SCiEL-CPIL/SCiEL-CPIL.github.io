import numpy as np
from scipy.optimize import minimize


def construct():
    # Hermite polynomial values at 0 for even degrees (physicists' convention)
    # H_0(0) = 1, H_4(0) = -3, H_8(0) = 105, H_12(0) = -10395
    h0, h4, h8, h12 = 1.0, -3.0, 105.0, -10395.0
    
    # n + 1/2 for degrees 0, 4, 8, 12
    moments = np.array([0.5, 4.5, 8.5, 12.5])
    
    def objective(coeffs):
        """coeffs = [c0, c4, c8], c12 derived from g(0)=0"""
        c0, c4, c8 = coeffs
        # g(0) = c0*h0 + c4*h4 + c8*h8 + c12*h12 = 0
        # c12 = -(c0*h0 + c4*h4 + c8*h8) / h12
        c12 = -(c0 * h0 + c4 * h4 + c8 * h8) / h12
        c = np.array([c0, c4, c8, c12])
        
        c2 = c**2
        q = np.sum(c2)
        s = np.dot(c2, moments)
        
        if q < 1e-14:
            return 1e10
        return (s / q)**2
    
    # Try multiple initial points for robustness
    initial_points = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),  # seed
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, -0.5, 0.1]),
        np.array([-0.5, 0.1, 0.01]),
    ]
    
    best_x = None
    best_f = np.inf
    
    for x0 in initial_points:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 200000,
                                   'adaptive': True})
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    
    if best_x is not None:
        return np.array(best_x)
    else:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
