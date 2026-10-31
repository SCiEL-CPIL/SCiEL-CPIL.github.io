"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def _compute_integrals():
    """
    Compute integrals I_k_mn = int x^k exp(-x^2) H_m(x) H_n(x) dx for k in {0, 2}
    and m, n in {0, 4, 8, 12}.
    Uses the property that H_n(x) = (-1)^n e^{x^2} d^n/dx^n (e^{-x^2}).
    Or simply use recurrence relations or known values.
    Given the small set, we can compute them using Gauss-Hermite quadrature with high precision
    or symbolic expansion.
    Let's use Gauss-Hermite quadrature with a large number of points for accuracy, 
    as it's simpler to implement correctly than deriving the exact formulas for all cross terms.
    """
    # We need integrals of x^k e^{-x^2} H_m(x) H_n(x)
    # Standard Gauss-Hermite quadrature is for weight e^{-x^2}.
    # np.polynomial.hermite.hermgauss(n) gives nodes and weights for weight e^{-x^2}.
    # The integral is sum w_i * f(x_i).
    # Here f(x) = x^k H_m(x) H_n(x).
    
    # Use 100 points for high accuracy
    x, w = np.polynomial.hermite.hermgauss(100)
    
    degrees = [0, 4, 8, 12]
    H_vals = {d: np.polynomial.hermite.hermval(x, np.poly1d(np.polynomial.hermite.herm2phys(d))) if False else _eval_hermite(d, x) for d in degrees}
    
    # Actually, scipy.special.eval_hermite is easier.
    from scipy.special import eval_hermite
    H_vals = {d: eval_hermite(d, x) for d in degrees}
    
    I0 = np.zeros((4, 4))
    I2 = np.zeros((4, 4))
    
    for i, m in enumerate(degrees):
        for j, n in enumerate(degrees):
            # Integrand for I0: e^{-x^2} H_m H_n
            # Quadrature sum: w_i * H_m(x_i) H_n(x_i)
            I0[i, j] = np.sum(w * H_vals[m] * H_vals[n])
            
            # Integrand for I2: x^2 e^{-x^2} H_m H_n
            I2[i, j] = np.sum(w * x**2 * H_vals[m] * H_vals[n])
            
    return I0, I2

def _eval_hermite(n, x):
    # Fallback if needed, but we use scipy.special
    from scipy.special import eval_hermite
    return eval_hermite(n, x)

# Precompute integrals
I0, I2 = _compute_integrals()

def objective(coeffs):
    c0, c4, c8 = coeffs
    # Constraint g(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-498960
    c12 = (c0 * 1 - c4 * 12 + c8 * 1680) / 498960.0
    
    c = np.array([c0, c4, c8, c12])
    
    # V_x = (c^T I2 c) / (c^T I0 c)
    # V_xi = (c_ft^T I2 c_ft) / (c_ft^T I0 c_ft)
    # where c_ft[k] = c[k] * (-1)^(deg/2)
    # degs: 0, 4, 8, 12
    # signs: 1, -1, 1, -1
    signs = np.array([1, -1, 1, -1])
    c_ft = c * signs
    
    num_x = c @ I2 @ c
    den_x = c @ I0 @ c
    num_xi = c_ft @ I2 @ c_ft
    den_xi = c_ft @ I0 @ c_ft
    
    if den_x < 1e-10 or den_xi < 1e-10:
        return 1e10
        
    V_x = num_x / den_x
    V_xi = num_xi / den_xi
    
    return np.sqrt(V_x * V_xi)

def construct():
    t0 = time.time()
    deadline = t0 + 170
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective(x0)
    best_x = x0
    
    # Try multiple random starts to avoid local minima
    np.random.seed(42)
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
            
        # Generate random initial guess
        # Scale based on seed values
        scale = 1.0 + np.random.rand() * 2.0
        x_rand = x0 * scale * (1 + 0.5 * np.random.randn(3))
        
        res = minimize(objective, x_rand, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
        
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_x = res.x
            
    # Final polish with L-BFGS-B if time permits
    if time.time() < deadline - 5:
        res_final = minimize(objective, best_x, method='L-BFGS-B',
                             options={'gtol': 1e-12, 'maxiter': 1000})
        if res_final.fun < best_score:
            best_score = res_final.fun
            best_x = res_final.x
            
    return np.array(best_x)

# EVOLVE-BLOCK-END
