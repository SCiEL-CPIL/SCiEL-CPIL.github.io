"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.linalg import solve
import sys

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leave 5s margin

# Representative cosmologies for training the specific error landscape
# We include a wide range of w0, wa to ensure universality
COSMOLOGIES = [
    (-1.0, 0.0),   # LambdaCDM
    (-1.0, 0.5),   # Quintessence
    (-0.9, 0.3),   # General CPL
    (-1.1, 0.0),   # Phantom
    (-0.5, 0.0),   # Matter dominated
    (-1.0, -0.5),  # Negative wa
    (-0.8, 0.2),   # Intermediate
    (-1.2, 0.1),   # More phantom
    (-0.95, 0.1),  # Near LCDM
    (-1.0, 0.2),   # Mild quintessence
    (-1.0, 0.8),   # Strong quintessence
    (-0.7, 0.4),   # Another variant
    (-1.5, 0.0),   # Extreme phantom
    (-0.3, 0.5),   # Low w0
]

# Redshift grid for evaluation
Z_VALS = np.linspace(0.05, 3.0, 50)

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^(-3*w0) * (1 - wa * z / (1+z))^(1.5*wa)
    term1 = (1 + z) ** (-3 * w0)
    base = 1 - wa * z / (1 + z)
    if base < 0:
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral(z, w0, wa):
    """Calculate the true integral part: int_0^1 dx / E(z*x)"""
    if z == 0:
        return 1.0
    def integrand(x):
        return 1.0 / E_z(z * x, w0, wa)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

def compute_worst_error(nodes, weights):
    """Compute the worst-case relative error percentage over the training set."""
    max_err = 0.0
    for z in Z_VALS:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(nodes, weights, z, w0, wa)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_err:
                max_err = rel_err
    return max_err * 100.0 # Return in percent

def construct():
    # Strategy:
    # 1. Initialize with Gauss-Legendre nodes/weights.
    # 2. Construct a "Galerkin" basis of functions that the quadrature should integrate well.
    #    We use Chebyshev polynomials T_k(x) for k=0..7.
    #    Ideally, sum w_i T_k(x_i) = int_0^1 T_k(x) dx.
    #    This ensures exactness for a broad class of smooth functions.
    # 3. Solve for weights that satisfy these conditions given fixed nodes (GL nodes).
    #    Actually, GL already does this for polynomials up to degree 15.
    #    The problem is that 1/E(zx) is not a polynomial.
    #    
    #    Alternative Structural Change: 
    #    Instead of optimizing 16 params blindly, we treat the nodes as fixed 
    #    (Gauss-Legendre) and only optimize the weights? No, weights are constrained to sum to 1.
    #    
    #    Let's try a "Moment Matching" approach with a richer basis.
    #    We want to minimize error for functions f(x) = 1/E(zx).
    #    These functions are smooth but have specific curvature properties.
    #    
    #    New Strategy: 
    #    1. Start with Gauss-Legendre nodes.
    #    2. Use a local search (Nelder-Mead) on the NODES ONLY, keeping weights 
    #       optimized for those nodes via a least-squares fit to a set of "proxy" functions 
    #       derived from the cosmologies.
    #       
    #    Actually, simpler and more robust:
    #    Use Nelder-Mead to optimize the 8 nodes and 8 weights (16 params) 
    #    but with a much better initialization and a more robust objective.
    #    The current program uses L-BFGS-B which can get stuck in local minima 
    #    or struggle with the non-smooth max() operator in the minimax objective.
    #    Nelder-Mead is derivative-free and handles non-smooth objectives better.
    
    # Initial guess: Gauss-Legendre
    x_init, w_init = np.polynomial.legendre.leggauss(8)
    x_init = 0.5 * (x_init + 1)
    w_init = w_init / 2.0
    
    params_init = np.concatenate([x_init, w_init])
    
    # Define a robust objective function
    def objective(params):
        if time.time() > DEADLINE - 2:
            return 1e9
            
        x = params[:8]
        w = params[8:]
        
        # Enforce constraints softly to allow the simplex to move
        # 1. x in [0,1] and increasing
        # 2. w > 0 and sum(w)=1
        
        # Soft constraints
        pen = 0.0
        # Nodes bounds and spacing
        if np.any(x < 0) or np.any(x > 1):
            pen += 1e6
        for i in range(7):
            if x[i+1] - x[i] < 1e-5:
                pen += 1e6 * (1e-5 - (x[i+1] - x[i]))**2
        
        # Weights
        w_sum = np.sum(w)
        if w_sum < 1e-6:
            pen += 1e6
        else:
            w_norm = w / w_sum
            if np.any(w_norm < 0):
                pen += 1e6 * np.sum(np.clip(w_norm, None, 0)**2)
            
            # Calculate error using normalized weights
            max_err = 0.0
            for z in Z_VALS:
                for w0, wa in COSMOLOGIES:
                    true_val = true_integral(z, w0, wa)
                    if true_val < 1e-10:
                        continue
                    approx_val = quadrature_approx(x, w_norm, z, w0, wa)
                    rel_err = abs(approx_val - true_val) / true_val
                    if rel_err > max_err:
                        max_err = rel_err
            
            return max_err * 100.0 + pen
    
    # Use Nelder-Mead for robustness against non-smoothness of the max-error metric
    try:
        result = minimize(
            objective, 
            params_init, 
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        if result.fun < 1e8: # Valid result
            params_init = result.x
    except Exception as e:
        pass # Keep initial GL if optimization fails

    # Final refinement with a small local search around the best found
    # Using COBYLA which handles constraints explicitly? 
    # Or just stick with the Nelder-Mead result.
    
    # Let's do a final quick polish with Powell if time permits
    if time.time() < DEADLINE - 5:
        try:
            result2 = minimize(
                objective, 
                params_init, 
                method='Powell',
                options={
                    'maxiter': 100,
                    'ftol': 1e-8
                }
            )
            if result2.fun < 1e8:
                params_init = result2.x
        except Exception as e:
            pass

    x = params_init[:8]
    w = params_init[8:]
    
    # Post-processing to ensure strict validity
    x = np.sort(x)
    # Clip to [epsilon, 1-epsilon]
    eps = 1e-6
    x = np.clip(x, eps, 1 - eps)
    
    # Ensure strict increase
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + eps
            
    # Re-clip after spacing adjustment
    x = np.clip(x, eps, 1 - eps)
    
    # Normalize weights
    w = np.abs(w)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        w = np.full(8, 1.0/8.0)
    else:
        w = w / w_sum
        
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
