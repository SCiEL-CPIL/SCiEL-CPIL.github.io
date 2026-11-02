"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.integrate import quad

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

def _time_remaining():
    return _DEADLINE - time.time()

# Precompute Gauss-Legendre nodes and weights for fast L1 integration
# We integrate |P(x)| * exp(-x^2/2) over [-R, R] where R is large enough
_R = 10.0  # exp(-100/2) = exp(-50) ~ 1.9e-22, negligible
_N_QUAD = 256
_x_nodes, _x_weights = np.polynomial.legendre.leggauss(_N_QUAD)
# Transform from [-1,1] to [-R, R]
_x_nodes = _x_nodes * _R
_x_weights = _x_weights * _R
# Precompute exp(-x^2/2) at nodes
_exp_vals = np.exp(-_x_nodes**2 / 2.0)
# Precompute powers of x_nodes for fast polynomial evaluation
_x_pows = np.array([_x_nodes**k for k in range(13)])  # powers 0..12

# Coefficients for H0, H4, H8, H12 as polynomials in x
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
# H12(x) = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
# Combine: P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# Coefficients by power of x (from x^0 to x^12):
# x^0: c0*1 + c4*12 + c8*10080 + c12*10321920
# x^2: c4*(-48) + c8*(-34560) + c12*(-34406400)
# x^4: c4*16 + c8*23040 + c12*34406400
# x^6: c8*(-4608) + c12*(-12902400)
# x^8: c8*256 + c12*2090112
# x^10: c12*(-147456)
# x^12: c12*4096

def _compute_objective(c0, c4, c8):
    """
    Compute the upper bound: (||g||_1^2) / (||g||_2^2)
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    Constraint: g(0) = 0 => c12 is determined.
    """
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Exact L2 norm squared
    norm2_sq = np.sqrt(np.pi) * (
        c0**2 * 1.0 +
        c4**2 * 384.0 +
        c8**2 * 10321920.0 +
        c12**2 * 1962010560000.0
    )
    
    if norm2_sq <= 0 or not np.isfinite(norm2_sq):
        return 1e10
    
    # Fast L1 norm using precomputed Gauss-Legendre quadrature
    # P(x) = sum_{k=0}^{6} a_k * x^{2k} (only even powers)
    # a_0 = c0 + 12*c4 + 10080*c8 + 10321920*c12
    # a_1 = -48*c4 - 34560*c8 - 34406400*c12
    # a_2 = 16*c4 + 23040*c8 + 34406400*c12
    # a_3 = -4608*c8 - 12902400*c12
    # a_4 = 256*c8 + 2090112*c12
    # a_5 = -147456*c12
    # a_6 = 4096*c12
    
    a0 = c0 + 12.0*c4 + 10080.0*c8 + 10321920.0*c12
    a1 = -48.0*c4 - 34560.0*c8 - 34406400.0*c12
    a2 = 16.0*c4 + 23040.0*c8 + 34406400.0*c12
    a3 = -4608.0*c8 - 12902400.0*c12
    a4 = 256.0*c8 + 2090112.0*c12
    a5 = -147456.0*c12
    a6 = 4096.0*c12
    
    # Evaluate P(x) at quadrature nodes using Horner-like method for even powers
    # P(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10 + a6*x^12
    # Let t = x^2, then P = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 + a6*t^6
    t = _x_nodes**2
    # Horner's method: P = a0 + t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*(a5 + t*a6))))))
    poly_vals = a0 + t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * (a5 + t * a6)))))
    
    # L1 norm = sum |P(x_i)| * exp(-x_i^2/2) * w_i
    norm1 = np.sum(np.abs(poly_vals) * _exp_vals * _x_weights)
    
    if not np.isfinite(norm1) or norm1 < 0:
        return 1e10
        
    bound = (norm1**2) / norm2_sq
    return bound

def construct():
    """
    Find coefficients c0, c4, c8 that minimize the uncertainty bound.
    Returns np.array([c0, c4, c8]).
    """
    # Seed coefficients from the prompt
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = seed_result.copy()
    best_score = _compute_objective(*seed_result)
    
    # 1. Local optimization from seed with tight tolerances
    if _time_remaining() > 5:
        try:
            res = minimize(_compute_objective, seed_result, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
        
    # 2. Multi-start local optimization from perturbed seeds
    if _time_remaining() > 30:
        rng = np.random.RandomState(42)
        n_starts = 20
        for i in range(n_starts):
            if _time_remaining() < 5:
                break
            try:
                # Perturb seed with decreasing magnitudes
                scale = 0.1 * (i + 1) / n_starts
                perturbation = scale * rng.randn(3) * np.array([1.0, 0.05, 0.001])
                start = seed_result + perturbation
                res = minimize(_compute_objective, start, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_score and np.isfinite(res.fun):
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
        
    # 3. Basin hopping for global search
    if _time_remaining() > 60:
        try:
            def _objective_wrapper(x):
                return _compute_objective(x[0], x[1], x[2])
            
            res_bh = basinhopping(
                _objective_wrapper,
                x0=best_result.copy(),
                niter=50,
                T=0.5,
                stepsize=0.1,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-13}},
                seed=42
            )
            if res_bh.fun < best_score and np.isfinite(res_bh.fun):
                best_score = res_bh.fun
                best_result = res_bh.x.copy()
        except Exception:
            pass
        
    # 4. Final high-precision polish
    if _time_remaining() > 5:
        try:
            res_final = minimize(_compute_objective, best_result, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if res_final.fun < best_score and np.isfinite(res_final.fun):
                best_score = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass
            
    return best_result

# EVOLVE-BLOCK-END
