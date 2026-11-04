"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_ratio(h):
    """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    # Circular autocorrelation
    ac = np.fft.irfft(np.fft.rfft(h, n=2 * N) ** 2, n=2 * N)
    ac_inf = np.max(ac)
    if ac_inf <= 0:
        return 0.0
    return l2_sq / (l1 * ac_inf)


def _smooth_objective(params):
    """
    Objective: -log(R) using DFT parameterization.
    params: real vector of length 2*(N//2+1) representing complex coefficients
    of a polynomial P of degree N//2. h_k = |P(w^k)|^2 where w = exp(2*pi*i/N).
    """
    ncoeff = N // 2 + 1
    real_part = params[:ncoeff]
    imag_part = params[ncoeff:2 * ncoeff]
    coeffs = real_part + 1j * imag_part
    
    # Evaluate |P(w^k)|^2 for k = 0, ..., N-1
    # Use FFT-based evaluation
    freqs = np.arange(N)
    # P(w^k) = sum_{j=0}^{N//2} c_j * w^{k*j}
    # This is like evaluating a polynomial at roots of unity
    # We can use the fact that if we zero-pad coeffs to length N, 
    # then the DFT gives us P(w^k)
    full_coeffs = np.zeros(N, dtype=np.complex128)
    full_coeffs[:ncoeff] = coeffs
    h = np.abs(np.fft.fft(full_coeffs)) ** 2
    h = np.real(h)
    h = np.maximum(h, 0.0)
    
    return -np.log(_compute_ratio(h) + 1e-300), h


def construct():
    t_start = time.time()
    deadline = 130.0  # Leave 20s safety margin
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _compute_ratio(best_h)
    
    ncoeff = N // 2 + 1
    dim = 2 * ncoeff
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() - t_start > deadline:
            break
        
        # Random complex coefficients
        real_init = np.random.randn(ncoeff) * 0.5
        imag_init = np.random.randn(ncoeff) * 0.5
        params_init = np.concatenate([real_init, imag_init])
        
        # Evaluate initial
        _, h_init = _smooth_objective(params_init)
        score_init = _compute_ratio(h_init)
        
        if score_init > best_score:
            best_score = score_init
            best_h = h_init.copy()
        
        # Local optimization with L-BFGS-B
        if time.time() - t_start > deadline - 5:
            break
        
        try:
            result = minimize(
                lambda p: _smooth_objective(p)[0],
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -np.log(score_init + 1e-300):
                _, h_opt = _smooth_objective(result.x)
                score_opt = _compute_ratio(h_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Also try a few structured initializations
    structured_inits = []
    
    # All ones
    structured_inits.append(np.ones(dim) * 0.1)
    
    # Low frequency only
    for f in [1, 2, 3]:
        init = np.zeros(dim)
        init[f] = 1.0
        init[ncoeff + f] = 0.5
        structured_inits.append(init)
    
    for init in structured_inits:
        if time.time() - t_start > deadline:
            break
        
        try:
            result = minimize(
                lambda p: _smooth_objective(p)[0],
                init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < -np.log(best_score + 1e-300):
                _, h_opt = _smooth_objective(result.x)
                score_opt = _compute_ratio(h_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: try differential evolution if time permits
    if time.time() - t_start < deadline - 10:
        try:
            bounds = [(-2, 2)] * dim
            
            def de_obj(p):
                return _smooth_objective(p)[0]
            
            de_result = differential_evolution(
                de_obj,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42,
                tol=1e-8,
                polish=True
            )
            
            _, h_de = _smooth_objective(de_result.x)
            score_de = _compute_ratio(h_de)
            
            if score_de > best_score:
                best_score = score_de
                best_h = h_de.copy()
        except Exception:
            pass
    
    return list(np.real(best_h))

# EVOLVE-BLOCK-END
