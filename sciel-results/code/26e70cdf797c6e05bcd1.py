"""Second autocorrelation inequality -- SOFT-PARAMETERIZED OPTIMIZATION.

Strategy:
1. The objective R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
2. To avoid the non-differentiability of the max operator and the sparsity traps
   of direct L-BFGS-B on non-negative constraints, we parameterize h as the
   squared magnitude of the DFT of a complex vector z: h = |FFT(z)|^2.
   This ensures h >= 0 and provides a smooth manifold for optimization.
3. We use a "soft" ratio function based on log-sum-exp to approximate the
   supremum norm of the autoconvolution, making the objective smooth and
   differentiable.
4. We generate diverse initializations (Gaussian, Lorentzian, Sidon-like, random)
   and optimize each using L-BFGS-B on the complex parameters z.
5. We perform a final polish on the best found solution using direct h-optimization
   with the analytic gradient of the exact (hard) ratio, to snap to the true maximum.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_DEADLINE = 140.0  # Safety margin before 150s kill
_SAFETY = 10.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _exact_ratio(h):
    """Compute exact R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


def _soft_ratio(h, beta=100.0):
    """
    Compute a smooth approximation of R using log-sum-exp for the max term.
    This is differentiable and avoids the kinks of the hard max.
    """
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    
    # Normalize to L1=1 for numerical stability in the soft max
    h_norm = h / l1
    l2sq = float(np.dot(h_norm, h_norm))
    
    full = np.convolve(h_norm, h_norm, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    
    if np.max(ac) <= 1e-300:
        return 0.0
        
    # Log-sum-exp trick for max
    m = float(np.max(ac))
    if m <= 0:
        return 0.0
    log_sum = m + np.log(np.sum(np.exp(ac - m)))
    soft_max = log_sum / beta # Wait, log-sum-exp approximates sum(exp(beta*x)). 
    # We want approx max(ac). 
    # max(x) ~ (1/beta) * log(sum(exp(beta*x)))
    # So soft_max = (1/beta) * log_sum_exp(beta * ac)
    
    shifted = beta * (ac - m)
    log_sum_exp_val = m * beta + np.log(np.sum(np.exp(shifted)))
    soft_max_approx = log_sum_exp_val / beta
    
    if soft_max_approx <= 1e-300:
        return 0.0
        
    return l2sq / soft_max_approx


def _neg_soft_ratio(z_flat):
    """Objective for complex parameter optimization."""
    z = z_flat[:N] + 1j * z_flat[N:]
    h = np.abs(np.fft.fft(z)) ** 2
    # Scale h to have L1=1 to keep it in a good range, though ratio is scale invariant
    # Actually, since ratio is scale invariant, we can just compute it.
    # But FFT magnitudes can be large. Let's normalize h to L1=1 inside the ratio function if needed.
    # _soft_ratio handles normalization.
    r = _soft_ratio(h)
    if r <= 0:
        return 1e10
    return -math.log(r) if r > 0 else 1e10


def _neg_exact_ratio_log(h):
    """Objective for direct h optimization (log of exact ratio)."""
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 1e10
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 1e10
    R = l2sq / (l1 * acmax)
    if R <= 1e-300:
        return 1e10
    return -math.log(R)


def _neg_exact_ratio_log_grad(h):
    """Objective and gradient for direct h optimization."""
    h_pos = np.maximum(h, 0.0)
    l2sq = float(np.dot(h_pos, h_pos))
    l1 = float(np.sum(h_pos))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 1e10, np.zeros_like(h)

    full = np.convolve(h_pos, h_pos, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 1e10, np.zeros_like(h)

    R = l2sq / (l1 * acmax)
    logR = math.log(R)
    val = -logR

    # Gradient of log R
    # d logR / dh_i = 2h_i/l2sq - 1/l1 - (1/acmax) * d(acmax)/dh_i
    m = int(np.argmax(ac))
    dac_dh = np.zeros(N)
    j = m - np.arange(N)
    valid = (j >= 0) & (j < N)
    dac_dh[valid] = 2.0 * DH * h_pos[j[valid]]

    dlogR_dh = (2.0 * h_pos / l2sq) - (1.0 / l1) - (dac_dh / acmax)
    
    # Gradient of -log R is -dlogR_dh
    grad = -dlogR_dh
    
    # Mask for non-negativity constraint (L-BFGS-B handles bounds, but gradient should be 0 where h=0? 
    # Actually, for bounds, the gradient projection is handled by the solver. 
    # However, if h_i=0, the derivative from the left is not defined in the same way, 
    # but we can just return the computed gradient. L-BFGS-B with bounds will project it.
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(h)
        
    return val, grad


def _generate_initializations():
    """Generate diverse initializations for h."""
    inits = []
    x = np.arange(N)
    center = N / 2.0
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Gaussians
    for width in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        for shift in [-5.0, 0.0, 5.0]:
            h = np.exp(-((x - (center + shift)) ** 2) / (2.0 * width ** 2))
            if np.sum(h) > 0:
                inits.append(h / np.sum(h))
                
    # Lorentzians
    for gamma in [0.5, 1.0, 2.0, 4.0]:
        h = (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
        if np.sum(h) > 0:
            inits.append(h / np.sum(h))
            
    # Power laws
    for k in [2, 4, 6]:
        h = np.abs(x - center) ** k
        h = np.maximum(h, 1e-6) # Avoid zeros
        inits.append(h / np.sum(h))
        
    # Sidon-like sets (random sparse)
    rng = np.random.default_rng(42)
    for _ in range(20):
        k = rng.integers(5, 20)
        indices = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        inits.append(h / np.sum(h))
        
    # Random
    for _ in range(20):
        h = rng.random(N)
        if np.sum(h) > 0:
            inits.append(h / np.sum(h))
            
    return inits


def _h_to_z(h):
    """Convert h to complex z such that |FFT(z)|^2 is proportional to h."""
    # We want h = |FFT(z)|^2. 
    # Let H = FFT(z). Then |H|^2 = h.
    # We can choose H = sqrt(h) * exp(i*phi). 
    # To keep it simple, let phi=0. H = sqrt(h).
    # Then z = IFFT(H).
    H = np.sqrt(np.maximum(h, 0.0))
    z = np.fft.ifft(H)
    return z


def _optimize_z(z0, max_iter=200):
    """Optimize z to maximize soft ratio."""
    z_flat = np.concatenate([z0.real, z0.imag])
    
    def objective(z_flat):
        if _time_left() < 5.0:
            return 1e10 # Timeout signal
        return _neg_soft_ratio(z_flat)
        
    try:
        res = minimize(
            objective, z_flat,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - _start) > _DEADLINE - _SAFETY else x),
        )
        z_opt = res.x[:N] + 1j * res.x[N:]
        h_opt = np.abs(np.fft.fft(z_opt)) ** 2
        return h_opt
    except Exception:
        return None


def _optimize_h_direct(h0, max_iter=200):
    """Optimize h directly using exact ratio gradient."""
    h0 = np.maximum(h0, 0.0)
    
    def objective(h):
        if _time_left() < 5.0:
            return 1e10
        val, _ = _neg_exact_ratio_log_grad(h)
        return val
        
    def jac(h):
        _, g = _neg_exact_ratio_log_grad(h)
        return g
        
    try:
        res = minimize(
            objective, h0, jac=jac,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - _start) > _DEADLINE - _SAFETY else x),
        )
        h_opt = np.maximum(res.x, 0.0)
        return h_opt
    except Exception:
        return None


def construct():
    global _start
    _start = time.time()

    best_h = np.ones(N) / N
    best_score = _exact_ratio(best_h)

    # 1. Generate initializations
    inits = _generate_initializations()
    
    # 2. Sort by soft ratio score to prioritize promising starts
    scored_inits = []
    for h_init in inits:
        s = _soft_ratio(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])

    # Update best from initializations
    if scored_inits[0][0] > best_score:
        # Verify with exact ratio
        exact_s = _exact_ratio(scored_inits[0][1])
        if exact_s > best_score:
            best_score = exact_s
            best_h = scored_inits[0][1].copy()

    # 3. Optimize top initializations using Z-parameterization
    top_k = min(10, len(scored_inits))
    for i in range(top_k):
        if _time_left() < 15.0:
            break
            
        _, h_init = scored_inits[i]
        z_init = _h_to_z(h_init)
        
        # Optimize in Z-space
        h_opt_z = _optimize_z(z_init, max_iter=150)
        if h_opt_z is not None:
            h_opt_z = np.maximum(h_opt_z, 0.0)
            if np.sum(h_opt_z) > 1e-10:
                h_opt_z = h_opt_z / np.sum(h_opt_z)
                s = _exact_ratio(h_opt_z)
                if s > best_score:
                    best_score = s
                    best_h = h_opt_z.copy()
                    
        # Refine best found so far in H-space (direct)
        if _time_left() < 10.0:
            break
            
        h_opt_h = _optimize_h_direct(best_h, max_iter=100)
        if h_opt_h is not None:
            h_opt_h = np.maximum(h_opt_h, 0.0)
            if np.sum(h_opt_h) > 1e-10:
                h_opt_h = h_opt_h / np.sum(h_opt_h)
                s = _exact_ratio(h_opt_h)
                if s > best_score:
                    best_score = s
                    best_h = h_opt_h.copy()

    # 4. Final polish on the best found
    if _time_left() > 15.0:
        # Direct high-precision polish
        h_polish = _optimize_h_direct(best_h, max_iter=300)
        if h_polish is not None:
            h_polish = np.maximum(h_polish, 0.0)
            if np.sum(h_polish) > 1e-10:
                h_polish = h_polish / np.sum(h_polish)
                s = _exact_ratio(h_polish)
                if s > best_score:
                    best_score = s
                    best_h = h_polish.copy()
        
        # Try Z-polish from the best H
        if _time_left() > 10.0:
            z_polish = _h_to_z(best_h)
            h_polish_z = _optimize_z(z_polish, max_iter=200)
            if h_polish_z is not None:
                h_polish_z = np.maximum(h_polish_z, 0.0)
                if np.sum(h_polish_z) > 1e-10:
                    h_polish_z = h_polish_z / np.sum(h_polish_z)
                    s = _exact_ratio(h_polish_z)
                    if s > best_score:
                        best_score = s
                        best_h = h_polish_z.copy()
                        
                    # And a final direct polish on the Z-result
                    if _time_left() > 5.0:
                        h_final = _optimize_h_direct(best_h, max_iter=200)
                        if h_final is not None:
                            h_final = np.maximum(h_final, 0.0)
                            if np.sum(h_final) > 1e-10:
                                h_final = h_final / np.sum(h_final)
                                s = _exact_ratio(h_final)
                                if s > best_score:
                                    best_score = s
                                    best_h = h_final.copy()

    # Ensure non-negative and finite
    best_h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        
    # Final normalization (though ratio is invariant, it's good practice)
    l1 = np.sum(best_h)
    if l1 > 1e-10:
        best_h = best_h / l1

    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _exact_ratio(h))
    print("time =", time.time() - t0)
