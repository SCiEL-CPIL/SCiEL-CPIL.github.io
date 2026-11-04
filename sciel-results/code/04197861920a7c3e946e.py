"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Structural change: Use Differential Evolution (global optimizer) instead of local
L-BFGS-B. The key insight from inspirations is that global search finds configurations
where autoconvolution peaks are minimized relative to ||h||_2^2, by finding Sidon-like
sparse structures or optimal interference patterns that local methods miss.

Key observations:
- Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
- (h*h)(0) = ||h||_2^2, so ||h*h||_inf >= ||h||_2^2
- Ratio <= 1/||h||_1, so we want small L1 norm
- Scale: ratio(c*h) = ratio(h)/c, so smaller scale gives larger ratio
- The seed (all ones) gives ratio = 1.0
- DFT parameterization h = |FFT(P)|^2 scored ~1.22
- Differential Evolution can escape local minima to find better global structures
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30 or l2sq < 1e-30:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)

def _objective_direct(x):
    """Objective for direct optimization on h (nonnegative via x^2)."""
    h = x**2
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak < 1e-30:
        return 0.0
    return -l2sq / (l1 * peak)

def _objective_P(P_flat):
    """Objective for DFT parameterization: h = |FFT(P)|^2."""
    P_complex = P_flat[0:N] + 1j * P_flat[N:2*N]
    H = np.fft.fft(P_complex)
    h = np.abs(H)**2
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak < 1e-30:
        return 0.0
    return -l2sq / (l1 * peak)

def _refine_local(h0, time_budget, t0):
    """Refine a solution using L-BFGS-B on x = sqrt(h) parameterization."""
    x0 = np.sqrt(np.maximum(h0, 0))
    
    def obj_x(x):
        if time.time() - t0 > time_budget:
            return np.inf
        return _objective_direct(x)
    
    try:
        result = minimize(
            obj_x,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
        )
        h_opt = np.maximum(result.x**2, 0.0)
    except:
        h_opt = h0
    return h_opt

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    t0 = time.time()
    
    # Start with seed solution
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Strategy 1: Differential Evolution on direct h parameterization (x^2)
    # Use a bounded search space for x values
    x_bounds = [(0, 10)] * N
    
    def de_objective(x):
        if time.time() - t0 > _DEADLINE:
            return np.inf
        return _objective_direct(x)
    
    remaining = _DEADLINE - (time.time() - t0)
    if remaining > 30:
        try:
            de_result = differential_evolution(
                de_objective,
                bounds=x_bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True,
                init='latinhypercube',
                time_callback=None,
            )
            h_de = np.maximum(de_result.x**2, 0.0)
            score_de = _ratio(h_de)
            if score_de > best_score:
                best_score = score_de
                best_h = h_de
        except Exception:
            pass
    
    # Strategy 2: Differential Evolution on DFT parameterization
    remaining = _DEADLINE - (time.time() - t0)
    if remaining > 25:
        P_bounds = [(-5, 5)] * (2 * N)
        
        def de_objective_P(P_flat):
            if time.time() - t0 > _DEADLINE:
                return np.inf
            return _objective_P(P_flat)
        
        try:
            de_result_P = differential_evolution(
                de_objective_P,
                bounds=P_bounds,
                seed=123,
                maxiter=40,
                popsize=12,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True,
                init='latinhypercube',
            )
            P_opt = de_result_P.x[0:N] + 1j * de_result_P.x[N:2*N]
            H = np.fft.fft(P_opt)
            h_de_P = np.abs(H)**2
            score_de_P = _ratio(h_de_P)
            if score_de_P > best_score:
                best_score = score_de_P
                best_h = h_de_P
        except Exception:
            pass
    
    # Strategy 3: Local refinement of best solution found so far
    remaining = _DEADLINE - (time.time() - t0)
    if remaining > 10:
        h_refined = _refine_local(best_h, remaining, t0)
        score_ref = _ratio(h_refined)
        if score_ref > best_score:
            best_score = score_ref
            best_h = h_refined
    
    # Strategy 4: Try sparse Sidon-like structures
    # A Sidon set has all pairwise sums distinct, minimizing autoconv overlap
    # Try various sparse patterns and refine locally
    remaining = _DEADLINE - (time.time() - t0)
    if remaining > 15:
        sparse_candidates = []
        
        # Try different sparsity levels
        for n_spikes in [2, 3, 4, 5, 6, 7, 8]:
            if time.time() - t0 > _DEADLINE - 5:
                break
            for seed in range(5):
                r = np.random.RandomState(seed * 100 + n_spikes)
                # Choose random positions
                positions = r.choice(N, size=n_spikes, replace=False)
                # Random amplitudes
                amplitudes = r.uniform(0.5, 2.0, size=n_spikes)
                h_sparse = np.zeros(N)
                for pos, amp in zip(positions, amplitudes):
                    h_sparse[pos] = amp
                sparse_candidates.append(h_sparse)
        
        # Also try structured sparse patterns
        # Fibonacci-like spacing
        for start in range(0, 10):
            positions = []
            pos = start
            while pos < N:
                positions.append(pos)
                pos += 7  # relatively prime to N=50
            if len(positions) >= 2:
                h_fib = np.zeros(N)
                for p in positions:
                    h_fib[p] = 1.0
                sparse_candidates.append(h_fib)
        
        # Try golden ratio spacing
        phi = (1 + np.sqrt(5)) / 2
        for start in range(0, 5):
            positions = []
            pos = start
            while pos < N:
                positions.append(int(pos))
                pos += phi * 5  # scaled golden ratio
            if len(positions) >= 2:
                h_gold = np.zeros(N)
                for p in positions:
                    if p < N:
                        h_gold[p] = 1.0
                sparse_candidates.append(h_gold)
        
        # Refine each sparse candidate locally
        for h_sparse in sparse_candidates:
            if time.time() - t0 > _DEADLINE - 3:
                break
            h_ref = _refine_local(h_sparse, 2.0, t0)
            score_s = _ratio(h_ref)
            if score_s > best_score:
                best_score = score_s
                best_h = h_ref
    
    # Strategy 5: Final local polish on the best solution
    remaining = _DEADLINE - (time.time() - t0)
    if remaining > 5:
        h_final = _refine_local(best_h, remaining - 2, t0)
        score_final = _ratio(h_final)
        if score_final > best_score:
            best_score = score_final
            best_h = h_final
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
