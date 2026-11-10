"""Optimized search for flattest +/-1 polynomial with 24 coefficients."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# Deadline guard: evaluator kills at 150s; use a safety margin.
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _sup_norm(coefs, n_samples=4096, refine=True, refine_iters=60):
    """Compute max |P(z)| over |z|=1 via dense sampling + golden-section refinement."""
    # P(z) = sum c_k z^k, k=0..23
    # Use FFT-based evaluation for speed
    # We need P(e^{i theta}) = sum c_k e^{i k theta}
    # This is the DFT of c at frequencies theta.
    
    # Dense sampling
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    # Compute P at all these points efficiently
    # P(e^{i theta_j}) = sum_k c_k e^{i k theta_j}
    # This is like evaluating a polynomial at roots of unity, but shifted.
    # We can use: if we define d_k = c_k, then P(e^{i theta}) is the DFT.
    # Actually, let's just use np.dot with complex exponentials for clarity, 
    # but that's O(N^2). Better: use FFT.
    
    # P(e^{i theta_j}) = sum_k c_k (e^{i theta_j})^k
    # Let's compute this as: for each theta_j, it's the DFT of c evaluated at frequency theta_j.
    # We can use: P(e^{i theta}) = IDFT of something? 
    # Simpler: use numpy's polyval or direct matrix multiplication for 24 points x 4096 samples.
    # 24 * 4096 = ~100k ops, very fast.
    
    # Build matrix: M[j, k] = exp(i * k * theta_j)
    # P[j] = sum_k M[j,k] * c_k
    # This is M @ c
    
    k = np.arange(24)
    # theta_j = 2*pi*j/n_samples
    # exp(i * k * theta_j) = exp(i * 2*pi * k * j / n_samples)
    # This is the DFT matrix (unnormalized)
    # P[j] = sum_k c_k * exp(i * 2*pi * k * j / n_samples)
    # This is the DFT of c, but with sign convention: DFT[c][j] = sum_k c_k exp(-i 2 pi k j / N)
    # So P[j] = DFT[c_conj][j] * (-1)^j? Let's be careful.
    # exp(i * 2*pi * k * j / N) vs exp(-i * 2*pi * k * j / N)
    # So P[j] = sum_k c_k exp(i * 2*pi * k * j / N) = DFT of c with positive sign.
    # np.fft.fft uses negative sign. So P = np.fft.fft(c)[::-1] conjugated? 
    # Actually: DFT_plus[c][j] = sum_k c_k exp(i 2 pi k j / N)
    # DFT_minus[c][j] = sum_k c_k exp(-i 2 pi k j / N)
    # DFT_plus[c][j] = DFT_minus[c*][j] * exp(i 2 pi * 0 * j / N)? No.
    # DFT_plus[c][j] = conj(DFT_minus[conj(c)][j])
    # Since c is real: DFT_plus[c][j] = conj(DFT_minus[c][j])
    # So P[j] = conj(np.fft.fft(c)[j])
    
    c = np.array(coefs, dtype=np.float64)
    # Pad to next power of 2 for FFT efficiency? 24 is small, just use 4096.
    # Actually, we want to evaluate at 4096 points, not 24.
    # The FFT of a 24-point sequence gives values at 24 frequencies.
    # We need values at 4096 frequencies. This is interpolation.
    # Better to just do the matrix multiplication directly. It's only 24*4096 = 98304 complex mults.
    
    # Direct computation
    # thetas = 2*pi*j/n_samples
    # P[j] = sum_k c_k exp(i k theta_j)
    
    # Vectorized: 
    # k_theta = k[None, :] * thetas[:, None]  # shape (n_samples, 24)
    # P = exp(i * k_theta) @ c  # shape (n_samples,)
    
    # This is memory efficient enough: 4096 * 24 complex numbers = ~98k * 16 bytes = 1.5 MB
    
    k_idx = np.arange(24)
    theta_j = 2 * np.pi * np.arange(n_samples) / n_samples
    # k_idx shape (24,), theta_j shape (4096,)
    # We want exp(i * k_idx[m] * theta_j[n]) for all m, n
    # = exp(i * outer(k_idx, theta_j))
    # But we can do: exp(i * theta_j[:, None] * k_idx[None, :]) @ c
    
    phase = np.exp(1j * np.outer(theta_j, k_idx))  # (4096, 24)
    P_vals = phase @ c  # (4096,)
    abs_vals = np.abs(P_vals)
    
    if not refine:
        return np.max(abs_vals)
    
    # Find the top few peaks and refine
    # Get indices of top 10 peaks
    top_k = 10
    top_indices = np.argsort(abs_vals)[-top_k:]
    
    max_val = 0.0
    for idx in top_indices:
        # Golden section refinement around theta = 2*pi*idx/n_samples
        center = 2 * np.pi * idx / n_samples
        # Search in a small interval around center
        # The spacing is 2*pi/n_samples
        delta = 2 * np.pi / n_samples
        a = center - delta
        b = center + delta
        
        # Use golden section to minimize -|P(z)|, i.e., maximize |P(z)|
        # Define objective as -|P(e^{i theta})|
        
        def neg_abs_p(theta):
            # Compute |P(e^{i theta})|
            # P(e^{i theta}) = sum_k c_k exp(i k theta)
            # Use direct sum for accuracy
            vals = np.exp(1j * theta * k_idx)
            p_val = np.dot(c, vals)
            return -np.abs(p_val)
        
        # Golden section search
        result = minimize_scalar(neg_abs_p, bounds=(a, b), method='bounded', 
                                  options={'xatol': 1e-12, 'maxiter': refine_iters})
        local_max = -result.fun
        if local_max > max_val:
            max_val = local_max
    
    # Also check the global max from sampling in case refinement missed something
    global_max = np.max(abs_vals)
    return max(max_val, global_max)

def _sup_norm_fast(coefs):
    """Fast sup norm computation with fewer samples for search."""
    return _sup_norm(coefs, n_samples=1024, refine=False)

def _sup_norm_refined(coefs):
    """High-precision sup norm for final evaluation."""
    return _sup_norm(coefs, n_samples=8192, refine=True, refine_iters=80)

def _local_search(coefs, max_iter=2000, time_budget=5.0):
    """Local search: flip one coefficient at a time to reduce sup norm."""
    coefs = list(coefs)
    current_score = _sup_norm_fast(coefs)
    
    start = time.time()
    no_improve = 0
    max_no_improve = 50
    
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        if _remaining() < 2.0:
            break
        
        best_flip = -1
        best_score = current_score
        
        for i in range(24):
            # Flip coefficient i
            new_coefs = coefs[:]
            new_coefs[i] = -new_coefs[i]
            score = _sup_norm_fast(new_coefs)
            if score < best_score - 1e-12:
                best_score = score
                best_flip = i
        
        if best_flip >= 0:
            coefs[best_flip] = -coefs[best_flip]
            current_score = best_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return coefs, current_score

def _random_restart_search(n_restarts=50, time_budget=5.0):
    """Random restart local search."""
    best_coefs = None
    best_score = np.inf
    
    start = time.time()
    
    for r in range(n_restarts):
        if time.time() - start > time_budget:
            break
        if _remaining() < 5.0:
            break
        
        # Random initial configuration
        coefs = [1 if np.random.rand() < 0.5 else -1 for _ in range(24)]
        
        # Local search
        coefs, score = _local_search(coefs, max_iter=1000, time_budget=3.0)
        
        if score < best_score:
            best_score = score
            best_coefs = coefs[:]
    
    return best_coefs, best_score

def _refine_with_golden(coefs, n_samples=16384, top_peaks=5, refine_iters=100):
    """Final high-precision refinement of the sup norm."""
    return _sup_norm(coefs, n_samples=n_samples, refine=True, refine_iters=refine_iters)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with GRS as a good initial guess
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    grs_coefs = p[:24]
    
    best_coefs = grs_coefs[:]
    best_score = _sup_norm_fast(grs_coefs)
    
    # Local search from GRS
    if _remaining() > 10.0:
        ls_coefs, ls_score = _local_search(grs_coefs, max_iter=3000, time_budget=8.0)
        if ls_score < best_score:
            best_coefs = ls_coefs
            best_score = ls_score
    
    # Random restarts
    if _remaining() > 30.0:
        rr_coefs, rr_score = _random_restart_search(n_restarts=100, time_budget=min(60.0, _remaining() - 10.0))
        if rr_coefs is not None and rr_score < best_score:
            best_coefs = rr_coefs
            best_score = rr_score
    
    # Final local polish on the best
    if _remaining() > 10.0:
        final_coefs, final_score = _local_search(best_coefs, max_iter=5000, time_budget=min(30.0, _remaining() - 5.0))
        if final_score < best_score:
            best_coefs = final_coefs
            best_score = final_score
    
    # Return the best coefficients found
    # The evaluator will compute the score; we just need to return coefficients
    return best_coefs
