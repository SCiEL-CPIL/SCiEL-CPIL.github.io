"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT (faster than np.convolve for large N)."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _soft_max_grad(x, beta):
    """Gradient of soft max with respect to x."""
    m = np.max(x)
    exp_vals = np.exp(beta * (x - m))
    probs = exp_vals / np.sum(exp_vals)
    return beta * probs

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _toeplitz_eigenvector_basis(K, rng):
    """
    Generate K eigenvectors of a random symmetric Toeplitz matrix.
    These provide a low-dimensional subspace of structured signals.
    """
    # Generate a random symmetric Toeplitz matrix
    # Use a random vector to define the first row
    first_row = rng.randn(N)
    # Make it symmetric Toeplitz
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            T[i, j] = first_row[abs(i - j)]
    
    # Compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(T)
    
    # Select top K eigenvectors (largest eigenvalues for structure)
    # Also include some from the middle for diversity
    idx = list(range(K // 2)) + list(range(N - K // 2, N))
    if len(idx) < K:
        idx = list(range(K))
    idx = idx[:K]
    
    basis = eigvecs[:, idx].astype(float)
    
    # Orthogonalize to ensure independence
    for i in range(1, K):
        for j in range(i):
            basis[:, i] -= np.dot(basis[:, i], basis[:, j]) * basis[:, j]
        norm = np.linalg.norm(basis[:, i])
        if norm > 1e-10:
            basis[:, i] /= norm
    
    return basis

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses eigenvector subspace parameterization with soft-max surrogate optimization.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (score 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Eigenvector subspace optimization
    # Represent h = b + sum_k c_k * v_k where v_k are eigenvectors
    # This constrains h to a low-dimensional subspace of structured signals
    rng = np.random.RandomState(42)
    
    for K in [5, 10, 15, 20]:
        if time.time() > deadline - 60:
            break
        
        # Generate basis
        basis = _toeplitz_eigenvector_basis(K, rng)
        
        # Objective in coefficient space
        def obj_coeff(coeffs):
            """
            coeffs: [b, c1, c2, ..., cK]
            h = b + sum_k c_k * v_k
            """
            b = coeffs[0]
            c = coeffs[1:]
            h = b + basis @ c
            
            if not np.all(np.isfinite(h)):
                return 1e10
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, 20.0)
            return abs(2.0 * N * sm / (N * N))
        
        # Initial guess: project best_h onto the subspace
        # b = mean(best_h), c_k = dot(best_h - mean, v_k)
        mean_h = np.mean(best_h)
        c_init = np.zeros(K)
        for k in range(K):
            c_init[k] = np.dot(best_h - mean_h, basis[:, k])
        coeffs_init = np.concatenate([[mean_h], c_init])
        
        # Try multiple random restarts
        for trial in range(3):
            if time.time() > deadline - 60:
                break
            
            if trial == 0:
                coeffs_try = coeffs_init.copy()
            else:
                coeffs_try = coeffs_init + 0.1 * rng.randn(K + 1)
            
            try:
                result = minimize(
                    obj_coeff,
                    coeffs_try,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                coeffs_opt = result.x
                b_opt = coeffs_opt[0]
                c_opt = coeffs_opt[1:]
                h_opt = b_opt + basis @ c_opt
                
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 3: Full-dimension soft-max optimization with multiple beta values
    for beta in [5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 30:
            break
        
        def obj_time(h):
            return _objective_soft(h, beta)
        
        rng2 = np.random.RandomState(100 + int(beta))
        
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.05 * rng2.randn(N)
                h_init -= np.mean(h_init)
                h_init += 1.0
                h_init = h_init * (N / np.sum(h_init))
            
            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 4: High-precision polish with very high beta
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
                break
            
            def obj_polish(h):
                return _objective_soft(h, beta)
            
            try:
                result = minimize(
                    obj_polish,
                    best_h.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 5: Coordinate descent polish in time domain
    if time.time() < deadline - 10:
        h = best_h.copy()
        step = 0.05
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            idx = np.argsort(np.abs(h))[::-1][:50]
            for i in idx:
                if time.time() > deadline - 5:
                    break
                old = h[i]
                for delta in [step, -step]:
                    h[i] = old + delta
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        h[i] = old
                        continue
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
