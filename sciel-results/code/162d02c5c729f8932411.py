"""Optimized program for the Third Autocorrelation Inequality task.

Strategy: Use Zadoff-Chu sequences windowed by smooth low-frequency functions to exploit
their near-zero autocorrelation property, combined with DCT-based spectral shaping.

Approach:
1. Zadoff-Chu windowed initializations (various roots, windows)
2. DCT low-frequency basis optimization (reduces dimensionality from 400 to ~30)
3. Smooth-max refinement with L-BFGS-B on the full 400-dim problem
4. Final polishing with Nelder-Mead
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _fft_convolve(h):
    """Fast convolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    size = 1 << (size - 1).bit_length()
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv


def _zadoff_chu_seq(root, length):
    """Generate a Zadoff-Chu sequence."""
    n = np.arange(length)
    if root % (2 * length) < length:
        seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        seq = np.exp(-1j * np.pi * root * n * (n - 1) / length)
    return seq


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    idx = np.arange(N)
    mid = (N - 1) / 2.0

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Zadoff-Chu windowed initializations
    candidate_list = []

    # Constant baseline
    candidate_list.append(np.ones(N))

    # Zadoff-Chu sequences with various windows
    for root in [1, 2, 3, 5, 7, 11, 13, 17]:
        zc = _zadoff_chu_seq(root, N)
        
        # Real part only (since we need real heights)
        zc_real = np.real(zc)
        
        # Add DC component to ensure non-zero sum
        dc_offset = -np.mean(zc_real) + 1.0
        
        # Apply various windows
        for win_type in ['hann', 'gaussian', 'triangular', 'rectangular']:
            if win_type == 'hann':
                win = 0.5 * (1 + np.cos(2 * np.pi * idx / N))
            elif win_type == 'gaussian':
                for sigma in [0.2, 0.3, 0.4, 0.5]:
                    win = np.exp(-0.5 * ((idx - mid) / (sigma * N)) ** 2)
                    h = dc_offset + zc_real * win
                    s = np.sum(h)
                    if abs(s) > 1e-10:
                        candidate_list.append(h / s)
                continue
            elif win_type == 'triangular':
                win = 1.0 - np.abs(idx - mid) / mid
            else:  # rectangular
                win = np.ones(N)
            
            h = dc_offset + zc_real * win
            s = np.sum(h)
            if abs(s) > 1e-10:
                candidate_list.append(h / s)
        
        # Also try imaginary part
        zc_imag = np.imag(zc)
        dc_offset = -np.mean(zc_imag) + 1.0
        for win_type in ['hann', 'gaussian']:
            if win_type == 'hann':
                win = 0.5 * (1 + np.cos(2 * np.pi * idx / N))
            else:
                win = np.exp(-0.5 * ((idx - mid) / (0.3 * N)) ** 2)
            
            h = dc_offset + zc_imag * win
            s = np.sum(h)
            if abs(s) > 1e-10:
                candidate_list.append(h / s)
    
    # Evaluate structured candidates
    for h in candidate_list:
        if time.time() > deadline - 80:
            break
        try_candidate(h)

    # Phase 2: DCT-based low-frequency optimization
    # Represent h as sum of first M DCT basis functions
    M_dct = 30  # Number of DCT basis functions
    
    # Precompute DCT basis
    dct_basis = np.zeros((N, M_dct))
    for k in range(M_dct):
        if k == 0:
            dct_basis[:, k] = 1.0 / np.sqrt(N)
        else:
            dct_basis[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi * (idx + 0.5) * k / N)
    
    # DCT coefficients for a signal h: c = dct_basis.T @ h
    # Reconstruct: h = dct_basis @ c
    
    def dct_smooth_obj(c, beta=200.0):
        h = dct_basis @ c
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        h_norm = h / s
        conv = _fft_convolve(h_norm)
        m = np.max(conv)
        shifted = (conv - m) * beta
        shifted = np.clip(shifted, -50, 50)
        sm = m + np.log(np.sum(np.exp(shifted))) / beta
        return sm

    # Optimize DCT coefficients with L-BFGS-B
    if time.time() < deadline - 50:
        # Initialize with DC component dominant
        c0 = np.zeros(M_dct)
        c0[0] = 1.0 / np.sqrt(N)  # This gives h ≈ constant
        
        # Try multiple initializations
        for init_trial in range(5):
            if time.time() > deadline - 30:
                break
            
            if init_trial == 0:
                c_init = c0.copy()
            else:
                # Add some random low-frequency content
                c_init = c0.copy()
                for k in range(1, min(10, M_dct)):
                    c_init[k] = np.random.uniform(-0.1, 0.1)
            
            try:
                res = minimize(
                    dct_smooth_obj,
                    c_init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 1500, 'gtol': 1e-8, 'beta': 200.0}
                )
                
                c_opt = res.x
                h_opt = dct_basis @ c_opt
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-10:
                    h_opt = h_opt / s_opt
                    try_candidate(h_opt)
            except Exception:
                pass

    # Phase 3: Full 400-dim refinement with smooth-max
    def soft_max(x, beta=200.0):
        m = np.max(x)
        shifted = (x - m) * beta
        shifted = np.clip(shifted, -50, 50)
        return m + np.log(np.sum(np.exp(shifted))) / beta

    def local_refine_full(h0, max_iter=300, beta=200.0):
        if time.time() > deadline - 15:
            return None

        def obj(x):
            conv = _fft_convolve(x)
            sm = soft_max(conv, beta=beta)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty

        try:
            res = minimize(
                obj,
                h0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'gtol': 1e-9}
            )
            return res.x
        except Exception:
            return None

    # Refine best solution
    if time.time() < deadline - 20:
        h_best = best_h.copy()
        s_best = np.sum(h_best)
        if abs(s_best) > 1e-10:
            h_best = h_best / s_best

        # Progressive beta refinement
        for beta_val in [100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 10:
                break
            h_refined = local_refine_full(h_best, max_iter=200, beta=beta_val)
            if h_refined is not None:
                try_candidate(h_refined)
                s_ref = np.sum(h_refined)
                if abs(s_ref) > 1e-10:
                    h_best = h_refined / s_ref

        # Perturbation-based restarts
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation in DCT space
            c_current = dct_basis.T @ h_best
            c_pert = c_current + 0.05 * np.random.randn(M_dct)
            h_pert = dct_basis @ c_pert
            s_pert = np.sum(h_pert)
            if abs(s_pert) > 1e-10:
                h_pert = h_pert / s_pert
                h_refined = local_refine_full(h_pert, max_iter=150, beta=200.0)
                if h_refined is not None:
                    try_candidate(h_refined)

    # Phase 4: Final Nelder-Mead polish
    if time.time() < deadline - 3:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        def obj_exact(x):
            conv = _fft_convolve(x)
            m = np.max(conv)
            s = np.sum(x)
            return m + 500.0 * (s - 1.0) ** 2

        try:
            res = minimize(
                obj_exact,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            h_final = res.x
            try_candidate(h_final)
        except Exception:
            pass

    return best_h
