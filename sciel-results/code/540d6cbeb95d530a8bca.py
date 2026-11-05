"""Combined strategy: sparse-spike parameterization + soft-max surrogate + FFT autoconv + L-BFGS-B refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    sum_exp = np.sum(np.exp(beta * x_shifted))
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _sparse_to_h(params, n_spikes=20):
    """
    Convert sparse parameterization to full height vector.
    params layout: [background (1), spike_amplitudes (n_spikes), spike_positions (n_spikes)]
    Spike positions are in [0, N), mapped to nearest integer index.
    Multiple spikes at same index sum their amplitudes.
    """
    bg = params[0]
    n = n_spikes
    amps = params[1:1+n]
    pos = params[1+n:1+2*n]
    
    h = np.full(N, bg)
    # Map positions to integer indices
    idx = np.round(pos).astype(int)
    idx = np.clip(idx, 0, N-1)
    # Accumulate spike amplitudes at their positions
    np.add.at(h, idx, amps)
    return h

def _objective_sparse(params, beta, n_spikes=20):
    """Soft-max objective for sparse parameterization."""
    h = _sparse_to_h(params, n_spikes)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _objective_full(params, beta):
    """Soft-max objective for full h vector."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # === Phase 1: Sparse-spike grid search ===
    # Parameterize as background + small number of spikes
    n_spikes = 15
    t = np.arange(N)
    
    # Try various backgrounds and spike configurations
    bg_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    spike_amp_fracs = [0.5, 1.0, 2.0, 3.0, 5.0]
    freqs = [1, 2, 3, 5, 7, 10]
    
    for bg in bg_values:
        for freq in freqs:
            for amp_frac in spike_amp_fracs:
                if time.time() > deadline - 40.0:
                    break
                
                # Create periodic spike pattern
                spike_positions = np.array([int((i * N) / freq) % N for i in range(freq)])
                spike_amps = np.full(freq, bg * amp_frac)
                
                h = np.full(N, bg)
                np.add.at(h, spike_positions, spike_amps)
                try_h(h)
                
                # Also try anti-phase spikes
                spike_amps_neg = np.full(freq, -bg * amp_frac)
                h_neg = np.full(N, bg)
                np.add.at(h_neg, spike_positions, spike_amps_neg)
                try_h(h_neg)
    
    # === Phase 2: Optimize sparse parameterization with L-BFGS-B ===
    # Build initial sparse params from best_h
    # Estimate background as mean, spikes as deviations
    bg_est = np.mean(best_h)
    # Find top deviation indices
    devs = best_h - bg_est
    top_idx = np.argsort(np.abs(devs))[-n_spikes:]
    
    init_params = np.zeros(1 + 2 * n_spikes)
    init_params[0] = bg_est
    for i, idx in enumerate(top_idx):
        init_params[1 + i] = devs[idx]
        init_params[1 + n_spikes + i] = float(idx)
    
    # Optimize sparse params with soft-max
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                lambda p: _objective_sparse(p, beta, n_spikes),
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = _sparse_to_h(res.x, n_spikes)
            val_opt = _evaluate_exact(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                # Update init_params for next beta
                bg_est = np.mean(best_h)
                devs = best_h - bg_est
                top_idx = np.argsort(np.abs(devs))[-n_spikes:]
                init_params = np.zeros(1 + 2 * n_spikes)
                init_params[0] = bg_est
                for i, idx in enumerate(top_idx):
                    init_params[1 + i] = devs[idx]
                    init_params[1 + n_spikes + i] = float(idx)
        except Exception:
            pass
    
    # === Phase 3: Full-vector L-BFGS-B refinement with soft-max ===
    # Two-stage: low beta then high beta
    for beta in [50.0, 200.0]:
        if time.time() > deadline - 20.0:
            break
        
        # Multiple restarts
        for restart in range(4):
            if time.time() > deadline - 20.0:
                break
            
            if restart == 0:
                x0 = best_h.copy()
            else:
                noise_scale = 0.05 * np.std(best_h) * (1 + 0.3 * restart)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            
            try:
                res = minimize(
                    lambda x: _objective_full(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = res.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # === Phase 4: Coordinate descent polish on exact objective ===
    if time.time() < deadline - 5.0:
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            # Try random coordinate perturbations
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                idx = rng.integers(0, N)
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    h_new = best_h.copy()
                    h_new[idx] += delta
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-12:
                        best_val = val_new
                        best_h = h_new
                        improved = True
                        break
    
    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
