import numpy as np
import time

N = 600
DELTA = (1/2) / N

def compute_autoconvolution(h):
    """Compute the full autoconvolution (autocorrelation) of step function h using FFT."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    autocorr = autocorr[:n]
    return autocorr * DELTA

def objective_l2(h):
    """Minimize max(autoconvolution) / (L2_norm)^2"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l2_sq = np.sum(h**2) * DELTA
    if l2_sq < 1e-10:
        return 1e10
    autocorr = compute_autoconvolution(h)
    max_ac = np.max(autocorr)
    return max_ac / l2_sq

def objective_l1(h):
    """Minimize max(autoconvolution) / (L1_norm)^2"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l1 = np.sum(h) * DELTA
    if l1 < 1e-10:
        return 1e10
    autocorr = compute_autoconvolution(h)
    max_ac = np.max(autocorr)
    return max_ac / (l1 * l1)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = np.ones(N)
    # Evaluate both to see which one is "better" relative to seed
    # Seed is uniform.
    # L1 Score for uniform: 2.
    # L2 Score for uniform: 1.
    
    # Try to optimize L2 objective with PGD
    h = np.ones(N)
    lr = 0.1
    for i in range(100):
        if time.time() > deadline - 10:
            break
        # Gradient of L2 objective is complex, use numerical or just random perturbations
        # Since L2 objective is 1 for all non-negative h, gradient is 0?
        # Actually, if max is at 0, objective is 1.
        # If max is not at 0 (impossible for h>=0), it's different.
        
        # Try asymmetric initialization
        h_trial = np.random.rand(N)
        # Project
        h_trial = np.maximum(h_trial, 0)
        
        # Check score
        score_l2 = objective_l2(h_trial)
        # If score < 1, we found something!
        
    return [float(x) for x in best_h]
