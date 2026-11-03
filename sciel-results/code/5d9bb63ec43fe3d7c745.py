import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast evaluation
# Using a fine grid to accurately capture the peak of the autoconvolution
# which is piecewise linear for step functions.
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to coarse step function intervals
# (x + 0.25) / W gives the index in [0, N)
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Computes the normalized maximum of the autoconvolution.
    h: array of step heights (length N)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # Check for trivial zero function
    if np.sum(h) < 1e-15:
        return float('inf')
        
    # Expand to fine grid
    f = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    # Compute L2 norm squared
    l2_sq = np.sum(f * f) * DX
    if l2_sq < 1e-15:
        return float('inf')
        
    return float(np.max(conv) / l2_sq)

def fourier_to_h_fast(coeffs):
    """
    Converts Fourier coefficients to step function heights.
    coeffs: array of length K+1 where coeffs[0] is DC, coeffs[k] are cosine/sine pairs?
    Let's use a real-valued Fourier series representation.
    We represent f(x) = a_0 + sum_{k=1}^K (a_k cos(2*pi*k*x) + b_k sin(2*pi*k*x))
    Since the domain is [-0.25, 0.25], the fundamental frequency is 1/(0.5) = 2.
    So the terms are cos(4*pi*k*x) and sin(4*pi*k*x).
    
    To ensure non-negativity, we will add a large DC offset if needed, or clip.
    Clipping is simpler and robust.
    """
    K = len(coeffs) // 2
    # coeffs layout: [a0, a1, b1, a2, b2, ..., aK, bK]
    # Actually, let's just use a simple array of coefficients for a polynomial-like basis or just Fourier.
    # Let's stick to the inspiration: "Represent the step function entirely by its first K Fourier coefficients"
    
    # Let's define the basis functions on X
    # f(x) = c_0 + sum_{k=1}^K c_k * cos(4*pi*k*x) + d_k * sin(4*pi*k*x)
    # coeffs = [c_0, c_1, d_1, c_2, d_2, ..., c_K, d_K]
    
    K = (len(coeffs) - 1) // 2
    h = np.zeros(N)
    h += coeffs[0]
    
    for k in range(1, K + 1):
        idx_c = 1 + 2*(k-1)
        idx_s = 2 + 2*(k-1)
        if idx_s < len(coeffs):
            h += coeffs[idx_c] * np.cos(4 * np.pi * k * X)
            h += coeffs[idx_s] * np.sin(4 * np.pi * k * X)
            
    # Enforce non-negativity
    h = np.clip(h, 0.0, None)
    
    # If the function is identically zero, return a small constant to avoid division by zero
    if np.sum(h) < 1e-15:
        h = np.ones(N) * 1e-6
        
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Number of Fourier modes
    # K=5 means 11 parameters. K=10 means 21 parameters.
    # Let's try K=8 (17 parameters) for a balance of flexibility and optimization difficulty.
    K = 8
    n_params = 1 + 2*K
    
    # Objective function for scipy
    def objective(coeffs):
        h = fourier_to_h_fast(coeffs)
        return ratio_from_h(h)
    
    # Initial guesses
    # 1. Gaussian-like bump (approximated by Fourier)
    # A Gaussian in time is Gaussian in frequency, so a few low-frequency terms should work well.
    # Let's start with a DC term and a few cosine terms.
    
    # We want a bump centered at 0.
    # f(x) ~ exp(-x^2/w^2).
    # Let's just use random initializations and let the optimizer find the structure.
    
    best_h = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    max_restarts = 15
    
    for r in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Initialize coefficients
        # DC term should be positive and relatively large to keep most of the function positive
        c0 = rng.uniform(0.5, 2.0)
        # Other coefficients smaller
        others = rng.normal(0, 0.5, size=n_params - 1)
        coeffs0 = np.concatenate(([c0], others))
        
        # Try L-BFGS-B
        try:
            res = minimize(objective, coeffs0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False})
            
            if res.success or res.fun < best_val:
                h = fourier_to_h_fast(res.x)
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_h = h
        except Exception as e:
            pass
            
    # If we didn't find anything valid, fallback to a simple Gaussian
    if best_h is None:
        h = np.exp(-((X - 0.0) / 0.05) ** 2)
        best_h = h
        best_val = ratio_from_h(h)
        
    # Final check: if the best value is still bad, try a few more specific hand-crafted functions
    # e.g. a triangular bump
    tri = np.clip(1 - np.abs(X)/0.1, 0, None)
    v_tri = ratio_from_h(tri)
    if v_tri < best_val:
        best_val = v_tri
        best_h = tri
        
    # A boxcar function
    box = np.where(np.abs(X) < 0.1, 1.0, 0.0)
    v_box = ratio_from_h(box)
    if v_box < best_val:
        best_val = v_box
        best_h = box
        
    return list(best_h)

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print(f"Score: {ratio_from_h(h)}")
