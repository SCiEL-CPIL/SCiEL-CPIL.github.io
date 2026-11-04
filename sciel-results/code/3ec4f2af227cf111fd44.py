"""Minimize normalized max autoconvolution via blocky binary structures."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 145.0

def _compute_score_fast(h):
    """Compute normalized max autoconvolution using FFT for speed."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    energy = np.sum(h ** 2)
    if energy < 1e-15:
        return np.inf
    # Use FFT to compute all lags at once
    # Autoconvolution: c[lag] = sum_i h[i]*h[i+lag] for lag>=0
    # Use FFT: ifft(|fft(h)|^2) gives circular autoconvolution
    # We need linear autoconvolution for lags 0..N-1
    # Pad to avoid wrap-around
    h_padded = np.zeros(2 * N - 1)
    h_padded[:N] = h
    H = np.fft.rfft(h_padded)
    autoconv = np.fft.irfft(H * np.conj(H), n=2*N-1)
    # autoconv[lag] = sum_i h[i]*h[i+lag] for lag in 0..N-1
    max_val = np.max(autoconv[:N])
    return max_val / energy

def _params_to_heights_block(params, K):
    """Convert parameters to blocky heights.
    
    params layout: for each block k in range(K):
      - pos_k: center position in [0, 0.5) (symmetric pair at ±pos)
      - width_k: half-width in (0, 0.5]
      - height_k: height in [0, max_h]
    
    Total params: 3*K
    """
    h = np.zeros(N)
    for k in range(K):
        pos = params[3*k] * 0.5  # center in [0, 0.25)
        half_w = params[3*k+1] * 0.25  # half-width in (0, 0.125]
        height = params[3*k+2]  # height
        
        # Two symmetric blocks: at +pos and -pos (i.e., pos and 0.5-pos in [0,1) normalized)
        # Interval is [-0.25, 0.25], mapped to [0, N)
        # pos in [0, 0.25) corresponds to index range
        # Block at +pos: from pos-half_w to pos+half_w
        # Block at -pos: from -pos-half_w to -pos+half_w (i.e., 0.25-(-pos+half_w) to 0.25-(-pos-half_w) in [0,0.25])
        
        # Convert to index ranges
        # Normalized position in [0, 0.5) for the interval [0, 0.5]
        # Actually, let's work in [0, 1) where the full interval [-0.25, 0.25] maps to [0, 1)
        # pos_norm in [0, 0.5) -> actual position -0.25 + pos_norm*0.5
        # Simpler: work directly in index space
        
        # pos is in [0, 0.25), representing offset from center (index N/2 = 300)
        # Block 1: centered at N/2 + pos*N/0.25... no, let me redo.
        
        # The interval [-0.25, 0.25] has 600 indices.
        # Index i corresponds to position -0.25 + (i+0.5)*0.5/600
        # Let's just use normalized [0,1) where 0=-0.25, 1=0.25
        # pos in [0, 0.5) means position -0.25 + pos*0.5
        # half_w in (0, 0.125] means width 2*half_w*0.5 = half_w*0.5 in position units
        
        # Block at +pos: from pos - half_w to pos + half_w (in [0,1) coords)
        # Block at -pos: from 1 - pos - half_w to 1 - pos + half_w (in [0,1) coords)
        
        lo1 = max(0.0, pos - half_w)
        hi1 = min(1.0, pos + half_w)
        lo2 = max(0.0, 1.0 - pos - half_w)
        hi2 = min(1.0, 1.0 - pos + half_w)
        
        i1_start = int(lo1 * N)
        i1_end = int(hi1 * N)
        i2_start = int(lo2 * N)
        i2_end = int(hi2 * N)
        
        h[i1_start:i1_end] += height
        h[i2_start:i2_end] += height
    
    return h

def construct():
    t0 = time.time()
    best_score = np.inf
    best_heights = np.ones(N)
    
    # Phase 1: Blocky structure with K=4 blocks
    K = 4
    bounds = []
    for _ in range(K):
        bounds.append((0.0, 0.5))    # pos
        bounds.append((0.01, 0.25))  # half_w
        bounds.append((0.0, 3.0))    # height
    
    def score_block(params):
        if time.time() - t0 > DEADLINE - 2:
            return 1e10
        h = _params_to_heights_block(params, K)
        return _compute_score_fast(h)
    
    try:
        res = differential_evolution(score_block, bounds,
                                       maxiter=80, popsize=25, seed=42, tol=1e-5, polish=True)
        if res.fun < best_score:
            best_score = res.fun
            best_heights = _params_to_heights_block(res.x, K)
    except Exception:
        pass
    
    if time.time() - t0 > DEADLINE - 20:
        return list(best_heights)
    
    # Phase 2: Blocky structure with K=6 blocks
    K2 = 6
    bounds2 = []
    for _ in range(K2):
        bounds2.append((0.0, 0.5))
        bounds2.append((0.01, 0.25))
        bounds2.append((0.0, 3.0))
    
    def score_block2(params):
        if time.time() - t0 > DEADLINE - 2:
            return 1e10
        h = _params_to_heights_block(params, K2)
        return _compute_score_fast(h)
    
    try:
        res2 = differential_evolution(score_block2, bounds2,
                                       maxiter=50, popsize=20, seed=123, tol=1e-5, polish=True)
        if res2.fun < best_score:
            best_score = res2.fun
            best_heights = _params_to_heights_block(res2.x, K2)
    except Exception:
        pass
    
    if time.time() - t0 > DEADLINE - 10:
        return list(best_heights)
    
    # Phase 3: Blocky with K=3 blocks (fewer, wider blocks)
    K3 = 3
    bounds3 = []
    for _ in range(K3):
        bounds3.append((0.0, 0.5))
        bounds3.append((0.05, 0.4))
        bounds3.append((0.0, 5.0))
    
    def score_block3(params):
        if time.time() - t0 > DEADLINE - 2:
            return 1e10
        h = _params_to_heights_block(params, K3)
        return _compute_score_fast(h)
    
    try:
        res3 = differential_evolution(score_block3, bounds3,
                                       maxiter=60, popsize=20, seed=777, tol=1e-5, polish=True)
        if res3.fun < best_score:
            best_score = res3.fun
            best_heights = _params_to_heights_block(res3.x, K3)
    except Exception:
        pass
    
    return list(best_heights)
