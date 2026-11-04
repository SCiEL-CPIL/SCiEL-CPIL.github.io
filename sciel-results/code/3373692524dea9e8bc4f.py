"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    h = 0.5 / N
    centers = -0.25 + (np.arange(N) + 0.5) * h
    
    def compute_score(heights):
        n_fft = 2048
        F = np.fft.rfft(heights, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * h
        max_conv = np.max(conv)
        total_mass = np.sum(heights) * h
        return max_conv / (total_mass ** 2)
    
    best_val = float('inf')
    best_heights = None
    
    # Family 1: Gaussians
    sigmas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
    for sigma in sigmas:
        heights = np.exp(-centers**2 / (2 * sigma**2))
        val = compute_score(heights)
        if val < best_val:
            best_val = val
            best_heights = heights.copy()
        if time.time() > deadline - 10:
            break
    
    # Family 2: Super-Gaussians (exp(-|x|^p / sigma^p))
    if time.time() < deadline - 10:
        for p in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
            for sigma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
                heights = np.exp(-np.abs(centers)**p / (sigma**p))
                val = compute_score(heights)
                if val < best_val:
                    best_val = val
                    best_heights = heights.copy()
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
    
    # Family 3: Truncated uniform (top-hat)
    if time.time() < deadline - 10:
        for width in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            heights = np.where(np.abs(centers) <= width, 1.0, 0.0)
            val = compute_score(heights)
            if val < best_val:
                best_val = val
                best_heights = heights.copy()
            if time.time() > deadline - 10:
                break
    
    # Family 4: B-spline-like smooth profiles
    if time.time() < deadline - 10:
        for width in [0.08, 0.10, 0.12, 0.15, 0.20]:
            for smooth in [0.02, 0.03, 0.05]:
                heights = np.where(np.abs(centers) <= width, 
                                  np.exp(-(np.abs(centers) - width)**2 / smooth**2), 
                                  np.exp(-width**2 / smooth**2))
                heights = np.maximum(heights, 0)
                val = compute_score(heights)
                if val < best_val:
                    best_val = val
                    best_heights = heights.copy()
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
    
    # Local optimization with adaptive step size
    if best_heights is not None and time.time() < deadline - 5:
        current = best_heights.copy()
        current_val = best_val
        np.random.seed(42)
        
        # Multi-scale local optimization
        scales = [0.05, 0.02, 0.01, 0.005, 0.002]
        for scale in scales:
            if time.time() > deadline - 5:
                break
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                mask = np.random.random(N) < 0.03
                if mask.sum() == 0:
                    continue
                perturbation = (np.random.random(mask.sum()) - 0.5) * scale
                new_heights = current.copy()
                new_heights[mask] += perturbation
                new_heights = np.maximum(new_heights, 0)
                
                val = compute_score(new_heights)
                if val < current_val:
                    current = new_heights
                    current_val = val
                    if current_val < best_val:
                        best_val = current_val
                        best_heights = current.copy()
    
    if best_heights is None:
        return [1.0] * N
    return list(best_heights)
# EVOLVE-BLOCK-END
