"""Rewritten program for the autocorrelation inequality task."""
import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-15:
            return float('inf')
        c = np.convolve(h, h)
        return abs(2 * N * np.max(c) / (s * s))
    
    best_h = np.ones(N)
    best_val = evaluate(best_h)
    
    # Try chirp signals: h[k] = cos(k * theta)
    # Optimize theta
    thetas = np.linspace(0, 2 * np.pi, 2000)
    for theta in thetas:
        if time.time() > deadline:
            break
        h = np.cos(np.arange(N) * theta)
        s = np.sum(h)
        if abs(s) < 1e-15:
            continue
        c = np.convolve(h, h)
        val = abs(2 * N * np.max(c) / (s * s))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Also try sine
    for theta in thetas:
        if time.time() > deadline:
            break
        h = np.sin(np.arange(N) * theta)
        s = np.sum(h)
        if abs(s) < 1e-15:
            continue
        c = np.convolve(h, h)
        val = abs(2 * N * np.max(c) / (s * s))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Try random perturbations of the best
    rng = np.random.RandomState(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        h = best_h + 0.1 * rng.randn(N)
        s = np.sum(h)
        if abs(s) < 1e-15:
            continue
        c = np.convolve(h, h)
        val = abs(2 * N * np.max(c) / (s * s))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
