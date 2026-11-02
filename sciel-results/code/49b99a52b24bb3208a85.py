import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        # Normalize to L1=1 to ensure consistent comparison if evaluator doesn't, 
        # or if evaluator does, this is harmless.
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h_norm = h / l1
        s = compute_score(h_norm)
        if s > best_score:
            best_score = s
            best_h = h_norm.copy()
        return s

    # 1. Top-Hat Strategy:
    # Try all possible widths K from 1 to N
    # For a top-hat of width K, the ratio is 1 (if normalized).
    # But maybe non-uniform top-hats (trapezoids) do better?
    
    # 2. Structured Search:
    # We will use coordinate descent on a few good initializations.
    
    # Initialization 1: Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    try_candidate(h_spike)
    
    # Initialization 2: Uniform
    h_uni = np.ones(N)
    try_candidate(h_uni)
    
    # Initialization 3: Top-hats of various widths
    for K in range(1, N+1):
        h_th = np.zeros(N)
        h_th[:K] = 1.0
        try_candidate(h_th)
        
    # Initialization 4: Two spikes
    for gap in range(1, N):
        h_2s = np.zeros(N)
        h_2s[0] = 1.0
        h_2s[gap] = 1.0
        try_candidate(h_2s)
        
    # Refinement: Coordinate Descent
    def coordinate_descent(h, max_iter=100):
        h = h.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try scaling this element
                for factor in [0.5, 0.8, 1.2, 2.0, 0.0]:
                    h[i] = old * factor
                    s = compute_score(h)
                    if s > best_score:
                        improved = True
                if not improved:
                    h[i] = old
            if not improved:
                break
        return h

    # Run refinement on the best candidate found so far
    if time.time() < deadline - 10:
        h_refined = coordinate_descent(best_h, max_iter=50)
        try_candidate(h_refined)

    return list(best_h)
