import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    # Search 2-point distributions [a, 1-a]
    # We can place them at any indices, but due to shift invariance of the ratio 
    # (if the domain is infinite or boundaries don't matter), position might not matter.
    # However, with finite N and 'full' convolution, boundaries matter.
    # Placing them in the center is usually best to avoid boundary effects reducing the conv max?
    # Actually, for 'full' convolution, the length is 2N-1.
    # If we put mass at the edges, the convolution is truncated? No, 'full' includes everything.
    # But if mass is at 0, the conv starts at 0.
    # Let's try positions: center, edge.
    
    candidates = []
    
    # 1. Delta (already best_h)
    
    # 2. Two points
    # Try a range of 'a' from 0 to 1
    # And positions: (0, 1), (N//2 - 1, N//2), (0, N//2)
    for pos_pair in [(0, 1), (N//2 - 1, N//2), (0, N//2)]:
        for a in np.linspace(0.01, 0.99, 200):
            h = np.zeros(N)
            h[pos_pair[0]] = a
            h[pos_pair[1]] = 1.0 - a
            sc = compute_ratio(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()
            if time.time() > deadline - 10.0:
                return list(best_h)

    # 3. Three points
    # Try symmetric 3-point: [a, b, a] or [a, 1-2a, a]
    # Positions: center
    center = N // 2
    for a in np.linspace(0.01, 0.49, 100):
        b = 1.0 - 2.0 * a
        if b < 0: continue
        h = np.zeros(N)
        h[center-1] = a
        h[center] = b
        h[center+1] = a
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        if time.time() > deadline - 10.0:
            return list(best_h)
            
    # 4. Gaussian-like
    # Vary sigma
    x = np.arange(N) - N // 2
    for sigma in np.linspace(0.5, 10.0, 50):
        h = np.exp(-0.5 * (x/sigma)**2)
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        if time.time() > deadline - 10.0:
            return list(best_h)

    return list(best_h)
