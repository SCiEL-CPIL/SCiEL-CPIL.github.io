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
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Baseline: 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0/3.0
    h2[1] = 1.0/3.0
    try_candidate(h2)
    
    # Baseline: uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)
    
    # Structural approach: search over different "shapes" using parameterized families
    # and refine each with coordinate descent
    
    def coordinate_descent(h, max_iter=200):
        """Cyclic coordinate descent for maximizing the ratio."""
        h = h.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a few perturbations
                for delta in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
                    h[i] = max(0.0, old + delta)
                    s = compute_score(h)
                    if s > best_score:
                        improved = True
                # Also try scaling: h[i] = old * factor
                for factor in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
                    h[i] = max(0.0, old * factor)
                    s = compute_score(h)
                    if s > best_score:
                        improved = True
                if not improved:
                    h[i] = old
            if not improved:
                break
        return h
    
    # Try various structured initializations
    structures = []
    
    # 1. Ramp functions
    for slope in [0.5, 1.0, 2.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # 2. Symmetric bump
    center = N // 2
    for width in [5, 10, 15, 20, 25]:
        h = np.zeros(N)
        for i in range(max(0, center-width), min(N, center+width+1)):
            h[i] = np.exp(-((i-center)**2) / (2*width**2))
        structures.append(h)
    
    # 3. Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            structures.append(h)
    
    # 4. Random sparse patterns
    np.random.seed(42)
    for sparsity in [0.1, 0.2, 0.3, 0.5]:
        for _ in range(5):
            h = np.zeros(N)
            n_nonzero = max(1, int(N * sparsity))
            idx = np.random.choice(N, n_nonzero, replace=False)
            h[idx] = np.random.uniform(0.1, 1.0, n_nonzero)
            structures.append(h)
    
    # 5. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i+1)**(-alpha)
        structures.append(h)
    
    # 6. Two-bump structures
    for pos1 in [5, 10, 15, 20, 25]:
        for pos2 in [30, 35, 40, 45]:
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            structures.append(h)
    
    # 7. Cosine windows
    for freq in [1, 2, 3, 4]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            structures.append(h)
    
    # Refine each structure with coordinate descent
    for h_init in structures:
        if time.time() > deadline - 10:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        h_refined = coordinate_descent(h_init, max_iter=50)
        try_candidate(h_refined)
    
    # Final: try a more intensive coordinate descent on the best found
    if time.time() < deadline - 5:
        h_final = coordinate_descent(best_h, max_iter=500)
        try_candidate(h_final)
    
    return list(best_h)
