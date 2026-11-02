import numpy as np
import time

N = 50
DEADLINE = 130

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    linf = np.max(h)
    if linf <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (l1 * linf * conv_inf)

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    # Generate diverse initial candidates
    np.random.seed(42)
    candidates = []
    
    # Baseline
    candidates.append(best_h.copy())
    
    # Uniform
    h = np.ones(N) / N
    candidates.append(h)
    
    # Triangular pulses of various widths
    for width in [3, 5, 7, 11, 15, 20, 30]:
        h = np.zeros(N)
        center = N // 2
        half = width // 2
        for i in range(-half, half + 1):
            idx = center + i
            if 0 <= idx < N:
                h[idx] = max(0, 1.0 - abs(i) / (half + 0.5))
        if np.sum(h) > 0:
            candidates.append(h)
    
    # Random sparse patterns
    for _ in range(200):
        if elapsed() > DEADLINE - 30:
            break
        k = np.random.randint(2, 20)
        indices = np.random.choice(N, size=k, replace=False)
        amplitudes = np.random.dirichlet(np.ones(k))
        h = np.zeros(N)
        h[indices] = amplitudes
        candidates.append(h)
    
    # Random dense patterns
    for _ in range(200):
        if elapsed() > DEADLINE - 30:
            break
        h = np.random.exponential(1.0, size=N)
        candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        if elapsed() > DEADLINE - 20:
            break
        h = np.maximum(h, 0)
        if np.sum(h) <= 0:
            continue
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Coordinate ascent on the best candidate
    h = best_h.copy().astype(float)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    
    for outer in range(200):
        if elapsed() > DEADLINE - 5:
            break
        
        improved = False
        for i in range(N):
            if elapsed() > DEADLINE - 5:
                break
            if h[i] == 0 and np.random.random() > 0.1:
                continue
            
            best_val = h[i]
            best_local = compute_ratio(h)
            
            # Try a few perturbations
            for scale in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                h_new = h.copy()
                h_new[i] = max(0, h[i] * scale)
                if np.sum(h_new) <= 0:
                    continue
                s = compute_ratio(h_new)
                if s > best_local:
                    best_local = s
                    best_val = h_new[i]
            
            # Also try setting to 0 if currently nonzero
            if h[i] > 0:
                h_new = h.copy()
                h_new[i] = 0
                if np.sum(h_new) > 0:
                    s = compute_ratio(h_new)
                    if s > best_local:
                        best_local = s
                        best_val = 0.0
            
            if best_val != h[i]:
                h[i] = best_val
                improved = True
        
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        if not improved:
            break
    
    result = best_h
    if np.sum(result) <= 0:
        result = np.zeros(N)
        result[N // 2] = 1.0
    
    return list(result)
