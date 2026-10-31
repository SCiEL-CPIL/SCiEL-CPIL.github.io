import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
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
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Delta at various positions
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Gaussian bumps
    x = np.arange(N)
    for center in range(0, N, 5):
        for width in [1, 2, 3, 5, 10]:
            h = np.exp(-((x - center)**2) / (2 * width**2))
            s = np.sum(h)
            if s > 0:
                candidates.append(h / s)
    
    # Two peaks
    for i in range(0, N, 5):
        for j in range(i+1, N, 5):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h / np.sum(h))
    
    # Ramp
    for start in [0, N//4, N//2]:
        for end in [N//2, 3*N//4, N-1]:
            if end > start:
                h = np.zeros(N)
                h[start:end+1] = np.linspace(1, 2, end-start+1)
                candidates.append(h / np.sum(h))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 10:
            break
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search: coordinate descent with random restarts
    def local_search(h, max_iter=5000):
        h = h.copy()
        h = h / np.sum(h)
        cur_score = compute_ratio(h)
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Pick a random coordinate to perturb
            i = np.random.randint(N)
            # Try increasing or decreasing
            for delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
                for sign in [1, -1]:
                    h_new = h.copy()
                    h_new[i] += sign * delta
                    if h_new[i] < 0:
                        continue
                    h_new = h_new / np.sum(h_new)
                    new_score = compute_ratio(h_new)
                    if new_score > cur_score:
                        cur_score = new_score
                        h = h_new
                        break
        return h, cur_score
    
    # Run local search from best candidate
    if time.time() < deadline - 10:
        h_local, score_local = local_search(best_h)
        if score_local > best_score:
            best_score = score_local
            best_h = h_local
    
    # Random restarts
    if time.time() < deadline - 10:
        np.random.seed(42)
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Random sparse candidate
            h = np.zeros(N)
            k = np.random.randint(1, 10)
            idx = np.random.choice(N, k, replace=False)
            h[idx] = np.random.dirichlet(np.ones(k))
            h_local, score_local = local_search(h, max_iter=2000)
            if score_local > best_score:
                best_score = score_local
                best_h = h_local
    
    # Final: try to sharpen the best candidate
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            i = np.random.randint(N)
            h_new = best_h.copy()
            h_new[i] *= 1.5
            h_new = h_new / np.sum(h_new)
            if compute_ratio(h_new) > best_score:
                best_score = compute_ratio(h_new)
                best_h = h_new
    
    return list(best_h)
