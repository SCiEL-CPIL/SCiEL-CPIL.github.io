import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
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
    deadline = t0 + 140.0  # Safety margin for 150s kill
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N//2] = 1.0
    best_score = compute_ratio(best_h)
    
    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Efficient parametric searches
    
    # 1. Two-point distributions - focus on relative positions
    # Due to shift invariance of the ratio, only relative position matters
    # Try key separations with fine grid on weights
    for d in range(1, min(10, N)):
        if time_left() < 30.0:
            return list(best_h)
        a_vals = np.linspace(0.05, 0.95, 50)
        for a in a_vals:
            h = np.zeros(N)
            h[N//2 - d//2] = a
            h[N//2 - d//2 + d] = 1.0 - a
            sc = compute_ratio(h)
            update_best(h, sc)
    
    # 2. Three-point symmetric distributions
    for d in range(1, 8):
        if time_left() < 30.0:
            return list(best_h)
        for a in np.linspace(0.05, 0.4, 30):
            b = 1.0 - 2.0 * a
            if b < 0.01:
                continue
            h = np.zeros(N)
            center = N // 2
            h[center - d] = a
            h[center] = b
            h[center + d] = a
            sc = compute_ratio(h)
            update_best(h, sc)
    
    # 3. Sparse sets - Sidon-like and random
    rng = np.random.default_rng(42)
    
    def find_sidon_set_greedy(n_points, start_idx=0):
        A = []
        used_sums = set()
        for x in range(start_idx, N):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if valid and 2 * x not in used_sums:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
            if len(A) >= n_points:
                break
        return A
    
    # Try various sparse sets
    for k in [2, 3, 4, 5, 6]:
        if time_left() < 25.0:
            return list(best_h)
        # Sidon sets
        A = find_sidon_set_greedy(k)
        if len(A) >= k:
            h = np.zeros(N)
            for idx in A:
                h[idx] = 1.0
            sc = compute_ratio(h)
            update_best(h, sc)
        
        # Random sparse sets
        for _ in range(20):
            idx = rng.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            sc = compute_ratio(h)
            update_best(h, sc)
    
    # 4. Gaussian-like and other smooth shapes
    x = np.arange(N) - N // 2
    if time_left() < 20.0:
        return list(best_h)
    for sigma in np.linspace(1.0, 8.0, 30):
        h = np.exp(-0.5 * (x/sigma)**2)
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # 5. Triangular shapes
    for width in range(3, 15, 2):
        h = np.zeros(N)
        center = N // 2
        for i in range(-width//2, width//2 + 1):
            h[center + i] = 1.0 - abs(i) / (width//2 + 1)
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 2: Local optimization with coordinate descent
    def local_optimize(h_init, max_iter=100):
        h = h_init.copy()
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return h, 0.0
        h = h / s
        
        current_score = compute_ratio(h)
        best_local_h = h.copy()
        best_local_score = current_score
        
        # Coordinate descent with line search
        for _ in range(max_iter):
            if time_left() < 15.0:
                break
            
            improved = False
            # Try adjusting each coordinate
            for i in range(N):
                if time_left() < 15.0:
                    break
                
                # Try increasing h[i]
                for delta in [0.01, 0.05, 0.1]:
                    h_trial = h.copy()
                    h_trial[i] += delta
                    h_trial = np.maximum(h_trial, 0.0)
                    s_trial = np.sum(h_trial)
                    if s_trial < 1e-15:
                        continue
                    h_trial = h_trial / s_trial
                    sc_trial = compute_ratio(h_trial)
                    if sc_trial > current_score:
                        h = h_trial
                        current_score = sc_trial
                        improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_h = h.copy()
                        break
                
                # Try decreasing h[i]
                for delta in [0.01, 0.05, 0.1]:
                    h_trial = h.copy()
                    h_trial[i] -= delta
                    h_trial = np.maximum(h_trial, 0.0)
                    s_trial = np.sum(h_trial)
                    if s_trial < 1e-15:
                        continue
                    h_trial = h_trial / s_trial
                    sc_trial = compute_ratio(h_trial)
                    if sc_trial > current_score:
                        h = h_trial
                        current_score = sc_trial
                        improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_h = h.copy()
                        break
            
            if not improved:
                # Try random perturbations
                for _ in range(10):
                    h_trial = h.copy()
                    idx = rng.choice(N, size=2, replace=False)
                    h_trial[idx[0]] += rng.uniform(-0.05, 0.05)
                    h_trial[idx[1]] += rng.uniform(-0.05, 0.05)
                    h_trial = np.maximum(h_trial, 0.0)
                    s_trial = np.sum(h_trial)
                    if s_trial < 1e-15:
                        continue
                    h_trial = h_trial / s_trial
                    sc_trial = compute_ratio(h_trial)
                    if sc_trial > current_score:
                        h = h_trial
                        current_score = sc_trial
                        improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_h = h.copy()
                        break
                if not improved:
                    break
        
        return best_local_h, best_local_score
    
    # Refine the global best
    h_opt, sc_opt = local_optimize(best_h)
    update_best(h_opt, sc_opt)
    
    # Refine from promising starts
    if time_left() > 20.0:
        # From 2-point [2/3, 1/3]
        h_2pt = np.zeros(N)
        h_2pt[N//2] = 2.0/3.0
        h_2pt[N//2 + 1] = 1.0/3.0
        h_opt, sc_opt = local_optimize(h_2pt, max_iter=50)
        update_best(h_opt, sc_opt)
    
    if time_left() > 15.0:
        # From random sparse starts
        for _ in range(10):
            if time_left() < 15.0:
                break
            k = rng.integers(2, 7)
            idx = rng.choice(N, size=k, replace=False)
            h_rand = np.zeros(N)
            h_rand[idx] = 1.0
            h_opt, sc_opt = local_optimize(h_rand, max_iter=30)
            update_best(h_opt, sc_opt)
    
    return list(best_h)
