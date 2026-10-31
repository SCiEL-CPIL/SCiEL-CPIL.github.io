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

def coordinate_descent(h, max_iter=5000, tol=1e-10):
    """Local optimization via coordinate descent with adaptive steps."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h, 0.0
    
    best_score = compute_ratio(h)
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            for step in step_sizes:
                # Try increasing h[i]
                h_old = h[i]
                h[i] = h_old + step
                sc = compute_ratio(h)
                if sc > best_score + tol:
                    best_score = sc
                    improved = True
                    # Don't break; keep trying larger steps
                else:
                    h[i] = h_old
                    break
                
                # Try decreasing h[i]
                h[i] = h_old - step
                if h[i] < 0:
                    h[i] = 0
                sc = compute_ratio(h)
                if sc > best_score + tol:
                    best_score = sc
                    improved = True
                else:
                    h[i] = h_old
                    break
        
        if not improved:
            break
        if iteration % 100 == 0 and time.time() > time.time():  # placeholder for time check
            break
    
    return h, best_score

def smooth_max_surrogate(h, temperature=0.01):
    """Compute a smooth approximation of the max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    # Smooth max via log-sum-exp
    max_val = np.max(conv)
    if max_val < 1e-15:
        return 0.0, 0.0
    # log-sum-exp trick for numerical stability
    shifted = conv / max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted - 1.0)))
    # The smooth max is approximately max_val * (1 + small correction)
    # For our purposes, use the actual max but with a penalty for spread
    conv_inf = max_val
    return h2_sq / conv_inf, conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    # Phase 1: Parametric family search (same as before)
    
    # 2. Two points
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
    x = np.arange(N) - N // 2
    for sigma in np.linspace(0.5, 10.0, 50):
        h = np.exp(-0.5 * (x/sigma)**2)
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        if time.time() > deadline - 10.0:
            return list(best_h)

    # Phase 2: Local optimization via coordinate descent
    if time.time() < deadline - 20.0:
        h_opt, sc_opt = coordinate_descent(best_h, max_iter=3000)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt

    # Phase 3: Try multiple random restarts with local optimization
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(42)
        for restart in range(5):
            # Generate a random starting point
            h_rand = np.abs(rng.randn(N))
            h_rand /= np.sum(h_rand) if np.sum(h_rand) > 0 else 1.0
            sc_rand = compute_ratio(h_rand)
            
            if sc_rand > best_score * 0.9:  # Only optimize promising starts
                h_opt, sc_opt = coordinate_descent(h_rand, max_iter=2000)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt
            
            if time.time() > deadline - 10.0:
                break

    # Phase 4: Final polish with finer steps
    if time.time() < deadline - 5.0:
        h_final = best_h.copy()
        for i in range(N):
            for step in [0.001, 0.0005, 0.0002, 0.0001]:
                h_test = h_final.copy()
                h_test[i] += step
                sc = compute_ratio(h_test)
                if sc > best_score:
                    best_score = sc
                    h_final = h_test.copy()
                
                h_test = h_final.copy()
                h_test[i] = max(0, h_test[i] - step)
                sc = compute_ratio(h_test)
                if sc > best_score:
                    best_score = sc
                    h_final = h_test.copy()
        
        best_h = h_final

    return list(best_h)
