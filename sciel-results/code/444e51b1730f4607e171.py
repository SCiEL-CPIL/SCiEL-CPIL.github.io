"""Erdos minimum overlap: coordinate descent + Nesterov with spectral initialization."""
import numpy as np
import time

N = 100
HALF_N = N / 2.0

def _overlap_matrix(h):
    """Compute overlap(k) for all circular shifts k=0..N-1.
    overlap(k) = (1/N) * sum_i h_i * (1 - h_{(i-k) % N})
    """
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H))) / N
    return 1.0 - A

def _max_overlap(h):
    return float(np.max(_overlap_matrix(h)))

def _project(h):
    """Project h onto {x in [0,1]^N : sum(x) = N/2}."""
    lo, hi = -1.0, 1.0
    if np.sum(np.clip(h - lo, 0, 1)) < HALF_N:
        lo = -10.0
    if np.sum(np.clip(h - hi, 0, 1)) > HALF_N:
        hi = 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h - mid, 0, 1))
        if s > HALF_N:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    return np.clip(h - lam, 0.0, 1.0)

def _smooth_obj_grad(h, tau):
    """Log-sum-exp smooth surrogate for max overlap. Returns (value, gradient)."""
    ov = _overlap_matrix(h)
    z = ov / tau
    zmax = np.max(z)
    lse = zmax + np.log(np.sum(np.exp(z - zmax)))
    value = tau * lse
    
    w = np.exp(z - zmax)
    w = w / np.sum(w)
    
    grad = np.zeros(N)
    for k in range(N):
        wk = w[k]
        if wk < 1e-15:
            continue
        if k == 0:
            grad += -wk * (2.0 * h / N)
        elif N % 2 == 0 and k == N // 2:
            grad += -wk * (2.0 * h / N)
        else:
            idx_minus = (np.arange(N) - k) % N
            idx_plus = (np.arange(N) + k) % N
            contrib = (h[idx_minus] + h[idx_plus]) / N
            grad += -wk * contrib
    
    return value, tau * grad

def _coordinate_descent_polish(h, max_iters=2000, deadline=None):
    """Cyclic coordinate descent with line search on each coordinate."""
    h = _project(h.copy())
    best_val = _max_overlap(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try moving coordinate j in both directions
            old_j = h[j]
            best_j = old_j
            best_obj = _max_overlap(h)
            
            # Binary search for best value of h[j]
            # We need to keep sum = HALF_N, so when we change h[j], we must adjust others
            # Actually, let's do a simpler approach: vary h[j] and re-project
            for delta in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                for sign in [1, -1]:
                    h_try = h.copy()
                    h_try[j] = np.clip(old_j + sign * delta, 0, 1)
                    h_try = _project(h_try)
                    obj = _max_overlap(h_try)
                    if obj < best_obj - 1e-10:
                        best_obj = obj
                        best_j = h_try[j]
                        h = h_try.copy()
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try smaller steps
                for delta in [0.005, 0.002, 0.001, 0.0005, 0.0001]:
                    for sign in [1, -1]:
                        h_try = h.copy()
                        h_try[j] = np.clip(old_j + sign * delta, 0, 1)
                        h_try = _project(h_try)
                        obj = _max_overlap(h_try)
                        if obj < best_obj - 1e-10:
                            best_obj = obj
                            h = h_try.copy()
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return h, best_val

def _nesterov_solve(h_init, max_iter, tau, lr, deadline):
    """Nesterov accelerated gradient to minimize smooth max overlap."""
    h = _project(h_init.copy())
    x_prev = h.copy()
    momentum = 0.0
    
    best_h = h.copy()
    best_max_ov = _max_overlap(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        momentum_new = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = h + (momentum / momentum_new) * (h - x_prev)
        
        _, grad_y = _smooth_obj_grad(y, tau)
        
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        x_prev = h
        h = h_new
        momentum = momentum_new
        
        max_ov = _max_overlap(h)
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
    
    return best_h, best_max_ov

def _spectral_init(freq, phase):
    """Generate h from spectral representation: h_i = 0.5 + a*cos(2*pi*freq*i/N + phase)."""
    i = np.arange(N)
    h = 0.5 + 0.4 * np.cos(2 * np.pi * freq * i / N + phase)
    h = np.clip(h, 0, 1)
    h = _project(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Phase 1: Try multiple spectral initializations with Nesterov
    tau = 0.05
    lr = 0.02
    max_iter = 300
    
    # Try various frequencies and phases
    for freq in [1, 2, 3, 5, 7]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            if time.time() > deadline:
                break
            h_init = _spectral_init(freq, phase)
            h_sol, max_ov = _nesterov_solve(h_init, max_iter, tau, lr, deadline)
            if max_ov < best_score:
                best_score = max_ov
                best_h = h_sol.copy()
        
        # Also try with random perturbation
        if time.time() > deadline:
            break
        h_init = _spectral_init(freq, np.random.uniform(0, 2*np.pi))
        h_init += 0.05 * np.random.randn(N)
        h_init = _project(h_init)
        h_sol, max_ov = _nesterov_solve(h_init, max_iter, tau, lr, deadline)
        if max_ov < best_score:
            best_score = max_ov
            best_h = h_sol.copy()
    
    # Phase 2: Coordinate descent polish on the best solution
    if time.time() < deadline - 30:
        h_polished, score_polished = _coordinate_descent_polish(best_h, max_iters=5000, deadline=deadline - 30)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Phase 3: Another round of Nesterov with smaller tau from the polished solution
    if time.time() < deadline - 20:
        tau_final = 0.02
        lr_final = 0.01
        max_iter_final = 400
        h_sol, max_ov = _nesterov_solve(best_h, max_iter_final, tau_final, lr_final, deadline - 20)
        if max_ov < best_score:
            best_score = max_ov
            best_h = h_sol
    
    # Phase 4: Final coordinate descent polish
    if time.time() < deadline - 10:
        h_polished, score_polished = _coordinate_descent_polish(best_h, max_iters=3000, deadline=deadline - 10)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Final projection to ensure exact sum
    best_h = _project(best_h)
    
    return best_h
