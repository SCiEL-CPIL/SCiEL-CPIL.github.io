"""Erdos minimum overlap: bisection on t with Nesterov accelerated projected gradient."""
import numpy as np
import time

N = 100
HALF_N = N / 2.0

def _overlap_matrix(h):
    """Compute overlap(k) for all circular shifts k=0..N-1.
    overlap(k) = (1/N) * sum_i h_i * (1 - h_{(i-k) % N})
    """
    # autocorrelation: A(k) = (1/N) * sum_i h_i * h_{(i-k)%N}
    # overlap(k) = 1 - A(k)
    h = np.asarray(h, dtype=np.float64)
    # Use FFT for circular autocorrelation
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H))) / N
    # A[k] = (1/N) sum_i h_i * h_{(i-k)%N}  (circular)
    # overlap[k] = 1 - A[k]
    return 1.0 - A

def _project(h):
    """Project h onto {x in [0,1]^N : sum(x) = N/2}."""
    # Use bisection on the threshold for box constraint with sum constraint
    # We need sum(clamp(h - lambda, 0, 1)) = N/2
    lo, hi = -1.0, 1.0
    # Expand range if needed
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

def _smooth_obj(h, t, tau):
    """Log-sum-exp of (overlap(k) - t) as smooth surrogate for max_k overlap(k) - t.
    Returns (value, gradient) of the smooth objective.
    """
    ov = _overlap_matrix(h)
    z = (ov - t) / tau
    # log-sum-exp
    zmax = np.max(z)
    lse = zmax + np.log(np.sum(np.exp(z - zmax)))
    # value = tau * lse
    # gradient w.r.t. h: (1/N) * d/dh [sum_k exp(ov_k - t)] / sum... 
    # Actually d/dh_k of log-sum-exp((ov - t)/tau) * tau
    # ov_k = 1 - A_k, A_k = (1/N) sum_i h_i h_{(i-k)%N}
    # d ov_k / d h_j = -(1/N) * (h_{(j+k)%N} + h_{(j-k)%N}) for k != 0
    # For k=0: ov_0 = 1 - (1/N)sum h_i^2, d/dh_j = -2 h_j / N
    # Let's compute gradient numerically-free via structure
    # 
    # Let w_k = exp(z_k) / sum(exp(z))  (softmax weights)
    # grad_obj[j] = tau * sum_k w_k * d ov_k / d h_j
    # d ov_k / d h_j = -d A_k / d h_j
    # A_k = (1/N) sum_i h_i h_{(i-k)%N}
    # d A_k / d h_j = (1/N) * (h_{(j+k)%N} + h_{(j-k)%N})  [for the two terms where index matches j]
    #   except when 2k ≡ 0 mod N, then it's (1/N)*2*h_{j}... let's be careful
    # Actually: A_k = (1/N) sum_i h_i * h_{(i-k)%N}
    # dA_k/dh_j: term i=j gives h_{(j-k)%N}/N; term i-k≡j mod N i.e. i≡j+k gives h_j * 1/N... 
    # Wait: A_k = (1/N) sum_i h_i h_{(i-k) mod N}
    # d/dh_j: for each i, if i=j, contributes h_{(j-k)%N}/N. If (i-k)%N = j i.e. i=(j+k)%N, contributes h_i/N = h_{(j+k)%N}/N.
    # If j = (j+k)%N i.e. k≡0, these are the same term, so it's 2*h_j/N... no wait.
    # When k=0: A_0 = (1/N) sum_i h_i^2, dA_0/dh_j = 2 h_j / N.
    # When k≠0 and N is even and k=N/2: A_k = (1/N) sum_i h_i h_{(i+k)%N}, and (j+k)%N = j only if 2k≡0, i.e., k=0 or k=N/2 (if N even).
    # For k=N/2 with N even: (j+k)%N = j, so both terms give h_j, total 2 h_j / N.
    # Otherwise: (h_{(j-k)%N} + h_{(j+k)%N}) / N.
    
    # Let's just compute the gradient via the formula
    w = np.exp(z - zmax)
    w = w / np.sum(w)  # softmax weights, sum to 1
    
    # grad_ov_k / d h_j = -dA_k/dh_j
    # Build gradient array
    grad = np.zeros(N)
    for k in range(N):
        wk = w[k]
        if wk < 1e-15:
            continue
        if k == 0:
            # dA_0/dh_j = 2 h_j / N
            grad += -wk * (2.0 * h / N)
        elif N % 2 == 0 and k == N // 2:
            # dA_k/dh_j = 2 h_j / N
            grad += -wk * (2.0 * h / N)
        else:
            # dA_k/dh_j = (h_{(j-k)%N} + h_{(j+k)%N}) / N
            idx_minus = (np.arange(N) - k) % N
            idx_plus = (np.arange(N) + k) % N
            contrib = (h[idx_minus] + h[idx_plus]) / N
            grad += -wk * contrib
    
    return tau * lse, tau * grad

def _nesterov_solve(t, max_iter, tau, lr, deadline):
    """Nesterov accelerated gradient to minimize smooth_obj(h, t) subject to constraints.
    Returns (h, obj_value, max_overlap).
    """
    # Initialize with uniform 0.5
    h = np.full(N, 0.5)
    h = _project(h)
    
    # Nesterov: y = x + momentum * (x - x_prev)
    x_prev = h.copy()
    momentum = 0.0
    
    best_h = h.copy()
    best_max_ov = np.max(_overlap_matrix(h))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Nesterov step
        momentum_new = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = h + (momentum / momentum_new) * (h - x_prev)
        
        # Compute gradient at y
        _, grad_y = _smooth_obj(y, t, tau)
        
        # Gradient step
        h_new = y - lr * grad_y
        
        # Project
        h_new = _project(h_new)
        
        # Update
        x_prev = h
        h = h_new
        momentum = momentum_new
        
        # Track best
        ov = _overlap_matrix(h)
        max_ov = np.max(ov)
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
    
    return best_h, best_max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Bisection on t: find smallest t such that there exists h with max overlap <= t
    # Lower bound: 0 (theoretical min), Upper bound: 0.5 (uniform h gives overlap 0.5 at all shifts)
    t_lo = 0.0
    t_hi = 0.5
    
    best_h = np.full(N, 0.5)
    best_t = 0.5
    
    # First, try to find a good solution with Nesterov for various t
    # Bisection: for ~30 iterations
    tau = 0.05  # Temperature for log-sum-exp
    lr = 0.01   # Learning rate
    max_iter = 200
    
    for bisect_iter in range(35):
        if time.time() > deadline:
            break
        
        t_mid = (t_lo + t_hi) / 2.0
        
        h_sol, max_ov = _nesterov_solve(t_mid, max_iter, tau, lr, deadline)
        
        if max_ov <= t_mid + 1e-6:
            # Feasible: t_mid is achievable
            t_hi = t_mid
            best_h = h_sol.copy()
            best_t = t_mid
        else:
            # Not feasible: need larger t
            t_lo = t_mid
        
        # Adapt learning rate and tau as we go
        tau = max(0.01, tau * 0.95)
    
    # Final polish: run Nesterov with the best t found, with smaller tau
    if time.time() < deadline - 10.0:
        tau_final = 0.01
        max_iter_final = 500
        h_polished, max_ov_polished = _nesterov_solve(best_t, max_iter_final, tau_final, lr * 0.5, deadline)
        if max_ov_polished < np.max(_overlap_matrix(best_h)):
            best_h = h_polished
            best_t = max_ov_polished
    
    # Final projection to ensure exact sum
    best_h = _project(best_h)
    
    return best_h
