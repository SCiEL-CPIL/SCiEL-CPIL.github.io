import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)
    return C[:2*N-1]

def _evaluate(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _soft_max_and_grad(h, k=50.0):
    """
    Compute soft max of autoconvolution and its gradient w.r.t h.
    Returns (soft_max_value, grad_soft_max).
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    c = _autoconv_fft(h)
    m = np.max(c)
    
    # Avoid overflow
    exp_k_c = np.exp(k * (c - m))
    sum_exp = np.sum(exp_k_c)
    M = m + np.log(sum_exp) / k
    
    # Weights for soft max gradient
    w = exp_k_c / sum_exp
    
    # Gradient of M w.r.t h
    # grad_M = 2 * cross_corr(w, h)
    # cross_corr(w, h)[l] = sum_j w_j h[l-j]
    # Using FFT: ifft(W * conj(H))
    size = 2 * N
    W = np.fft.rfft(w, n=size)
    H = np.fft.rfft(h, n=size)
    cross_corr = np.fft.irfft(W * np.conj(H), n=size)
    grad_M = 2.0 * cross_corr[:N]
    
    return M, grad_M

def _objective_and_grad(h, k=50.0):
    """
    Compute objective J(h) = soft_max(c(h)) / (sum(h))^2
    and its gradient.
    """
    s = np.sum(h)
    if s == 0:
        return np.inf, np.zeros_like(h)
    
    M, grad_M = _soft_max_and_grad(h, k)
    
    # Objective
    J = M / (s * s)
    
    # Gradient of J
    # J = M * s^-2
    # dJ = dM * s^-2 + M * (-2) s^-3 * ds
    # ds = sum(dh) = 1 (vector of ones)
    grad_J = grad_M / (s * s) - 2.0 * M * np.ones(N) / (s * s * s)
    
    return J, grad_J

def _sa_step(h, conv, current_val, T, rng):
    """
    Perform one simulated annealing step.
    h: current heights
    conv: current autoconvolution
    current_val: current objective value (using max, not soft max)
    T: temperature
    rng: random number generator
    Returns: (new_h, new_conv, new_val, accepted)
    """
    # Pick a random index to flip or perturb
    idx = rng.randint(N)
    
    # Try flipping the sign or adding a random perturbation
    # Strategy: with some probability flip sign, otherwise add random noise
    if rng.random() < 0.3:
        # Flip sign
        new_h = h.copy()
        new_h[idx] = -new_h[idx]
    else:
        # Add random perturbation
        new_h = h.copy()
        new_h[idx] += rng.randn() * 0.1 * (np.max(np.abs(h)) + 1e-10)
    
    # Ensure finite
    if not np.all(np.isfinite(new_h)):
        return h, conv, current_val, False
    
    # Compute new autoconvolution
    new_conv = _autoconv_fft(new_h)
    
    # Compute new objective value
    s_new = np.sum(new_h)
    if s_new == 0:
        return h, conv, current_val, False
    
    new_val = abs(2.0 * N * np.max(new_conv)) / (s_new * s_new)
    
    # Accept if better or with probability exp(-delta/T)
    if new_val < current_val:
        return new_h, new_conv, new_val, True
    else:
        delta = new_val - current_val
        if delta > 0 and T > 0:
            if rng.random() < np.exp(-delta / T):
                return new_h, new_conv, new_val, True
        return h, conv, current_val, False

def _simulated_annealing(h_init, max_iters=5000, T0=1.0, T_min=0.001, cooling=0.999):
    """
    Run simulated annealing to minimize the maximum autoconvolution coefficient.
    """
    h = h_init.copy()
    conv = _autoconv_fft(h)
    s = np.sum(h)
    if s == 0:
        # Add small DC offset
        h += 1e-6
        s = np.sum(h)
        conv = _autoconv_fft(h)
    
    current_val = abs(2.0 * N * np.max(conv)) / (s * s)
    
    best_h = h.copy()
    best_val = current_val
    
    rng = np.random.RandomState(42)
    
    T = T0
    for it in range(max_iters):
        h, conv, current_val, accepted = _sa_step(h, conv, current_val, T, rng)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _refine_continuous(h, deadline, time_budget=10.0):
    """
    Refine h using continuous optimization (L-BFGS-B).
    """
    start_time = time.time()
    
    def func(x):
        val, grad = _objective_and_grad(x, k=50.0)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(x)
        return val, grad
    
    try:
        res = scipy.optimize.minimize(func, h, method='L-BFGS-B', 
                                      jac=True, options={'maxiter': 100, 'gtol': 1e-8})
        h_opt = res.x
        if np.all(np.isfinite(h_opt)):
            return h_opt
    except Exception:
        pass
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Random Gaussian
    for _ in range(5):
        h0 = rng.randn(N)
        h0 += 1.0  # Ensure non-zero sum
        initializations.append(h0)
    
    # 3. Binary sequences (Golay-like)
    for _ in range(10):
        h0 = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        # Ensure sum is not zero
        if abs(np.sum(h0)) < 1.0:
            h0 += 0.1
        initializations.append(h0)
    
    # 4. Comb patterns
    for period in [2, 3, 4, 5, 8, 10, 16]:
        h0 = np.where(np.arange(N) % period < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.randn(N)
        if abs(np.sum(h0)) > 0.5:
            initializations.append(h0)
    
    # 5. Sinusoids
    i = np.arange(N)
    for k in [1, 2, 3, 4, 5]:
        for a in [0.5, 1.0, 1.5]:
            h0 = a + np.sin(2 * np.pi * k * i / N)
            initializations.append(h0)
    
    # Process each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Normalize to have sum = 1
        s = np.sum(h0)
        if abs(s) < 1e-10:
            continue
        h0 = h0 / s
        
        # Phase 1: Simulated Annealing (discrete refinement)
        time_left = deadline - time.time()
        if time_left > 20.0:
            sa_iters = min(3000, int(5.0 * 1000 / max(1.0, (deadline - time.time()) / 10.0)))
            h_sa, val_sa = _simulated_annealing(h0, max_iters=sa_iters)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
            h_current = h_sa
        else:
            h_current = h0
        
        # Phase 2: Continuous refinement
        time_left = deadline - time.time()
        if time_left > 10.0:
            h_refined = _refine_continuous(h_current, deadline, time_budget=min(5.0, time_left / 2))
            val_refined = _evaluate(h_refined)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish with a few more SA iterations if time permits
    time_left = deadline - time.time()
    if time_left > 5.0 and best_val < np.inf:
        h_final, val_final = _simulated_annealing(best_h, max_iters=min(2000, int(time_left * 200)))
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    return best_h
