import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_overlaps(h):
    """Compute overlap at each shift: sum_i h[i] * (1 - h[(i+k)%N])"""
    sum_h = np.sum(h)
    H = np.fft.fft(h)
    # Circular autocorrelation: R[k] = sum_i h[i] * h[(i-k)%N]
    # For real h, R[k] = R[-k], so R_pos[k] = R_neg[k]
    R = np.fft.ifft(np.abs(H)**2).real
    overlaps = sum_h - R
    return overlaps

def _max_overlap(h):
    return np.max(_compute_overlaps(h))

def _smooth_max(h, beta):
    """Smooth approximation of max overlap using log-sum-exp."""
    overlaps = _compute_overlaps(h)
    m = np.max(overlaps)
    if m == 0:
        return 0.0, np.zeros(N)
    shifted = overlaps - m
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    exps = np.exp(beta * shifted)
    lse = m + np.log(np.sum(exps)) / beta
    # Gradient
    weights = exps / np.sum(exps)
    # d(overlaps[k])/d(h[j]) = 1 - h[(j-k)%N] - h[(j+k)%N]
    # grad[j] = sum_k weights[k] * (1 - h[(j-k)%N] - h[(j+k)%N])
    H = np.fft.fft(h)
    W = np.fft.fft(weights)
    # conv1[j] = sum_k w[k]*h[(j-k)%N]
    conv1 = np.fft.ifft(W * H).real
    # conv2[j] = sum_k w[k]*h[(j+k)%N]
    w_neg = np.zeros(N)
    w_neg[0] = weights[0]
    w_neg[1:] = weights[:0:-1]
    W_neg = np.fft.fft(w_neg)
    conv2 = np.fft.ifft(W_neg * H).real
    grad = np.sum(weights) - conv1 - conv2
    return lse, grad

def _project_sum(h, target=N/2):
    """Project h to [0,1]^N with sum = target."""
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    if diff > 0:
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room > 1e-12:
            h += diff * room / total_room
    else:
        room = h
        total_room = np.sum(room)
        if total_room > 1e-12:
            h += diff * room / total_room
    h = np.clip(h, 0.0, 1.0)
    # Final fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            for i in range(N):
                if diff <= 1e-12:
                    break
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
        else:
            for i in range(N):
                if -diff <= 1e-12:
                    break
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
    return h

def _zadoff_chu_init(root=1, N=N):
    """Generate a Zadoff-Chu-like real sequence."""
    n = np.arange(N)
    # Zadoff-Chu: x[n] = exp(-j*pi*root*n*(n+1)/N) for N even
    # Take real part and map to [0,1]
    if N % 2 == 0:
        phase = -np.pi * root * n * (n + 1) / N
    else:
        phase = -np.pi * root * n * (n + 1) / N
    x = np.cos(phase)
    # Normalize to [0,1] with mean 0.5
    x = (x - np.mean(x)) / (np.max(x) - np.min(x) + 1e-12)
    x = 0.5 + 0.5 * (x - 0.5) * 2  # scale to roughly [0,1]
    x = 0.5 + 0.5 * (2*x - 1)
    x = np.clip(x, 0, 1)
    return _project_sum(x)

def _sinusoid_init(freq, amp=0.5, phase=0.0):
    """Generate a sinusoidal perturbation around 0.5."""
    n = np.arange(N)
    h = 0.5 + amp * np.cos(2 * np.pi * freq * n / N + phase)
    return _project_sum(h)

def _two_block_init(block_frac=0.5, levels=(0.0, 1.0)):
    """Two-block pattern."""
    h = np.zeros(N)
    n_low = int(N * block_frac)
    h[:n_low] = levels[0]
    h[n_low:] = levels[1]
    return _project_sum(h)

def construct():
    deadline = time.time() + 115  # Safety margin
    
    best_h = None
    best_val = np.inf
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.full(N, 0.5))
    
    # Zadoff-Chu variants
    for root in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
        inits.append(_zadoff_chu_init(root))
    
    # Sinusoids at various frequencies
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        for phase in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
            for amp in [0.3, 0.5, 0.7]:
                inits.append(_sinusoid_init(freq, amp, phase))
    
    # Two-block variants
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        inits.append(_two_block_init(frac, (0.0, 1.0)))
        inits.append(_two_block_init(frac, (0.2, 0.8)))
        inits.append(_two_block_init(frac, (0.1, 0.9)))
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        h = rng.random(N)
        inits.append(_project_sum(h))
    
    # Evaluate all initializations
    for h in inits:
        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if time.time() > deadline - 30:
        return best_h
    
    # Polish the best few with L-BFGS-B
    # Sort initializations by score, take top 10
    scored = []
    seen = set()
    for h in inits:
        key = tuple(np.round(h, 4))
        if key in seen:
            continue
        seen.add(key)
        val = _max_overlap(h)
        scored.append((val, h))
    scored.sort(key=lambda x: x[0])
    top_inits = [h for _, h in scored[:10]]
    
    for h0 in top_inits:
        if time.time() > deadline - 15:
            break
        
        # Adaptive beta: start high, reduce if stuck
        for beta in [20.0, 10.0, 5.0, 2.0]:
            if time.time() > deadline - 10:
                break
            
            def obj(x):
                val, _ = _smooth_max(x, beta)
                return val
            
            def grad(x):
                _, g = _smooth_max(x, beta)
                return g
            
            try:
                res = minimize(
                    obj, h0, jac=grad,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
            except Exception:
                continue
            
            h_new = _project_sum(res.x)
            val = _max_overlap(h_new)
            if val < best_val:
                best_val = val
                best_h = h_new.copy()
            
            # Use improved solution for next beta
            h0 = h_new
    
    # Final polish with SLSQP on the best solution
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        for beta in [30.0, 15.0, 8.0]:
            if time.time() > deadline - 3:
                break
            
            def obj(x):
                val, _ = _smooth_max(x, beta)
                return val
            
            def grad(x):
                _, g = _smooth_max(x, beta)
                return g
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2, 'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            try:
                res = minimize(
                    obj, h0, jac=grad,
                    method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
                )
            except Exception:
                continue
            
            h_new = _project_sum(res.x)
            val = _max_overlap(h_new)
            if val < best_val:
                best_val = val
                best_h = h_new.copy()
            h0 = h_new
    
    return best_h

# EVOLVE-BLOCK-END
