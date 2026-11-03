import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110

def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _overlap_direct(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _score_and_grad(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    max_val = float(np.max(cs))
    # Gradient is zero for the max-based score, but we can use subgradient
    # For smooth optimization, we'll rely on the smooth version
    return max_val

def _smooth_max_score(h, tau=5.0):
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-9:
        return 0.0
    return m + tau * np.log(np.sum(np.exp((cs - m) / tau)))

def _smooth_max_score_grad(h, tau=5.0):
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-9:
        return 0.0, np.zeros(len(h))
    # Softmax weights
    exp_cs = np.exp((cs - m) / tau)
    weights = exp_cs / np.sum(exp_cs)
    score = m + tau * np.log(np.sum(exp_cs))
    
    # Gradient with respect to h
    # cs = (2/n) * (n/2 - autocorr(h))
    # autocorr(h) at shift k = sum_j h[j] * h[j+k] (circular)
    # d(autocorr)/dh[j] involves h[j+k] and h[j-k]
    # This is complex, so we'll use finite differences or just the score
    return score, None

def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _fix_sum_fast(h):
    """Faster sum fixing with iterative clipping"""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _polish_time_domain(h0):
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max_score(x, tau=2.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
        if res.success and res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    return h0

def _polish_gradient(h0):
    """Polish using gradient descent with projection"""
    h = _fix_sum(h0)
    lr = 0.01
    tau = 5.0
    
    for i in range(500):
        if time.time() - _t0 > _DEADLINE - 5:
            break
        # Compute gradient via finite differences
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        if m < 1e-9:
            break
        exp_cs = np.exp((cs - m) / tau)
        weights = exp_cs / np.sum(exp_cs)
        
        # Gradient of smooth max: sum_k weights[k] * d(cs[k])/dh
        # cs[k] = (2/N) * (N/2 - autocorr[k])
        # d(autocorr[k])/dh[j] = h[(j+k)%N] + h[(j-k)%N] for j != 0, and 2*h[j] for j=0
        # Actually for circular autocorrelation: autocorr[k] = sum_j h[j]*h[(j+k)%N]
        # d(autocorr[k])/dh[j] = h[(j+k)%N] + h[(j-k)%N]
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] < 1e-10:
                continue
            wk = (2.0 / N) * (-weights[k])  # negative because cs = c - autocorr
            # d(autocorr[k])/dh[j] for each j
            for j in range(N):
                grad[j] += wk * (h[(j+k) % N] + h[(j-k) % N])
        
        # Project gradient onto the constraint space
        # We want to minimize, so go against gradient
        h_new = h - lr * grad
        h_new = _fix_sum_fast(h_new)
        
        # Check if improved
        new_score = _score(h_new)
        old_score = _score(h)
        if new_score < old_score:
            h = h_new
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
    
    return h

def construct():
    global _t0
    _t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        if np.any(h > 1.0 + 1e-9) or np.any(h < -1e-9):
            return
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Fallback: Constant
    _update_best(np.full(N, 0.5))
    # Fallback: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)

    # Sparse Frequency Optimization with better initialization and more supports
    def _h_from_x(x_params, support):
        x = np.zeros(N, dtype=np.complex128)
        if not support:
            return np.zeros(N)
        
        x[support[0]] = x_params[0]
        
        for i, f in enumerate(support[1:]):
            idx_re = 1 + 2*i
            idx_im = 2 + 2*i
            if idx_re >= len(x_params): break
            x[f] = x_params[idx_re] + 1j * x_params[idx_im]
            x[N-f] = np.conj(x[f])
            
        y = np.fft.ifft(x)
        h_raw = np.abs(y)**2
        sum_h = np.sum(h_raw)
        if sum_h == 0:
            return np.zeros(N)
        h = h_raw / sum_h * (N / 2.0)
        return h

    def _objective(params, support):
        h = _h_from_x(params, support)
        penalty = 0.0
        if np.any(h > 1.0):
            penalty += np.sum((h[h > 1.0] - 1.0)**2) * 100.0
        if np.any(h < 0.0):
            penalty += np.sum((-h[h < 0.0])**2) * 100.0
        
        s = _smooth_max_score(h, tau=5.0)
        return s + penalty

    # Try different supports with multiple random restarts
    supports = []
    # Low frequencies
    for K in [10, 20, 30, 40]:
        supports.append(list(range(K)))
    # High frequencies
    for K in [10, 20, 30]:
        supports.append([0] + list(range(N-K+1, N)))
    # Bandpass
    supports.append([0] + list(range(5, 15)))
    supports.append([0] + list(range(35, 45)))
    # Random
    np.random.seed(123)
    for _ in range(15):
        K = np.random.randint(10, 25)
        idx = np.random.choice(np.arange(1, N//2), K, replace=False)
        supports.append([0] + list(idx))

    for support in supports:
        if time.time() - _t0 > _DEADLINE - 30:
            break
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Try multiple random initializations
        for restart in range(3):
            if time.time() - _t0 > _DEADLINE - 30:
                break
            # Initial guess: random phases, varied amplitudes
            x0 = np.zeros(num_params)
            x0[0] = 1.0 # DC component
            for i in range(K-1):
                amp = np.random.uniform(0.05, 0.3)
                phase = np.random.rand() * 2 * np.pi
                x0[1+2*i] = amp * np.cos(phase)
                x0[2+2*i] = amp * np.sin(phase)
                
            res = minimize(_objective, x0, args=(support,), method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-10})
            
            if res.fun < best_score:
                h_opt = _h_from_x(res.x, support)
                _update_best(h_opt)
                
    # Polish the best result in time domain
    if best_h is not None:
        polished_h = _polish_time_domain(best_h)
        if _score(polished_h) < best_score:
            best_h = polished_h
            best_score = _score(polished_h)
        
        # Also try gradient polish
        if time.time() - _t0 < _DEADLINE - 10:
            grad_h = _polish_gradient(best_h)
            if _score(grad_h) < best_score:
                best_h = grad_h
                best_score = _score(grad_h)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_h = _polish_time_domain(best_h)
        
    return best_h
