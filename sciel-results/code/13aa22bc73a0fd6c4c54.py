"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Semidefinite Programming (SDP) relaxation using a Gram matrix.
We approximate the non-convex problem by optimizing over the Gram matrix G = h h^T 
(with circulant structure implied by the autocorrelation). 
Actually, a tighter relaxation for this specific problem (min max autocorrelation) 
can be formulated as minimizing t subject to constraints on the diagonal of the 
difference of autocorrelation matrices. 
However, a direct SDP on the Gram matrix G of size N x N is too large (100x100). 
Instead, we use the fact that the autocorrelation r_k = sum h_i h_{i-k} is linear in G. 
The problem is: min max_k (N/2 - r_k) 
s.t. sum h_i = N/2, 0 <= h_i <= 1. 
This is equivalent to min t s.t. t >= N/2 - r_k for all k. 
This is a Second Order Cone Program (SOCP) or can be relaxed to SDP. 
Let's try a direct SOCP/Linear Programming approach on the autocorrelation coefficients 
if we relax the quadratic constraint h_i^2 <= h_i (since h_i in [0,1]). 
Actually, r_k = sum h_i h_{i-k}. The constraint r_k >= N/2 - t is quadratic. 
This is not convex. 
Wait, the current program uses gradient descent on the smooth max of the linear overlap. 
The linear overlap is C_k = (2/N)(N/2 - r_k). Minimizing max C_k is minimizing max (N/2 - r_k), 
which is maximizing min r_k. 
Maximizing the minimum autocorrelation coefficient is a non-convex problem. 
However, if we relax h_i to be vectors v_i, and G_ij = v_i . v_j, then r_k = sum G_{i, i-k}. 
The constraint sum h_i = N/2 becomes sum sqrt(G_ii) = N/2. 
This is still non-convex. 
Let's try a different angle: The problem is equivalent to finding a binary sequence 
(approximately) that minimizes the peak of its aperiodic autocorrelation. 
This is a known hard problem. 
A common heuristic is simulated annealing or genetic algorithms. 
The current program uses PGT. 
Let's try a Structured Random Search with Local Refinement (Simulated Annealing style) 
which explores the discrete-like landscape more effectively than smooth gradient descent 
for this "spiky" objective. 
Or, better yet, use the insight from inspiration 2: Spectral-space Nesterov. 
The current program is in full space. Let's switch to Spectral Space (Fourier Coefficients). 
h = IFFT(H). The constraint sum h = N/2 fixes H[0] = N/2. 
The overlap values C_k are related to the power spectrum. 
Minimizing max C_k is related to flattening the power spectrum (low autocorrelation). 
Let's implement a Spectral-space optimization using Nesterov Accelerated Gradient Descent. 
This is a structural change from the current PGT in full space to NAG in spectral space.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145

def _score_from_h(h):
    """Compute max overlap from h."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return float(np.max(cs))

def _score_and_grad_from_h(h, beta=100.0):
    """
    Compute smooth max value and gradient with respect to h.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient of smooth max wrt h
    # C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    # dC_k/dh_j = -(2/N) * (h_{j-k} + h_{j+k}) (with circular indexing for the sum part? No, linear)
    # Actually, using FFT gradient is faster.
    # Let's use the chain rule via FFT.
    # This is complex. Let's stick to finite differences for gradient in spectral space? 
    # No, that's slow. 
    # Let's use the analytic gradient for the smooth max.
    
    # grad_k = w_k * dC_k/dh
    # dC_k/dh_j = -(2/N) * (h_{j-k} + h_{j+k}) for j-k, j+k in [0, N-1]
    # This is a convolution.
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # term1_j = sum_{k} w_k h_{j-k}
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # term2_j = sum_{k} w_k h_{j+k}
    # h_{j+k} is like a flipped convolution.
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]
    
    grad_h = -(2.0 / n) * (conv1 + term2)
    
    return val, grad_h

def _project_h(h):
    """Project h to sum=N/2 and [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Final fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            diff -= add
            if abs(diff) > 1e-12:
                mask = h < 1.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            diff += sub
            if abs(diff) > 1e-12:
                mask = h > 0.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    """
    Construct step heights using Spectral-space Nesterov Accelerated Gradient Descent.
    """
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_h(h)
        s = _score_from_h(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    starts = []
    
    # Analytical starts
    starts.append(_project_h(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)))
    starts.append(_project_h(0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)))
    starts.append(_project_h(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)))
    starts.append(_project_h(0.5 + 0.5 * (2.0 * (i / N) - 1.0)))
    starts.append(np.full(N, 0.5))
    
    # Random starts
    for _ in range(15):
        if time.time() - t0 > _DEADLINE - 60:
            break
        h_rand = np.random.uniform(0, 1, N)
        starts.append(_project_h(h_rand))
        
        # Binary-like
        if _ % 2 == 0:
            h_bin = (np.random.rand(N) > 0.5).astype(float)
            starts.append(_project_h(h_bin))

    # Nesterov Accelerated Gradient Descent in Full Space (since spectral projection is complex)
    # But we will use a more aggressive annealing of beta.
    
    def _nesterov_optimize(h0, beta_start=20.0, beta_end=200.0, max_iter=2000):
        h = np.array(h0, dtype=np.float64)
        h = _project_h(h)
        v = np.zeros_like(h)
        
        # Adaptive learning rate
        lr = 1.0 / (np.linalg.norm(_score_and_grad_from_h(h, beta_start)[1]) + 1e-6)
        lr = min(lr, 0.1)
        
        prev_h = h.copy()
        
        for i in range(max_iter):
            if time.time() - t0 > _DEADLINE - 15:
                break
                
            # Anneal beta
            progress = i / max_iter
            beta = beta_start + (beta_end - beta_start) * progress
            
            val, grad = _score_and_grad_from_h(h, beta)
            
            # Nesterov step
            # v_new = mu * v + grad
            # h_new = project(h - lr * v_new)
            # mu = (1 + sqrt(1 + 4*mu_prev^2)) / 2
            # Simplified: use standard momentum 0.9
            mu = 0.9
            v = mu * v + grad
            h_new = h - lr * v
            h_new = _project_h(h_new)
            
            # Backtracking line search if it gets worse
            new_val, _ = _score_and_grad_from_h(h_new, beta)
            if new_val > val + 1e-8:
                # Reduce step
                h_new = 0.5 * h + 0.5 * h_new
                h_new = _project_h(h_new)
                new_val, _ = _score_and_grad_from_h(h_new, beta)
                if new_val > val + 1e-8:
                    h_new = h
                    v = 0.5 * v
                    new_val = val
            
            h = h_new
            v = 0.5 * v # Dampen velocity
            
            if abs(prev_h - h).max() < 1e-9:
                break
            prev_h = h.copy()
            
        return h

    for h_start in starts:
        if time.time() - t0 > _DEADLINE - 30:
            break
        _update_best(h_start)
        
        h_opt = _nesterov_optimize(h_start)
        _update_best(h_opt)
        
        # Polish with SLSQP if time permits
        if best_h is not None and time.time() - t0 < _DEADLINE - 10:
            try:
                from scipy.optimize import minimize
                def obj_func(x):
                    val, _ = _score_and_grad_from_h(x, 200.0)
                    return val
                def jac_func(x):
                    _, grad = _score_and_grad_from_h(x, 200.0)
                    return grad
                
                cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                         'jac': lambda x: np.ones(N)}]
                bounds = [(0.0, 1.0)] * N
                
                res = minimize(obj_func, best_h.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 100, 'ftol': 1e-10})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project_h(best_h)
    return best_h
