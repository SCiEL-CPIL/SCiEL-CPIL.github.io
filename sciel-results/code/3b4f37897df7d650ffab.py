"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov accelerated gradient descent on a smooth-max surrogate 
with adaptive temperature and projection onto the constraint set, 
followed by a coordinate descent polish.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145

def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _project_sum(h, target_sum):
    """
    Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on a global shift.
    """
    h = np.asarray(h, dtype=np.float64)
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12:
        return np.clip(h, 0.0, 1.0)
    
    low = -1.0
    high = 1.0
    
    if np.sum(np.clip(h + low, 0, 1)) > target_sum:
        low = -np.max(h) - 1.0
    if np.sum(np.clip(h + high, 0, 1)) < target_sum:
        high = 1.0 - np.min(h) + 1.0
        
    for _ in range(50):
        mid = (low + high) / 2.0
        s_mid = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
            
    delta = (low + high) / 2.0
    h_proj = np.clip(h + delta, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] = min(1.0, h_proj[idx] + diff)
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] = max(0.0, h_proj[idx] + diff)
            
    return h_proj

def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C(h).
    phi_beta(C) = (1/beta) * log(sum(exp(beta * C_k)))
    """
    cs = _linear_overlap_values(h)
    # Numerical stability for log-sum-exp
    max_c = np.max(cs)
    exp_cs = np.exp(beta * (cs - max_c))
    sum_exp = np.sum(exp_cs)
    val = max_c + np.log(sum_exp) / beta
    
    # Weights w_k
    w = exp_cs / sum_exp
    
    # Gradient of C_k w.r.t h
    # dC_k/dh_j = -2/N * (h_{j-k} * 1_{j>=k} + h_{j+k} * 1_{j+k<N})
    # We compute grad_h = sum_k w_k * dC_k/dh
    grad_h = np.zeros(N)
    
    # Vectorized computation of contributions
    # For each k, we add w_k * (h_{j-k} for j>=k) + w_k * (h_{j+k} for j+k<N)
    # This is equivalent to:
    # grad_h[j] = sum_{k<=j} w_k h_{j-k} + sum_{k<N-j} w_k h_{j+k}
    
    # Let's compute using loops for clarity, N=100 is small
    for k in range(N):
        if w[k] < 1e-12:
            continue
        wk = w[k]
        # Term 1: h_{j-k} for j >= k
        if k < N:
            # j ranges from k to N-1
            # h[j-k] for j in [k, N-1] is h[0 ... N-1-k]
            grad_h[k:] += wk * h[:N-k]
        # Term 2: h_{j+k} for j+k < N => j < N-k
        if k < N:
            # j ranges from 0 to N-1-k
            # h[j+k] for j in [0, N-1-k] is h[k ... N-1]
            grad_h[:N-k] += wk * h[k:]
            
    grad_h *= -2.0 / N
    return val, grad_h

def _nesterov_accelerated_gradient(h, beta, steps, lr):
    """
    Run Nesterov Accelerated Gradient Descent for a fixed beta.
    """
    h = np.array(h, dtype=np.float64)
    y = h.copy()
    v = np.zeros(N)
    t = 1.0
    
    for i in range(steps):
        if time.time() > _DEADLINE - 10:
            break
            
        # Nesterov step
        # y = x - (t-1)/(t+2) * v
        # Actually, standard NAG:
        # y_k = x_k + (t_{k-1}-1)/(t_k+2) * (x_k - x_{k-1})
        # Let's use a simpler momentum version or standard NAG update
        # x_{k+1} = y_k - lr * grad(y_k)
        # t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        # y_{k+1} = x_{k+1} + (t_k - 1) / (t_{k+1} + 2) * (x_{k+1} - x_k)
        
        val, grad_y = _smooth_max_obj_grad(y, beta)
        
        x_new = y - lr * grad_y
        x_new = _project_sum(x_new, N / 2.0)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y_new = x_new + ((t - 1.0) / (t_new + 2.0)) * (x_new - h)
        y_new = _project_sum(y_new, N / 2.0)
        
        h = x_new
        y = y_new
        t = t_new
        
    return h

def _coordinate_descent_polish(h, max_iters=1000):
    """
    Local search by flipping individual bits or adjusting values.
    Since the problem is combinatorial in nature (binary is often optimal),
    we snap to binary and then try swaps.
    """
    h = np.array(h, dtype=np.float64)
    best_h = h.copy()
    best_score = _score(h)
    
    # Snap to binary
    h_bin = np.where(h > 0.5, 1.0, 0.0)
    ones = np.sum(h_bin == 1.0)
    if ones > N / 2:
        idx_ones = np.where(h_bin == 1.0)[0]
        idx_sorted = idx_ones[np.argsort(h[idx_ones])]
        to_remove = idx_sorted[:ones - N // 2]
        h_bin[to_remove] = 0.0
    elif ones < N / 2:
        idx_zeros = np.where(h_bin == 0.0)[0]
        idx_sorted = idx_zeros[np.argsort(-h[idx_zeros])]
        to_add = idx_sorted[:N // 2 - ones]
        h_bin[to_add] = 1.0
        
    s_bin = _score(h_bin)
    if s_bin < best_score:
        best_score = s_bin
        best_h = h_bin.copy()
        
    h_cur = best_h.copy()
    s_cur = best_score
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        idx_ones = np.where(h_cur == 1.0)[0]
        idx_zeros = np.where(h_cur == 0.0)[0]
        
        # Try random swaps
        num_swaps_to_try = 50
        for _ in range(num_swaps_to_try):
            if len(idx_ones) == 0 or len(idx_zeros) == 0:
                break
            i = idx_ones[np.random.randint(len(idx_ones))]
            j = idx_zeros[np.random.randint(len(idx_zeros))]
            
            h_test = h_cur.copy()
            h_test[i] = 0.0
            h_test[j] = 1.0
            
            s_test = _score(h_test)
            if s_test < s_cur - 1e-12:
                h_cur = h_test
                s_cur = s_test
                best_h = h_cur.copy()
                best_score = s_cur
                improved = True
                idx_ones = np.where(h_cur == 1.0)[0]
                idx_zeros = np.where(h_cur == 0.0)[0]
                break
                
        if not improved:
            # Try single flips if possible? No, sum must be N/2.
            # Try 2-swap (two 1s to 0s, two 0s to 1s) ?
            # For now, break.
            break
            
    return best_h, best_score

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Initial h: random binary
    h_init = np.random.choice([0.0, 1.0], size=N)
    h_init = _project_sum(h_init, N / 2.0)
    
    # Nesterov phases with increasing beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    h_cur = h_init
    
    for beta in betas:
        if time.time() > _DEADLINE - 20:
            break
            
        # Adaptive learning rate
        # Start with a small lr
        lr = 0.01
        steps = 500
        
        # Run NAG
        h_cur = _nesterov_accelerated_gradient(h_cur, beta, steps, lr)
        
        # Polish
        h_pol, s_pol = _coordinate_descent_polish(h_cur, max_iters=200)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()
            
        # If score is very good, we can stop or increase beta more aggressively
        if best_score < 0.5:
            break
            
    # Final polish with more iterations
    if best_h is not None:
        h_final, s_final = _coordinate_descent_polish(best_h, max_iters=2000)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum(best_h, N / 2.0)
    return best_h
