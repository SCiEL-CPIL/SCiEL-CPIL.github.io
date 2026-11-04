import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _ratio_and_grad(h, beta=100.0):
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros_like(h)
    
    u = h / l1
    l2_sq = np.sum(u ** 2)
    conv = _autoconv(u)
    
    # Soft max
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp)
    S = (log_sum_exp + beta * max_val) / beta
    
    if S <= 0:
        return 0.0, np.zeros_like(h)
        
    val = l2_sq / S
    
    # Gradient of S w.r.t u
    # S_u_j = sum_k p_k * dC_k/du_j
    # dC_k/du_j = 2 * u_{k-j} (if 0 <= k-j < N)
    # p_k = exp(beta * conv_k) / sum_exp
    p = exp_shifted / sum_exp
    
    # Compute S_u
    S_u = np.zeros(N)
    # conv length is 2N-1
    # u length is N
    # For each j in 0..N-1:
    # S_u[j] = 2 * sum_{k=j}^{j+N-1} p[k] * u[k-j]
    # This is cross correlation of p and u.
    # We can use np.correlate(p, u, mode='valid')?
    # p length 2N-1, u length N.
    # 'valid' returns length 2N-1 - N + 1 = N.
    # np.correlate(p, u, mode='valid') computes sum_k p[k] u[k-j]?
    # Let's check np.correlate definition.
    # np.correlate(a, v) = sum_k a[k] v[k-j].
    # If a=p, v=u, then result[j] = sum_k p[k] u[k-j].
    # This matches our formula.
    S_u = 2.0 * np.correlate(p, u, mode='valid')
    
    # Gradient of F(u) = l2_sq / S
    # F_u_j = (2 u_j S - l2_sq S_u_j) / S^2
    F_u = (2.0 * u * S - l2_sq * S_u) / (S * S)
    
    # Gradient w.r.t h
    # G_j = 1/l1 * F_u_j - 1/l1^2 * h_j * sum_i F_u_i h_i
    dot_F_u_h = np.dot(F_u, h)
    G = (1.0 / l1) * F_u - (1.0 / (l1 * l1)) * h * dot_F_u_h
    
    return val, G

def _ratio_neg(h, beta=100.0):
    val, _ = _ratio_and_grad(h, beta)
    return -val

def _ratio_neg_grad(h, beta=100.0):
    val, grad = _ratio_and_grad(h, beta)
    return -val, -grad

def _dft_parameterize(z):
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h

def _dft_inverse(h):
    H_mags = np.sqrt(h + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z

def _optimize_from_z(z0, start_time, max_iter=100, tol=1e-8):
    z = z0.astype(complex).copy()
    
    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        return _ratio_neg(h, beta=100.0)
        
    def objective_grad(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        val, grad_h = _ratio_and_grad(h, beta=100.0)
        # Chain rule: dval/dz = dval/dh * dh/dz
        # h = |fft(z)|^2
        # This gradient is complex and tricky. 
        # It's safer to just use finite differences for z or stick to h optimization.
        # Given time, let's skip analytic grad for z and use h optimization primarily.
        # Or just return None for grad to use finite diff?
        # L-BFGS-B with finite diff on 100 vars is slow.
        # Let's just return val and let it use finite diff for z?
        # Or better, don't use z optimization if it's not efficient.
        # The current code uses both. Let's keep it but maybe it's less important.
        return -val, None # If None, scipy uses finite diff

    z_flat = np.concatenate([z.real, z.imag])
    
    result = minimize(objective, z_flat, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    z_opt = result.x[:N] + 1j * result.x[N:]
    h_opt = _dft_parameterize(z_opt)
    score, _ = _ratio_and_grad(h_opt)
    return h_opt, score

def _local_optimize(h0, start_time, max_iter=200, tol=1e-9):
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat, beta=100.0)
        
    def objective_grad(h_flat):
        return _ratio_neg_grad(h_flat, beta=100.0)
        
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      jac=objective_grad,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    h_opt = np.maximum(result.x, 0.0)
    score, _ = _ratio_and_grad(h_opt)
    return h_opt, score

def _gaussian_profile(center, width, height=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h

def _lorentzian_profile(center, gamma, height=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
    return h

def _power_profile(k, side='right'):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = np.zeros(N)
    if side == 'right':
        mask = x > 0
        h[mask] = x[mask] ** k
    elif side == 'left':
        mask = x < 0
        h[mask] = (-x[mask]) ** k
    else:
        h = np.abs(x) ** k
    return h

def _two_gaussian(c1, w1, c2, w2, h1=1.0, h2=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h1_arr = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    h2_arr = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    return h1_arr + h2_arr

def _sinc_like_profile(width, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    t = (x - center) / width
    h = np.sinc(t) ** 2
    return np.maximum(h, 0)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score, _ = _ratio_and_grad(best_h)
    
    initializations = []
    initializations.append(np.ones(N))
    
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
            initializations.append(_gaussian_profile(center, width, 1.0))
            
    for center in [-0.15, 0.0, 0.15]:
        for gamma in [0.03, 0.05, 0.08, 0.1, 0.15]:
            initializations.append(_lorentzian_profile(center, gamma, 1.0))
            
    for k in [2, 3, 4, 5, 6, 8]:
        initializations.append(_power_profile(k, 'right'))
        initializations.append(_power_profile(k, 'left'))
        initializations.append(_power_profile(k, 'both'))
        
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.05, 0.05), (0.0, 0.1)]:
        for w in [0.03, 0.05, 0.08, 0.12]:
            initializations.append(_two_gaussian(c1, w, c2, w, 1.0, 1.0))
            
    for width in [0.05, 0.1, 0.15]:
        for center in [0.0, 0.05, -0.05]:
            initializations.append(_sinc_like_profile(width, center))
            
    rng = np.random.RandomState(42)
    for _ in range(10):
        h_rand = rng.random(N)
        initializations.append(h_rand)
        
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        s, _ = _ratio_and_grad(h_init)
        scored_inits.append((s, h_init))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    if scored_inits[0][0] > best_score:
        best_score = scored_inits[0][0]
        best_h = scored_inits[0][1].copy()
        
    top_k = min(15, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 5.0:
            break
            
        s_init, h_init = scored_inits[i]
        
        h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=200, tol=1e-9)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
        if _time_left(start_time) < 5.0:
            break
            
        z_init = _dft_inverse(h_init)
        h_dft_opt, s_dft = _optimize_from_z(z_init, start_time, max_iter=100, tol=1e-8)
        if s_dft > best_score:
            best_score = s_dft
            best_h = h_dft_opt.copy()
            
    if _time_left(start_time) > 10.0:
        h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=500, tol=1e-12)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()
            
        if _time_left(start_time) > 5.0:
            z_polish = _dft_inverse(best_h)
            h_dft_polish, s_dft_polish = _optimize_from_z(z_polish, start_time, max_iter=200, tol=1e-10)
            if s_dft_polish > best_score:
                best_score = s_dft_polish
                best_h = h_dft_polish.copy()
                
    return best_h.tolist()
