import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _objective_grad(h, beta=100.0):
    """
    Computes objective value and analytical gradient for the smooth surrogate.
    f(h) = M(h) / S(h)^2, where M(h) = log-sum-exp(beta * |ac|) - beta*max + max (approx)
    Actually M(h) is the soft max.
    Let L(h) = log(sum exp(beta * |ac_k|)). M(h) = (L(h) + beta * max_k |ac_k|) / beta.
    Wait, the standard log-sum-exp trick is:
    m = max |ac|
    M = m + (1/beta) * log(sum exp(beta * (|ac| - m)))
    
    Gradient:
    dM/dh_i = (1/beta) * d/dh_i [ log(sum exp(beta * (|ac_k| - m))) ]
    Note m is the max, so |ac_k| - m <= 0.
    Let w_k = exp(beta * (|ac_k| - m)).
    Sum W = sum w_k.
    dM/dh_i = (1/beta) * (1/Sum W) * sum_k (w_k * beta * d|ac_k|/dh_i)
    dM/dh_i = (1/Sum W) * sum_k (w_k * d|ac_k|/dh_i)
    
    d|ac_k|/dh_i:
    ac_k = sum_j h_j h_{j+k}.
    d ac_k / d h_i = h_{i+k} + h_{i-k} (with boundary handling: if index out of bounds, term is 0).
    d |ac_k| / d h_i = sign(ac_k) * (h_{i+k} + h_{i-k}).
    
    So grad_M[i] = (1/Sum W) * sum_k (w_k * sign(ac_k) * (h_{i+k} + h_{i-k})).
    
    This sum can be computed efficiently via convolution/FFT?
    Let g_k = w_k * sign(ac_k).
    Then grad_M = (1/Sum W) * (g * h + g * h[::-1])?
    Let's check:
    sum_k g_k h_{i+k} is convolution (g * h)_i.
    sum_k g_k h_{i-k} is convolution (g * h[::-1])_i? No, correlation.
    Actually, let's just use the FFT definition.
    G = FFT(g). H = FFT(h).
    The term sum_k g_k h_{i+k} is the circular convolution of g and h, but ac is linear.
    Since g is defined on 0..2N-2, and h on 0..N-1.
    We can pad h and g to size 2N-1 or power of 2.
    Let's compute via FFT to be fast.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        M_val = 0.0
        grad_M = np.zeros(2*N-1)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        M_val = m + np.log(sum_exp) / beta
        
        # Weights for gradient
        w = exp_scaled / sum_exp
        # d|ac_k|/dh_i = sign(ac_k) * (h_{i+k} + h_{i-k})
        signs = np.sign(ac)
        g = w * signs
        
        # Compute grad_M[i] = sum_k g_k (h_{i+k} + h_{i-k})
        # This is (g * h) + (g * h_rev) roughly.
        # Let's use FFT for linear convolution.
        fsize = _get_fft_size(2*N) # Safe size
        G = np.fft.rfft(g, fsize)
        H = np.fft.rfft(h, fsize)
        
        # conv1: sum_k g_k h_{i+k}
        # This is cross-correlation of g and h?
        # Let's verify indices.
        # ac_k = sum_j h_j h_{j+k}.
        # d ac_k / d h_i:
        # If j=i, term is h_{i+k}.
        # If j+k=i => j=i-k, term is h_{i-k}.
        # So yes.
        
        # Linear convolution c[i] = sum_k g[k] h[i-k] ?
        # np.convolve(g, h) gives c[i] = sum_k g[k] h[i-k].
        # We want sum_k g[k] h[i+k].
        # This is np.convolve(g, h[::-1])[::-1] ? Or just correlation.
        # Let's use the property:
        # sum_k g_k h_{i+k} = (g * h)[i] where * is correlation?
        # Correlation r[i] = sum_k g[k] h[k+i].
        # Yes, that's what we want.
        # Correlation can be done via FFT: IFFT( conj(G) * H ) ?
        # Let's just use np.correlate or manual FFT.
        # Actually, simpler:
        # sum_k g_k h_{i+k} is the convolution of g and h, but with g reversed?
        # Let's stick to the gradient definition and compute via FFT of the product in frequency domain if possible, or just use the fact that it's a linear operation.
        
        # Let's use the FFT approach for the sum.
        # We want v[i] = sum_k g[k] h[i+k] + g[k] h[i-k].
        # This is v = corr(g, h) + conv(g, h)?
        # corr(g, h)[i] = sum_k g[k] h[k+i].
        # conv(g, h)[i] = sum_k g[k] h[i-k].
        # So v = corr(g, h) + conv(g, h).
        
        # Compute corr(g, h) via FFT:
        # corr(g, h) = IFFT( conj(G) * H ) ?
        # Let's check: IFFT( G * conj(H) ) is conv(g, h).
        # IFFT( conj(G) * H ) is corr(g, h).
        
        G_full = np.fft.rfft(g, fsize)
        H_full = np.fft.rfft(h, fsize)
        
        conv_val = np.fft.irfft(G_full * H_full, fsize)
        corr_val = np.fft.irfft(np.conj(G_full) * H_full, fsize)
        
        # We need the linear part.
        # g has length 2N-1. h has length N.
        # conv length 3N-2.
        # We want indices 0..N-1 for the result v[i].
        # For conv(g, h)[i] = sum_{k=0}^{i} g[k] h[i-k].
        # For corr(g, h)[i] = sum_{k=0}^{N-1-i} g[k] h[k+i]. (Assuming g is long enough).
        
        # Let's just extract the valid parts.
        # conv part: conv_val[0:N]
        # corr part: corr_val[0:N]
        
        grad_M_full = (conv_val[0:N] + corr_val[0:N])
        
        # Wait, is it exactly that?
        # d ac_k / d h_i = h_{i+k} + h_{i-k}.
        # Sum over k.
        # Term 1: sum_k g_k h_{i+k}. This is corr(g, h) at i?
        # corr(g, h)[i] = sum_k g[k] h[k+i]. Yes.
        # Term 2: sum_k g_k h_{i-k}. This is conv(g, h) at i?
        # conv(g, h)[i] = sum_k g[k] h[i-k]. Yes.
        
        grad_M = grad_M_full
        
    obj = M_val / (s * s)
    
    # d obj / d h_i = (s^2 * grad_M_i - M_val * 2s * 1) / s^4
    # = grad_M_i / s^2 - 2 M_val / s^3
    # = (grad_M_i - 2 M_val / s) / s^2
    
    grad_obj = (grad_M - (2.0 * M_val / s)) / (s * s)
    
    return obj, grad_obj

def _generate_candidate(rng, strategy=0):
    h = np.zeros(N)
    t = np.arange(N)
    
    if strategy == 0:
        h = rng.standard_normal(N)
    elif strategy == 1:
        h = rng.choice([1.0, -1.0], N)
    elif strategy == 2:
        mask = rng.random(N) < 0.2
        h[mask] = rng.choice([1.0, -1.0], mask.sum())
    elif strategy == 3:
        num_blocks = 10
        block_size = N // num_blocks
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else N
            sign = rng.choice([1.0, -1.0])
            h[start:end] = sign
    elif strategy == 4:
        noise = rng.standard_normal(N)
        H = np.fft.rfft(noise)
        cutoff = 10
        H[cutoff:] = 0
        h = np.fft.irfft(H, N)
        if np.sum(np.abs(h)) < 1e-6:
            h = rng.standard_normal(N)
    elif strategy == 5:
        x = np.arange(N)
        env = np.cos(np.pi * x / N) ** 2
        signs = rng.choice([1.0, -1.0], N)
        h = env * signs
    elif strategy == 6:
        # Chirp
        freq = 1.0 + rng.random() * 2.0
        phase = 2 * np.pi * freq * (t**2) / N
        h = np.sign(np.cos(phase))
        h[h == 0] = 1.0
    elif strategy == 7:
        # Golden Ratio
        phi = (1 + np.sqrt(5)) / 2
        h = np.sign(np.cos(2 * np.pi * phi * t))
        h[h == 0] = 1.0
    elif strategy == 8:
        # Zadoff-Chu
        root = rng.integers(1, N)
        n = np.arange(N)
        phase = -np.pi * root * n * (n + 1) / N
        h = np.cos(phase)
        h += rng.standard_normal(N) * 0.1
    elif strategy == 9:
        # Frank-Zadoff-Chu
        m = rng.choice([2, 4, 8])
        n = np.arange(N)
        phase1 = np.pi * n**2 / N
        phase2 = np.pi * (n % m) * n / N
        phase = phase1 + phase2
        h = np.cos(phase)
        h += rng.standard_normal(N) * 0.05
    elif strategy == 10:
        # Multi-chirp superposition
        num_chirps = rng.integers(2, 6)
        h = np.zeros(N)
        for _ in range(num_chirps):
            freq = rng.random() * 3.0 + 0.5
            phase = 2 * np.pi * freq * (t**2) / N
            h += np.cos(phase)
    else:
        h = rng.standard_normal(N)
        
    return h

def _local_search_block(h, deadline, max_iters=10):
    h = h.copy()
    # Keep continuous values but maybe snap to sign if it helps?
    # The task says "signed step heights", but usually continuous optimization finds better values.
    # The score is max|ac|/s^2.
    # Let's stick to continuous local search.
    
    best_score = _score(h)
    block_sizes = [1, 3, 5, 7]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        sizes_to_try = block_sizes.copy()
        np.random.shuffle(sizes_to_try)
        
        for bs in sizes_to_try:
            if time.time() > deadline:
                break
            num_flips = 50
            for _ in range(num_flips):
                if time.time() > deadline:
                    break
                start = np.random.randint(0, N - bs + 1)
                h[start:start+bs] = -h[start:start+bs]
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]
                    
        if not improved:
            for _ in range(20):
                if time.time() > deadline:
                    break
                j = np.random.randint(0, N)
                h[j] = -h[j]
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[j] = -h[j]
            
            if not improved:
                break
                
    return h, best_score

def _gradient_refine(h, deadline):
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    best_score = _score(h)
    
    for beta in [50, 100, 200, 400]:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        def obj_grad(x):
            val, grad = _objective_grad(x, beta=beta)
            return val, grad
        
        try:
            res = minimize(obj_grad, h, method='L-BFGS-B', 
                           jac=True,
                           options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                h_grad = res.x
                s = np.sum(h_grad)
                if abs(s) > 1e-9:
                    h_grad = h_grad / s
                    val = _score(h_grad)
                    if val < best_score:
                        best_score = val
                        h = h_grad
        except Exception:
            pass
            
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse candidates
    candidates = []
    strategies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    num_candidates = 30
    for i in range(num_candidates):
        if time.time() > deadline - 30:
            break
        strat = strategies[i % len(strategies)]
        h = _generate_candidate(rng, strat)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        val = _score(h)
        candidates.append((val, h))
        
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # 2. Refine top candidates with gradient optimization (using analytical gradient)
    top_k = 10
    
    for i in range(min(top_k, len(candidates))):
        if time.time() > deadline - 20:
            break
        init_val, h_init = candidates[i]
        grad_deadline = min(deadline - 10, time.time() + 10.0)
        h_opt, score_opt = _gradient_refine(h_init, grad_deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
        
        # Add to pool
        candidates.append((score_opt, h_opt.copy()))
        
    # Sort again to pick best for local search
    candidates.sort(key=lambda x: x[0])
    
    # 3. Local search on best candidates
    pool = [(best_score, best_h)]
    for i in range(min(5, len(candidates))):
        pool.append(candidates[i])
    pool.sort(key=lambda x: x[0])
    
    for i in range(min(5, len(pool))):
        if time.time() > deadline - 8:
            break
        init_val, h_init = pool[i]
        local_deadline = min(deadline - 3, time.time() + 15.0)
        h_opt, score_opt = _local_search_block(h_init, local_deadline, max_iters=15)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # 4. Final polish with high beta gradient
    if time.time() < deadline - 2:
        final_deadline = deadline - 1
        h_opt, score_opt = _gradient_refine(best_h, final_deadline)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
