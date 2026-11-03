import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

_FREQ_IDX = np.arange(_FSIZE // 2 + 1)
_EXP_TABLE = np.exp(-2j * np.pi * _FREQ_IDX / _FSIZE)

def _autoconv_full(h):
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs(ac, beta=50.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    log_sum = np.log(np.sum(exp_scaled))
    soft_max = m + log_sum / beta
    # Gradient wrt ac: beta * exp_scaled / sum(exp_scaled)
    grad_ac = beta * exp_scaled / np.sum(exp_scaled)
    return soft_max, grad_ac

def _smooth_obj_and_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    sm, grad_ac = _soft_max_abs(ac, beta)
    obj = sm / (s * s)
    # Gradient: d(sm/s^2)/dh = (grad_ac * d(ac)/dh) / s^2 - 2*sm/s^3 * sum(dh)
    # d(ac)/dh: ac = ifft(H^2), H = fft(h)
    # This is complex; use numerical or simplify
    # For simplicity, use a simplified gradient
    grad = np.zeros_like(h)
    # Approximate gradient using the structure
    return obj, grad

def _smooth_obj(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _local_search_flip_fast(h, max_passes=5, deadline=None):
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            new_sum = current_sum - 2 * old_val
            delta = -2 * old_val * _EXP_TABLE[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            if abs(new_sum) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
            if new_val < best_val:
                best_val = new_val
                h[i] = -old_val
                H = H_new
                current_sum = new_sum
                improved = True
        if not improved:
            break
    return h, best_val

def _local_search_block_fast(h, deadline=None, block_sizes=[3,5,7]):
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            step = max(1, (N - bs) // 80)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                new_sum = current_sum - 2 * np.sum(block)
                inner_sum = np.sum(block * _EXP_TABLE[:bs])
                delta_vec = -2.0 * _EXP_TABLE[start] * inner_sum
                H_new = H + delta_vec
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                if abs(new_sum) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    current_sum = new_sum
                    improved = True
                else:
                    h[start:start+bs] = block
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    t = np.arange(N)
    candidates = [best_x.copy()]
    
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:8]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 60:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        for beta in [20, 50, 100, 200]:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(_smooth_obj, h_norm, args=(beta,), method='L-BFGS-B',
                               options={'maxiter': 500, 'disp': False, 'gtol': 1e-9,
                                        'ftol': 1e-11, 'maxls': 40})
                h_opt = res.x
                val = compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                
                h_norm = h_bin / np.sum(h_bin)
            except Exception:
                pass
    
    if time.time() < deadline - 30:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        H = np.fft.rfft(h_work, _FSIZE)
        current_sum = np.sum(h_work)
        best_val_work = best_val
        
        for round_num in range(6):
            if time.time() > deadline - 15:
                break
            ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
            abs_ac = np.abs(ac)
            peak_indices = np.argsort(abs_ac)[::-1][:10]
            
            improved_this_round = False
            for peak_idx in peak_indices:
                if time.time() > deadline - 12:
                    break
                center = peak_idx % N
                for flip_offset in range(-4, 5):
                    if time.time() > deadline - 10:
                        break
                    j = (center + flip_offset) % N
                    old_val = h_work[j]
                    new_sum = current_sum - 2 * old_val
                    delta = -2 * old_val * _EXP_TABLE[j]
                    H_new = H + delta
                    conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                    if abs(new_sum) < 1e-12:
                        new_val = float('inf')
                    else:
                        new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                    if new_val < best_val_work:
                        best_val_work = new_val
                        h_work[j] = -old_val
                        H = H_new
                        current_sum = new_sum
                        improved_this_round = True
                        break
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
        
        if best_val_work < best_val:
            best_val = best_val_work
            best_x = h_work.copy()
    
    if time.time() < deadline - 20:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_flip_fast(h_work, max_passes=10, deadline=deadline - 10)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        if time.time() < deadline - 8:
            h_work, val_work = _local_search_block_fast(h_work, deadline=deadline - 5,
                                                        block_sizes=[2, 3, 4, 5, 6, 7, 9, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    if time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 8:
                break
            perturbation_type = rng.integers(0, 4)
            if perturbation_type == 0:
                x_rand = rng.choice([1.0, -1.0], N)
            elif perturbation_type == 1:
                x_rand = best_x.copy()
                n_flips = rng.integers(5, 25)
                flip_idx = rng.choice(N, n_flips, replace=False)
                x_rand[flip_idx] = -x_rand[flip_idx]
            elif perturbation_type == 2:
                x_rand = best_x.copy()
                bs = rng.integers(3, 15)
                start = rng.integers(0, N - bs)
                x_rand[start:start+bs] = -x_rand[start:start+bs]
            else:
                x_rand = best_x.copy()
                for _ in range(rng.integers(2, 4)):
                    bs = rng.integers(2, 6)
                    start = rng.integers(0, N - bs)
                    x_rand[start:start+bs] = -x_rand[start:start+bs]
            
            h_work, val_work = _local_search_flip_fast(x_rand, max_passes=6, deadline=deadline - 5)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            h_work, val_work = _local_search_flip_fast(h_work, max_passes=6, deadline=deadline - 1)
            if val_work >= best_val:
                break
            best_val = val_work
            best_x = h_work.copy()
    
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
