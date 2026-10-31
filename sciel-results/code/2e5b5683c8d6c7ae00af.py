import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0

def _score(u):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    n_sums, n_diffs = _compute_set_sizes_fft(u)
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _compute_set_sizes_fft(u):
    """Compute |U+U| and |U-U| using FFT for exact counts."""
    if not u:
        return 0, 0
    m = max(u)
    size = 2 * m + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        vec[x] = 1.0
    fft_vec = np.fft.fft(vec)
    sum_fft = fft_vec * fft_vec
    sum_conv = np.fft.ifft(sum_fft).real
    n_sums = int(np.sum(sum_conv[:size] > 0.5))
    diff_fft = fft_vec * np.conj(fft_vec)
    diff_conv = np.fft.ifft(diff_fft).real
    n_diffs = int(np.sum(diff_conv[:m+1] > 0.5))
    if m > 0:
        n_diffs += int(np.sum(diff_conv[fft_size-m:fft_size] > 0.5))
    return n_sums, n_diffs

def _score_fast(u_list):
    """Score using brute force for small sets, FFT for larger."""
    if not u_list or len(u_list) < 2:
        return 0.0
    m = max(u_list)
    if m < 1:
        return 0.0
    if m <= 300:
        sums = set()
        diffs = set()
        for a in u_list:
            for b in u_list:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
        if n_sums == 0:
            return 0.0
        ratio = n_diffs / float(n_sums)
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * m + 1)
    else:
        return _score(u_list)

def _spectral_optimize(L, n_harmonics, n_iter, lr, seed, n_target):
    """
    Optimize a set U via spectral parameterization.
    f(x) = sum_k c_k * cos(2*pi*k*x/L)
    U = {x in [0, L) : f(x) > threshold}
    We optimize c_k to maximize log(|U-U|) - log(|U+U|).
    """
    if time.time() > _DEADLINE - 15.0:
        return None, 0.0
    
    rng = random.Random(seed)
    
    # Initialize coefficients
    c = np.zeros(n_harmonics + 1)
    for k in range(1, n_harmonics + 1):
        c[k] = rng.uniform(-1, 1)
    
    # Precompute cos values: cos_vals[x, k] = cos(2*pi*k*x/L)
    x_vals = np.arange(L, dtype=np.float64)
    cos_vals = np.zeros((L, n_harmonics + 1))
    for k in range(1, n_harmonics + 1):
        cos_vals[:, k] = np.cos(2 * np.pi * k * x_vals / L)
    
    # FFT size
    fft_size = 1
    while fft_size < 2 * L:
        fft_size *= 2
    
    best_c = c.copy()
    best_score_val = -1e18
    best_u = None
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Compute f(x) for all x
        f_vals = np.zeros(L, dtype=np.float64)
        for k in range(1, n_harmonics + 1):
            f_vals += c[k] * cos_vals[:, k]
        
        # Select U: top n_target values
        if n_target < L:
            # Use softmax-like selection: threshold at n_target-th largest
            sorted_f = np.sort(f_vals)[::-1]
            threshold = sorted_f[min(n_target - 1, L - 1)]
            # Add small random tie-breaking
            f_vals += rng.random() * 1e-6
            u_mask = f_vals >= threshold
            # Ensure exactly n_target elements (or as close as possible)
            n_selected = int(np.sum(u_mask))
            if n_selected > n_target:
                # Remove excess: pick the ones closest to threshold
                selected_indices = np.where(u_mask)[0]
                f_selected = f_vals[selected_indices]
                # Remove the smallest ones
                if n_selected > n_target:
                    order = np.argsort(f_selected)
                    remove_set = set(selected_indices[order[:n_selected - n_target]].tolist())
                    u_mask[list(remove_set)] = False
            elif n_selected < n_target:
                # Add more: pick next largest
                unselected_indices = np.where(~u_mask)[0]
                f_unselected = f_vals[unselected_indices]
                order = np.argsort(f_unselected)[::-1]
                add_set = unselected_indices[order[:n_target - n_selected]].tolist()
                u_mask[add_set] = True
            
            # Ensure 0 is in U
            if not u_mask[0]:
                # Replace the smallest element with 0
                selected_indices = np.where(u_mask)[0]
                if len(selected_indices) > 0:
                    min_idx = selected_indices[np.argmin(f_vals[selected_indices])]
                    u_mask[min_idx] = False
                    u_mask[0] = True
            
            u_list = np.where(u_mask)[0].tolist()
            if len(u_list) < 2:
                continue
            u_list = sorted(set(u_list))
            
            # Compute score
            m = max(u_list) if u_list else 0
            if m < 1:
                continue
            
            # FFT computation
            size = 2 * m + 1
            fsize = 1
            while fsize < size:
                fsize *= 2
            vec = np.zeros(fsize, dtype=np.float64)
            for x in u_list:
                vec[x] = 1.0
            fft_vec = np.fft.fft(vec)
            sum_conv = np.fft.ifft(fft_vec * fft_vec).real
            n_sums = max(1, int(np.sum(sum_conv[:size] > 0.5)))
            diff_conv = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
            n_diffs = int(np.sum(diff_conv[:m+1] > 0.5))
            if m > 0:
                n_diffs += int(np.sum(diff_conv[fsize-m:fsize] > 0.5))
            
            if n_sums <= 0:
                continue
            
            ratio = n_diffs / float(n_sums)
            if ratio <= 0:
                continue
            
            score_val = 1.0 + math.log(ratio) / math.log(2 * m + 1)
            
            if score_val > best_score_val:
                best_score_val = score_val
                best_u = u_list[:]
                best_c = c.copy()
        
        # Gradient step: approximate gradient of log(n_diffs) - log(n_sums) w.r.t. c
        # Use finite differences for a few key coefficients
        for k in range(1, min(n_harmonics + 1, 8)):
            if time.time() > _DEADLINE - 10.0:
                break
            eps = 0.01
            c[k] += eps
            f_vals_plus = np.zeros(L, dtype=np.float64)
            for kk in range(1, n_harmonics + 1):
                f_vals_plus += c[kk] * cos_vals[:, kk]
            sorted_fp = np.sort(f_vals_plus)[::-1]
            thresh_p = sorted_fp[min(n_target - 1, L - 1)]
            mask_p = f_vals_plus >= thresh_p
            n_sel_p = int(np.sum(mask_p))
            if n_sel_p > n_target:
                sel_idx = np.where(mask_p)[0]
                f_sel = f_vals_plus[sel_idx]
                order = np.argsort(f_sel)
                rem = set(sel_idx[order[:n_sel_p - n_target]].tolist())
                mask_p[list(rem)] = False
            elif n_sel_p < n_target:
                unsel_idx = np.where(~mask_p)[0]
                f_unsel = f_vals_plus[unsel_idx]
                order = np.argsort(f_unsel)[::-1]
                mask_p[unsel_idx[order[:n_target - n_sel_p]]] = True
            if not mask_p[0]:
                sel_idx = np.where(mask_p)[0]
                if len(sel_idx) > 0:
                    min_idx = sel_idx[np.argmin(f_vals_plus[sel_idx])]
                    mask_p[min_idx] = False
                    mask_p[0] = True
            u_p = np.where(mask_p)[0].tolist()
            if len(u_p) >= 2:
                m_p = max(u_p)
                if m_p >= 1:
                    sz = 2 * m_p + 1
                    fs = 1
                    while fs < sz:
                        fs *= 2
                    v = np.zeros(fs, dtype=np.float64)
                    for x in u_p:
                        v[x] = 1.0
                    fv = np.fft.fft(v)
                    sc = np.fft.ifft(fv * fv).real
                    ns = max(1, int(np.sum(sc[:sz] > 0.5)))
                    dc = np.fft.ifft(fv * np.conj(fv)).real
                    nd = int(np.sum(dc[:m_p+1] > 0.5))
                    if m_p > 0:
                        nd += int(np.sum(dc[fs-m_p:fs] > 0.5))
                    if ns > 0 and nd > 0:
                        s_p = math.log(nd / float(ns)) / math.log(2 * m_p + 1)
                    else:
                        s_p = -1e6
            else:
                s_p = -1e6
            
            c[k] -= 2 * eps
            f_vals_minus = np.zeros(L, dtype=np.float64)
            for kk in range(1, n_harmonics + 1):
                f_vals_minus += c[kk] * cos_vals[:, kk]
            sorted_fm = np.sort(f_vals_minus)[::-1]
            thresh_m = sorted_fm[min(n_target - 1, L - 1)]
            mask_m = f_vals_minus >= thresh_m
            n_sel_m = int(np.sum(mask_m))
            if n_sel_m > n_target:
                sel_idx = np.where(mask_m)[0]
                f_sel = f_vals_minus[sel_idx]
                order = np.argsort(f_sel)
                rem = set(sel_idx[order[:n_sel_m - n_target]].tolist())
                mask_m[list(rem)] = False
            elif n_sel_m < n_target:
                unsel_idx = np.where(~mask_m)[0]
                f_unsel = f_vals_minus[unsel_idx]
                order = np.argsort(f_unsel)[::-1]
                mask_m[unsel_idx[order[:n_target - n_sel_m]]] = True
            if not mask_m[0]:
                sel_idx = np.where(mask_m)[0]
                if len(sel_idx) > 0:
                    min_idx = sel_idx[np.argmin(f_vals_minus[sel_idx])]
                    mask_m[min_idx] = False
                    mask_m[0] = True
            u_m = np.where(mask_m)[0].tolist()
            if len(u_m) >= 2:
                m_m = max(u_m)
                if m_m >= 1:
                    sz = 2 * m_m + 1
                    fs = 1
                    while fs < sz:
                        fs *= 2
                    v = np.zeros(fs, dtype=np.float64)
                    for x in u_m:
                        v[x] = 1.0
                    fv = np.fft.fft(v)
                    sc = np.fft.ifft(fv * fv).real
                    ns = max(1, int(np.sum(sc[:sz] > 0.5)))
                    dc = np.fft.ifft(fv * np.conj(fv)).real
                    nd = int(np.sum(dc[:m_m+1] > 0.5))
                    if m_m > 0:
                        nd += int(np.sum(dc[fs-m_m:fs] > 0.5))
                    if ns > 0 and nd > 0:
                        s_m = math.log(nd / float(ns)) / math.log(2 * m_m + 1)
                    else:
                        s_m = -1e6
            else:
                s_m = -1e6
            
            c[k] += eps  # restore
            
            grad = (s_p - s_m) / (2 * eps)
            c[k] += lr * grad
            # Clamp coefficients
            c[k] = np.clip(c[k], -5, 5)
    
    return best_u, best_score_val

def _local_refine(u, timeout=3.0):
    """Light local refinement: try moving each element to nearby positions."""
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score_fast(u)
    
    current_score = _score_fast(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    improved = True
    while improved and time.time() < start_time + timeout and time.time() < _DEADLINE - 5.0:
        improved = False
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
                break
            old_val = u[i]
            for delta in range(-max(1, max_val // 10), max(1, max_val // 10) + 1, max(1, max_val // 20)):
                new_val = old_val + delta
                if new_val < 0 or new_val > max_val * 2:
                    continue
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score_fast(new_u)
                if s > current_score + 1e-12:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
    
    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Phase 1: Spectral optimization with various parameters
    # L = period, n_harmonics = number of cos terms, n_iter = gradient steps
    # n_target = desired set size
    
    spectral_configs = [
        # (L, n_harmonics, n_iter, lr, n_target, seed)
        (200, 5, 50, 0.1, 10, 0),
        (200, 5, 50, 0.1, 12, 1),
        (300, 6, 40, 0.1, 15, 2),
        (300, 6, 40, 0.1, 18, 3),
        (400, 7, 35, 0.1, 20, 4),
        (400, 7, 35, 0.1, 22, 5),
        (500, 8, 30, 0.1, 25, 6),
        (500, 8, 30, 0.1, 28, 7),
        (150, 4, 60, 0.15, 8, 8),
        (150, 4, 60, 0.15, 10, 9),
        (250, 5, 50, 0.1, 14, 10),
        (350, 6, 40, 0.1, 18, 11),
        (450, 7, 35, 0.1, 22, 12),
        (600, 8, 25, 0.1, 30, 13),
        (200, 6, 50, 0.08, 11, 14),
        (300, 7, 40, 0.08, 16, 15),
        (400, 8, 35, 0.08, 20, 16),
        (250, 5, 55, 0.12, 13, 17),
    ]
    
    for cfg in spectral_configs:
        if time.time() > _DEADLINE - 15.0:
            break
        L, nh, ni, lr, nt, sd = cfg
        u_opt, s_opt = _spectral_optimize(L, nh, ni, lr, sd, nt)
        if u_opt and len(u_opt) >= 2 and s_opt > best_s:
            best_s = s_opt
            best_u = u_opt[:]
    
    # Phase 2: Try some structured sets (Golomb-like, power-based)
    structured_sets = []
    # Powers of 2
    for base in [2, 3]:
        s = [0]
        v = 1
        while v < 300:
            s.append(v)
            v *= base
        structured_sets.append(sorted(set(s)))
    # Arithmetic progressions with gaps
    for start in range(1, 10):
        for step in [3, 4, 5, 7]:
            s = [0]
            v = start
            while v < 300:
                s.append(v)
                v += step
            structured_sets.append(sorted(set(s)))
    # Mixed: some powers + some random
    for seed in range(5):
        rng = random.Random(seed)
        s = [0, 1, 4, 9, 16]
        for _ in range(10):
            s.append(rng.randint(1, 200))
        structured_sets.append(sorted(set(s)))
    
    for s in structured_sets:
        if time.time() > _DEADLINE - 10.0:
            break
        if len(s) >= 2:
            sc = _score_fast(s)
            if sc > best_s:
                best_s = sc
                best_u = s[:]
    
    # Phase 3: Local refinement on the best solution
    if time.time() < _DEADLINE - 5.0:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            u_opt, s_opt = _local_refine(best_u, timeout=min(remaining, 8.0))
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
    
    # Ensure 0 is in the set
    if 0 not in best_u:
        best_u = [0] + best_u
    
    return sorted(set(best_u))
