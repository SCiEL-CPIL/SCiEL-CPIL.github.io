import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 143.0


def _autoconv_sup(h):
    """Sup norm of autoconvolution sampled at breakpoints."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    return float(np.max(c)) * (1.0 / n)


def _ratio(h):
    h = np.maximum(h, 0.0)
    s1 = float(np.sum(h))
    if s1 < 1e-15:
        return 0.0
    s2 = float(np.sum(h * h))
    sup = _autoconv_sup(h)
    if sup < 1e-15:
        return 0.0
    return s2 / (s1 * sup)


def _conv_at_lags(h):
    """Return the autoconvolution values at all breakpoints 0..2N-2."""
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    return c[:2 * n - 1]


def _support_local_search(h, deadline, max_iters=2000):
    """Local search over support: try adding, removing, or swapping indices.
    For each candidate support, optimize weights via a fast method."""
    best_h = h.copy()
    best_r = _ratio(h)
    support = np.where(best_h > 1e-8)[0]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try all single modifications: add, remove, swap
        improvements = []
        
        # Remove one index from support
        for i in support:
            new_support = np.delete(support, np.where(support == i)[0][0])
            if len(new_support) < 3:
                continue
            h_trial = np.zeros(N)
            h_trial[new_support] = 1.0
            r = _ratio(h_trial)
            improvements.append((r, 'remove', i, new_support))
        
        # Add one index not in support
        not_in = np.setdiff1d(np.arange(N), support)
        for i in not_in[:20]:  # limit for speed
            new_support = np.sort(np.append(support, i))
            h_trial = np.zeros(N)
            h_trial[new_support] = 1.0
            r = _ratio(h_trial)
            improvements.append((r, 'add', i, new_support))
        
        # Swap one index
        if len(support) >= 4:
            for i in support[:10]:
                for j in not_in[:10]:
                    new_support = np.sort(np.append(np.delete(support, np.where(support == i)[0][0]), j))
                    h_trial = np.zeros(N)
                    h_trial[new_support] = 1.0
                    r = _ratio(h_trial)
                    improvements.append((r, 'swap', (i, j), new_support))
        
        if not improvements:
            break
        
        improvements.sort(key=lambda x: x[0], reverse=True)
        best_imp = improvements[0]
        if best_imp[0] > best_r + 1e-10:
            best_r = best_imp[0]
            support = best_imp[3]
            best_h = np.zeros(N)
            best_h[support] = 1.0
        else:
            # No improvement; try random perturbation
            if len(support) > 3:
                idx = np.random.randint(len(support))
                support = np.delete(support, idx)
                new_idx = np.random.choice(np.setdiff1d(np.arange(N), support))
                support = np.sort(np.append(support, new_idx))
                best_h = np.zeros(N)
                best_h[support] = 1.0
                best_r = _ratio(best_h)
            else:
                break
    
    # Final weight optimization on best support
    best_h = _optimize_weights_on_support(best_h, deadline - 5.0)
    return best_h, _ratio(best_h)


def _optimize_weights_on_support(h, deadline):
    """Given a support pattern, optimize the relative weights."""
    support = np.where(h > 1e-8)[0]
    if len(support) < 2:
        return h
    
    def neg_r(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-15:
            return 0.0
        w = w / s
        full = np.zeros(N)
        full[support] = w
        return -_ratio(full)
    
    w0 = np.ones(len(support)) / len(support)
    try:
        res = minimize(neg_r, w0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        w = np.maximum(res.x, 0.0)
        s = w.sum()
        if s < 1e-15:
            return h
        w = w / s
        full = np.zeros(N)
        full[support] = w
        if _ratio(full) > _ratio(h):
            return full
    except Exception:
        pass
    return h


def _peak_avoidance_descent(h, deadline, eta0=0.05):
    """Directly penalize coordinates contributing to the peak lag."""
    h = h.copy()
    s = h.sum()
    if s < 1e-15:
        return h
    h = h / s
    best_h = h.copy()
    best_r = _ratio(h)
    stale = 0
    eta = eta0
    
    for it in range(4000):
        if time.time() > deadline:
            break
        conv = _conv_at_lags(h)
        sup_conv = conv.max()
        if sup_conv < 1e-15:
            break
        k_star = int(np.argmax(conv))
        
        # Compute gradient of conv[k_star] w.r.t. h[i]
        grad = np.zeros(N)
        for i in range(N):
            v = 0.0
            j1 = k_star - i
            if 0 <= j1 < N:
                v += h[j1]
            j2 = i - k_star
            if 0 <= j2 < N:
                v += h[j2]
            grad[i] = v
        
        # Multiplicative update
        h = h * np.exp(-eta * grad)
        s = h.sum()
        if s < 1e-15:
            break
        h = h / s
        r = _ratio(h)
        if r > best_r:
            best_r = r
            best_h = h.copy()
            stale = 0
        else:
            stale += 1
            if stale > 50:
                eta *= 0.95
                if eta < 1e-5:
                    break
    
    return best_h


def _sidon_set(max_size):
    """Greedy Sidon-like set: no two differences repeat."""
    s = [0]
    diffs = set()
    for c in range(1, N):
        new_diffs = set()
        ok = True
        for i in s:
            d = c - i
            if d in diffs or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            s.append(c)
            diffs.update(new_diffs)
        if len(s) >= max_size:
            break
    return s


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    def elapsed_ok(margin):
        return (time.time() - t0) < DEADLINE - margin
    
    # ---- Generate diverse initial candidates ----
    candidates = []
    
    # 1. Sidon sets of various sizes
    for sz in (6, 8, 10, 12, 14, 16):
        h = np.zeros(N)
        for i in _sidon_set(sz):
            h[i] = 1.0
        candidates.append(h / h.sum())
    
    # 2. Random sparse sets with varying density
    for _ in range(30):
        k = int(rng.integers(5, 18))
        h = np.zeros(N)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.3, 1.5, k)
        candidates.append(h / h.sum())
    
    # 3. Structured patterns
    for period in (2, 3, 4, 5, 7):
        h = np.zeros(N)
        h[::period] = 1.0
        candidates.append(h / h.sum())
    
    # 4. Gaussian-like and other smooth shapes
    t = np.linspace(-1, 1, N)
    for width in (0.2, 0.4, 0.6, 1.0):
        h = np.exp(-t**2 / width)
        candidates.append(h / h.sum())
    
    # 5. Two-peak patterns
    for d in (0.3, 0.5, 0.7):
        h = np.exp(-((t - d)**2) / 0.15) + np.exp(-((t + d)**2) / 0.15)
        candidates.append(h / h.sum())
    
    # 6. Random spectral masks
    for nb in (2, 3, 4, 5, 6, 8, 10):
        for _ in range(4):
            mask = np.zeros(N, dtype=complex)
            idx = rng.choice(N, size=nb, replace=False)
            mask[idx] = rng.uniform(0.5, 2.0, nb) + 1j * rng.uniform(-1.0, 1.0, nb)
            mask[0] += np.sum(mask.real) * 0.3
            h = np.abs(np.fft.ifft(mask))
            if h.sum() > 1e-15:
                candidates.append(h / h.sum())
    
    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_score = _ratio(best_h)
    for c in candidates[1:]:
        sc = _ratio(c)
        if sc > best_score:
            best_score = sc
            best_h = c.copy()
    
    # ---- Phase 1: Support local search from best candidate ----
    if elapsed_ok(30):
        try:
            h_ls, r_ls = _support_local_search(best_h.copy(), DEADLINE - 35.0, max_iters=500)
            if r_ls > best_score:
                best_score, best_h = r_ls, h_ls
        except Exception:
            pass
    
    # ---- Phase 2: Differential evolution (global) ----
    if elapsed_ok(25):
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0.0)
                s = float(np.sum(x))
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            bounds = [(0.0, 1.0)] * N
            res = differential_evolution(
                neg_ratio, bounds, seed=7, maxiter=50, popsize=30,
                tol=1e-8, init='sobol', polish=True,
                updating='immediate', mutation=(0.4, 1.0), recombination=0.7)
            h = np.maximum(res.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass
    
    # ---- Phase 3: Peak-avoidance descent ----
    if elapsed_ok(20):
        try:
            h_pa = _peak_avoidance_descent(best_h.copy(), DEADLINE - 25.0)
            sc = _ratio(h_pa)
            if sc > best_score:
                best_score, best_h = sc, h_pa
        except Exception:
            pass
    
    # ---- Phase 4: Weight optimization on current support ----
    if elapsed_ok(15):
        try:
            h_opt = _optimize_weights_on_support(best_h.copy(), DEADLINE - 20.0)
            sc = _ratio(h_opt)
            if sc > best_score:
                best_score, best_h = sc, h_opt
        except Exception:
            pass
    
    # ---- Phase 5: Nelder-Mead refinement ----
    if elapsed_ok(12):
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0.0)
                s = float(np.sum(x))
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            res = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            h = np.maximum(res.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass
    
    # ---- Phase 6: Final peak-avoidance pass ----
    if elapsed_ok(8):
        try:
            h_final = _peak_avoidance_descent(best_h.copy(), DEADLINE - 3.0)
            sc = _ratio(h_final)
            if sc > best_score:
                best_score, best_h = sc, h_final
        except Exception:
            pass
    
    # ---- Phase 7: Coordinate polish ----
    if elapsed_ok(5):
        try:
            cur = best_h.copy()
            for _ in range(50):
                if time.time() > DEADLINE - 2.0:
                    break
                improved = False
                for i in range(N):
                    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                        for sign in (1.0, -1.0):
                            trial = cur.copy()
                            trial[i] = max(0.0, cur[i] * (1.0 + sign * frac))
                            sc = _ratio(trial)
                            if sc > _ratio(cur) + 1e-12:
                                cur = trial
                                improved = True
                                if sc > best_score:
                                    best_score, best_h = sc, cur.copy()
                                break
                    if improved:
                        break
                if not improved:
                    break
        except Exception:
            pass
    
    return best_h.tolist()
