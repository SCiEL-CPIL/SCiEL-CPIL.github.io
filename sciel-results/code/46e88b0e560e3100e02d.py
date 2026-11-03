import numpy as np
import time

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


def _sidon_set(max_size, seed=0):
    """Greedy Sidon-like set with randomization."""
    rng = np.random.default_rng(seed)
    s = [0]
    diffs = set()
    order = list(range(1, N))
    rng.shuffle(order)
    for c in order:
        new_diffs = set()
        ok = True
        for i in s:
            d = abs(c - i)
            if d in diffs or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            s.append(c)
            diffs.update(new_diffs)
        if len(s) >= max_size:
            break
    return sorted(s)


def _best_response_dynamics(h, deadline, max_iters=3000):
    """
    Discrete potential game: each index i is a player choosing h_i.
    Payoff for player i: contribution to L2^2 / (L1 * sup_conv).
    Best response: set h_i to 0 if it disproportionately inflates the peak,
    otherwise set to optimal local value balancing L2 gain vs peak cost.
    """
    h = h.copy().astype(float)
    s = h.sum()
    if s < 1e-15:
        return h
    h = h / s
    best_h = h.copy()
    best_r = _ratio(h)
    stale = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        conv = _conv_at_lags(h)
        sup_conv = conv.max()
        if sup_conv < 1e-15:
            break
        k_star = int(np.argmax(conv))
        
        # Compute marginal contribution of each h_i to the peak
        # grad_i = d(conv[k_star])/d(h_i) = h[k_star-i] + h[i-k_star] (if valid)
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
        
        # For each index, compute the "payoff" of having mass there
        # Payoff_i = 2*h_i / (L1 * sup_conv) - lambda * grad_i
        # where lambda is a Lagrange multiplier for the peak constraint
        L1 = h.sum()
        L2_sq = (h * h).sum()
        
        # Estimate lambda: we want to balance L2 gain against peak inflation
        # At optimum, marginal L2 gain per unit mass = marginal peak cost
        # d(L2_sq)/d(h_i) = 2*h_i
        # d(sup_conv)/d(h_i) ≈ grad_i (if k_star is the active constraint)
        # So lambda ≈ 2*h_i / (L1 * grad_i) for active indices
        
        # Best response: for each i, decide whether to keep, increase, or zero out
        new_h = h.copy()
        improved = False
        
        for i in range(N):
            if h[i] < 1e-12:
                # Consider adding mass here
                # Estimate peak increase: grad_i (currently 0 or small)
                # Estimate L2 gain: 2*0 = 0, but second order...
                # Use a small trial mass
                trial = h.copy()
                trial[i] = 0.01
                trial /= trial.sum()
                r_trial = _ratio(trial)
                if r_trial > best_r + 1e-10:
                    new_h[i] = 0.01
                    new_h /= new_h.sum()
                    improved = True
                continue
            
            # Current mass h[i]
            # Marginal L2 contribution: 2*h[i]
            # Marginal peak cost: grad[i]
            # If grad[i] is large relative to h[i], this index is "overloaded"
            
            # Try zeroing out
            if h[i] > 1e-8:
                trial = h.copy()
                trial[i] = 0.0
                if trial.sum() > 1e-15:
                    trial /= trial.sum()
                    r_trial = _ratio(trial)
                    if r_trial > best_r + 1e-10:
                        new_h[i] = 0.0
                        improved = True
                        continue
            
            # Try adjusting: optimal h_i given others
            # Maximize 2*h_i*h_others / (L1 * sup) approximately
            # Simple: try a few candidate values
            candidates = [h[i] * 0.5, h[i] * 0.75, h[i] * 1.25, h[i] * 1.5, 0.0]
            best_cand = h[i]
            best_cand_r = best_r
            
            for cand in candidates:
                if cand < 0:
                    continue
                trial = h.copy()
                trial[i] = cand
                s_trial = trial.sum()
                if s_trial < 1e-15:
                    continue
                trial /= s_trial
                r_trial = _ratio(trial)
                if r_trial > best_cand_r:
                    best_cand_r = r_trial
                    best_cand = cand
            
            if best_cand != h[i]:
                new_h[i] = best_cand
                improved = True
        
        if not improved:
            stale += 1
            if stale > 30:
                # Perturb: randomly zero out one heavy index
                if h.sum() > 1e-15:
                    heavy_idx = np.argsort(h)[::-1][:5]
                    if len(heavy_idx) > 0:
                        idx_to_zero = heavy_idx[np.random.randint(len(heavy_idx))]
                        new_h = h.copy()
                        new_h[idx_to_zero] = 0.0
                        if new_h.sum() > 1e-15:
                            new_h /= new_h.sum()
                            r_new = _ratio(new_h)
                            if r_new > best_r:
                                best_r = r_new
                                best_h = new_h.copy()
                            stale = 0
                        else:
                            break
            if stale > 100:
                break
        else:
            stale = 0
            new_h = new_h / new_h.sum()
            r_new = _ratio(new_h)
            if r_new > best_r:
                best_r = r_new
                best_h = new_h.copy()
            h = new_h
    
    return best_h


def _sparse_support_search(h_init, deadline, max_iters=2000):
    """
    Search over sparse support patterns (Sidon-like) with optimized weights.
    """
    best_h = h_init.copy()
    best_r = _ratio(h_init)
    support = np.where(best_h > 1e-8)[0]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
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
        
        # Add one index not in support (prefer Sidon-compatible)
        not_in = np.setdiff1d(np.arange(N), support)
        # Score candidates by how well they fit Sidon property
        diffs = set()
        for a in support:
            for b in support:
                if a != b:
                    diffs.add(abs(a - b))
        
        for i in not_in[:30]:
            new_diffs = set()
            ok = True
            for j in support:
                d = abs(i - j)
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            bonus = 0.5 if ok else 0.0
            new_support = np.sort(np.append(support, i))
            h_trial = np.zeros(N)
            h_trial[new_support] = 1.0
            r = _ratio(h_trial) + bonus * 0.01
            improvements.append((r, 'add', i, new_support))
        
        # Swap one index
        if len(support) >= 4:
            for i in support[:15]:
                for j in not_in[:15]:
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
            # Random perturbation
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
        from scipy.optimize import minimize
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


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    def elapsed_ok(margin):
        return (time.time() - t0) < DEADLINE - margin
    
    # ---- Generate diverse initial candidates ----
    candidates = []
    
    # 1. Sidon sets of various sizes (key structural insight)
    for sz in (6, 8, 10, 12, 14, 16, 18):
        for seed in (0, 1, 2):
            s = _sidon_set(sz, seed)
            if len(s) >= 3:
                h = np.zeros(N)
                for i in s:
                    h[i] = 1.0
                candidates.append(h / h.sum())
    
    # 2. Random sparse sets with varying density
    for _ in range(30):
        k = int(rng.integers(5, 20))
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
    
    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_score = _ratio(best_h)
    for c in candidates[1:]:
        sc = _ratio(c)
        if sc > best_score:
            best_score = sc
            best_h = c.copy()
    
    # ---- Phase 1: Sparse support search from best candidate ----
    if elapsed_ok(40):
        try:
            h_ls, r_ls = _sparse_support_search(best_h.copy(), DEADLINE - 45.0, max_iters=1000)
            if r_ls > best_score:
                best_score, best_h = r_ls, h_ls
        except Exception:
            pass
    
    # ---- Phase 2: Best-response dynamics (discrete potential game) ----
    if elapsed_ok(30):
        try:
            h_br = _best_response_dynamics(best_h.copy(), DEADLINE - 35.0, max_iters=3000)
            sc = _ratio(h_br)
            if sc > best_score:
                best_score, best_h = sc, h_br
        except Exception:
            pass
    
    # ---- Phase 3: Another round of support search from improved candidate ----
    if elapsed_ok(20):
        try:
            h_ls2, r_ls2 = _sparse_support_search(best_h.copy(), DEADLINE - 25.0, max_iters=800)
            if r_ls2 > best_score:
                best_score, best_h = r_ls2, h_ls2
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
    
    # ---- Phase 5: Final best-response pass ----
    if elapsed_ok(10):
        try:
            h_final = _best_response_dynamics(best_h.copy(), DEADLINE - 5.0, max_iters=2000)
            sc = _ratio(h_final)
            if sc > best_score:
                best_score, best_h = sc, h_final
        except Exception:
            pass
    
    # ---- Phase 6: Coordinate polish ----
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
