"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Strategy:
  1.  Fast, exact objective via FFT autoconvolution.
  2.  Global search over a compact basis expansion (differential evolution).
  3.  Smooth-surrogate local optimisation (adaptive beta log-sum-exp).
  4.  Direct local search with incremental autoconvolution updates.
  5.  Peak-suppression via projected gradient descent.
  6.  Final high-precision polish.

All phases are deadline-guarded so the program always returns within 150 s.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _autoconv_max_abs(h):
    """max |h*h| (linear autoconvolution) computed via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_smooth(h, beta=80.0):
    """Differentiable surrogate: soft_max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    soft = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return soft / (s * s)


def _autoconv_and_grad(h, peak_idx=None):
    """Compute autoconvolution and gradient of |ac[peak_idx]| w.r.t. h.
    
    Returns (ac, grad) where grad is the gradient of |ac[peak_idx]| w.r.t. h.
    If peak_idx is None, uses the argmax of |ac|.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    if peak_idx is None:
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
    
    peak_sign = np.sign(ac[peak_idx])
    if peak_sign == 0:
        peak_sign = 1.0
    
    # Gradient of ac[k] w.r.t. h[j] = h[k-j] (for valid j)
    # ac[k] = sum_j h[j] * h[k-j]
    # d(ac[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] (when both terms contribute)
    # But for linear conv: ac[k] = sum_{j=0}^{n-1} h[j]*h[k-j] where h is zero-padded
    # d(ac[k])/d(h[j]) = h[k-j] if k-j in [0,n-1], else 0
    # Plus the symmetric term: when we vary h[j], it also appears as h[k-j'] where j'=k-j
    # Actually ac[k] = sum_{j=0}^{n-1} h[j] h[k-j] (with h[m]=0 for m outside [0,n-1])
    # d(ac[k])/d(h[j]) = h[k-j] + (if j != k-j, the term where h[j] is the second factor)
    # Wait, let's be careful: ac[k] = sum_{j} h[j] h[k-j]
    # d/d(h[j0]) sum_j h[j] h[k-j] = h[k-j0] + sum_j delta_{j,j0} h[k-j] ... no
    # d/d(h[j0]) sum_j h[j] h[k-j] = (d/d(h[j0]) h[j0]) * h[k-j0] + sum_j h[j] * (d/d(h[j0]) h[k-j])
    # = h[k-j0] + sum_j h[j] * delta_{k-j, j0} = h[k-j0] + h[k-j0] (if k-j0 in range, j0 in range)
    # Wait, the second term: sum_j h[j] delta_{k-j, j0} = h[j] where j = k-j0, so = h[k-j0] if k-j0 in [0,n-1]
    # So d(ac[k])/d(h[j0]) = 2 * h[k-j0] if both j0 and k-j0 are in [0, n-1]
    # But if j0 == k-j0, i.e., j0 = k/2, then it's just h[k/2] (not doubled)
    # Actually no: the sum is over j from 0 to n-1. h[j] h[k-j].
    # When j=j0: term is h[j0] h[k-j0], derivative w.r.t h[j0] is h[k-j0]
    # When k-j=j0, i.e., j=k-j0: term is h[k-j0] h[j0], derivative w.r.t h[j0] is h[k-j0]
    # If j0 != k-j0, these are two different terms, so total derivative is 2*h[k-j0]
    # If j0 == k-j0 (i.e., 2*j0 = k), it's the same term, derivative is h[j0]
    
    grad = np.zeros(n)
    kj = peak_idx - np.arange(n)
    mask = (kj >= 0) & (kj < n)
    if np.any(mask):
        grad[mask] += peak_sign * h[kj[mask]]
    # The symmetric contribution
    # For j0 where k-j0 is in [0,n-1] and j0 in [0,n-1], we add h[k-j0]
    # But we need to handle the case j0 == k-j0 carefully
    # Actually the formula d(ac[k])/d(h[j0]) = 2*h[k-j0] if j0 != k/2, else h[j0]
    # Let's just compute it directly
    for j0 in range(n):
        if 0 <= peak_idx - j0 < n:
            if j0 == peak_idx - j0:  # j0 = peak_idx/2
                grad[j0] += peak_sign * h[j0]
            else:
                grad[j0] += peak_sign * h[peak_idx - j0]
    
    return ac, grad


def _epigraph_objective(params, basis, t):
    """Epigraph formulation: minimize t subject to |ac[k]| <= t for all k.
    Using a smooth approximation via log-sum-exp."""
    h = np.dot(params, basis)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    # Smooth max via log-sum-exp
    beta = 100.0
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    soft_max = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return soft_max / (s * s)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _make_basis_functions(n, n_basis=32):
    """Return an (n_basis, n) array of smooth basis functions."""
    x = np.arange(n, dtype=float)
    basis = []
    basis.append(np.ones(n))                       # 0: constant
    for k in range(1, 12):                          # 1-11: cosines
        basis.append(np.cos(2 * np.pi * k * x / n))
    for k in range(1, 12):                          # 12-21: sines
        basis.append(np.sin(2 * np.pi * k * x / n))
    # Gaussian bumps
    for frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = frac * n
        sigma = n / 12.0
        basis.append(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    # Quadratic
    basis.append(((x - n / 2) / (n / 2)) ** 2)
    # Linear ramp
    basis.append(x / n - 0.5)
    # Alternating
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    # Block patterns (4 of them)
    bs = n // 8
    for b in range(4):
        blk = np.zeros(n)
        blk[b * bs:(b + 1) * bs] = 1.0
        blk[(b + 4) * bs:(b + 5) * bs] = -1.0
        basis.append(blk)
    return np.array(basis[:n_basis])


# ---------------------------------------------------------------------------
# Local search routines
# ---------------------------------------------------------------------------

def _local_search_flip(h, max_passes=6, deadline=None):
    """Try flipping each element's sign with incremental autoconv update; keep if it improves the score."""
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(555)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            d = -2.0 * h[i]
            new_ac = ac.copy()
            start = i
            end = i + n
            if end <= size:
                new_ac[start:end] += 2.0 * d * h
            else:
                valid_end = min(end, size)
                if start < valid_end:
                    new_ac[start:valid_end] += 2.0 * d * h[:valid_end - start]
            new_ac[i] += d * d
            new_sum = s_sum + d
            nsq = new_sum * new_sum
            if abs(nsq) < 1e-9:
                continue
            new_val = np.max(np.abs(new_ac)) / nsq
            if new_val < best_val:
                h[i] = -h[i]
                ac = new_ac
                s_sum = new_sum
                best_val = new_val
                improved = True
        if not improved:
            break
    return h, best_val


def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.04):
    """Coordinate-descent style: try small perturbations per element with incremental updates."""
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(777)
    current_step = step_size
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            best_local = old
            best_local_val = best_val
            best_local_ac = ac
            best_local_sum = s_sum
            for delta in (current_step, -current_step, 0.5 * current_step, -0.5 * current_step):
                new_ac = ac.copy()
                start = i
                end = i + n
                if end <= size:
                    new_ac[start:end] += 2.0 * delta * h
                else:
                    valid_end = min(end, size)
                    if start < valid_end:
                        new_ac[start:valid_end] += 2.0 * delta * h[:valid_end - start]
                new_ac[i] += delta * delta
                new_sum = s_sum + delta
                nsq = new_sum * new_sum
                if abs(nsq) < 1e-9:
                    continue
                v = np.max(np.abs(new_ac)) / nsq
                if v < best_local_val:
                    best_local_val = v
                    best_local = old + delta
                    best_local_ac = new_ac
                    best_local_sum = new_sum
            if best_local != old:
                h[i] = best_local
                ac = best_local_ac
                s_sum = best_local_sum
                best_val = best_local_val
                improved = True
        if not improved:
            break
        current_step *= 0.85
    return h, best_val


def _gaussian_perturbation(h, scale=0.06, deadline=None, max_iters=300):
    """Simulated-annealing style search with Gaussian perturbations."""
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    T_start, T_end = 0.4, 0.003
    restart_interval = 60
    no_improve = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, max_iters - 1))
        if rng.random() < 0.06:
            pert = rng.standard_normal(N) * scale * 3.0
        else:
            pert = rng.standard_normal(N) * scale
        h_trial = h_curr + pert
        trial_val = _score(h_trial)
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > restart_interval and it < max_iters - 10:
            h_curr = h_best.copy()
            curr_val = best_val
            no_improve = 0
        scale *= 0.996
    return h_best, best_val


def _incremental_flip_search(h, deadline=None, max_passes=200):
    """
    Fast local search using incremental autoconvolution updates.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            d = -2.0 * h[i]
            new_ac = ac.copy()
            start = i
            end = i + n
            if end <= size:
                new_ac[start:end] += 2.0 * d * h
            else:
                valid_end = min(end, size)
                if start < valid_end:
                    new_ac[start:valid_end] += 2.0 * d * h[:valid_end - start]
            new_ac[i] += d * d
            
            new_sum = s_sum + d
            nsq = new_sum * new_sum
            if abs(nsq) < 1e-9:
                continue
            new_val = np.max(np.abs(new_ac)) / nsq
            
            if new_val < best_val:
                h[i] = -h[i]
                ac = new_ac
                s_sum = new_sum
                best_val = new_val
                improved = True
        if not improved:
            break
    return h, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5 s safety margin

    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score

    n_basis = 32
    basis = _make_basis_functions(N, n_basis)

    # ---- Phase 1: differential evolution on basis coefficients ----
    p1_dl = deadline - 45
    if time.time() < p1_dl:
        bounds = [(-2.0, 2.0)] * n_basis

        def de_obj(params):
            if time.time() > p1_dl:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)

        try:
            res = differential_evolution(
                de_obj, bounds, seed=42, maxiter=40, tol=1e-6,
                popsize=14, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            h_de = np.dot(res.x, basis)
            s_de = np.sum(h_de)
            if abs(s_de) > 1e-6:
                h_de /= s_de
                v = _score(h_de)
                if v < best_score:
                    best_score = v
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Phase 2: smooth-surrogate local optimisation (adaptive beta) ----
    p2_dl = deadline - 22
    if time.time() < p2_dl:
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr /= s_curr
        params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

        rng = np.random.default_rng(123)
        inits = [params.copy()]
        for _ in range(6):
            inits.append(rng.standard_normal(n_basis) * 0.5)

        for p_init in inits[:7]:
            if time.time() > p2_dl - 4:
                break
            try:
                for beta in [60.0, 120.0, 250.0]:
                    if time.time() > p2_dl - 2:
                        break

                    def sobj(params, b=beta):
                        if time.time() > p2_dl - 1:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)

                    res_s = minimize(sobj, p_init, method='BFGS',
                                     options={'maxiter': 200, 'gtol': 1e-8})
                    h_s = np.dot(res_s.x, basis)
                    s_s = np.sum(h_s)
                    if abs(s_s) > 1e-6:
                        h_s /= s_s
                        p_init = np.linalg.lstsq(basis.T, h_s, rcond=None)[0]

                # Final refinement with true objective
                def tobj(params):
                    if time.time() > p2_dl - 0.5:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)

                res_t = minimize(tobj, p_init, method='L-BFGS-B',
                                 options={'maxiter': 300, 'gtol': 1e-10})
                h_t = np.dot(res_t.x, basis)
                s_t = np.sum(h_t)
                if abs(s_t) > 1e-6:
                    h_t /= s_t
                    v = _score(h_t)
                    if v < best_score:
                        best_score = v
                        best_h = h_t.copy()
            except Exception:
                pass

    # ---- Phase 3: peak-suppression via projected gradient descent ----
    p3_dl = deadline - 12
    if time.time() < p3_dl:
        h_ps = best_h.copy()
        n = len(h_ps)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h_ps, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size].copy()
        s_sum = np.sum(h_ps)
        best_ps = np.max(np.abs(ac)) / (s_sum * s_sum)
        rng = np.random.default_rng(888)
        step = 0.02
        for it in range(200):
            if time.time() > p3_dl:
                break
            abs_ac = np.abs(ac)
            peak_val = np.max(abs_ac)
            peak_idx = np.argmax(abs_ac)
            peak_sign = np.sign(ac[peak_idx])
            # Gradient of max|ac| w.r.t. h
            grad = np.zeros(n)
            valid_j = np.arange(n)
            kj = peak_idx - valid_j
            mask = (kj >= 0) & (kj < n)
            grad[valid_j[mask]] += peak_sign * h_ps[kj[mask]]
            # Symmetric contribution
            for j0 in range(n):
                if 0 <= peak_idx - j0 < n:
                    if j0 == peak_idx - j0:
                        grad[j0] += peak_sign * h_ps[j0]
                    else:
                        grad[j0] += peak_sign * h_ps[peak_idx - j0]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad /= grad_norm
            h_trial = h_ps - step * grad
            H_t = np.fft.rfft(h_trial, fsize)
            ac_t = np.fft.irfft(H_t * H_t, fsize)[:size]
            s_t = np.sum(h_trial)
            if abs(s_t) < 1e-9:
                step *= 0.9
                continue
            v_t = np.max(np.abs(ac_t)) / (s_t * s_t)
            if v_t < best_ps:
                h_ps = h_trial
                ac = ac_t
                s_sum = s_t
                best_ps = v_t
                step = min(step * 1.05, 0.05)
            else:
                step *= 0.9
        if best_ps < best_score:
            best_score = best_ps
            best_h = h_ps.copy()

    # ---- Phase 4: incremental flip local search ----
    p4_dl = deadline - 8
    if time.time() < p4_dl:
        h_fl, v_fl = _incremental_flip_search(best_h, deadline=p4_dl, max_passes=200)
        if v_fl < best_score:
            best_score = v_fl
            best_h = h_fl.copy()

    # ---- Phase 5: perturbation local search (now incremental) ----
    p5_dl = deadline - 4
    if time.time() < p5_dl:
        h_pt, v_pt = _local_search_perturb(best_h, max_passes=6,
                                           deadline=p5_dl, step_size=0.03)
        if v_pt < best_score:
            best_score = v_pt
            best_h = h_pt.copy()

    # ---- Phase 6: simulated-annealing polish ----
    p6_dl = deadline - 1.5
    if time.time() < p6_dl:
        h_sa, v_sa = _gaussian_perturbation(best_h, scale=0.04,
                                            deadline=p6_dl, max_iters=300)
        if v_sa < best_score:
            best_score = v_sa
            best_h = h_sa.copy()

    # ---- Phase 7: final high-precision polish ----
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr /= s_curr
            params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_f = minimize(final_obj, params, method='L-BFGS-B',
                             options={'maxiter': 500, 'gtol': 1e-12})
            h_f = np.dot(res_f.x, basis)
            s_f = np.sum(h_f)
            if abs(s_f) > 1e-6:
                h_f /= s_f
                v_f = _score(h_f)
                if v_f < best_score:
                    best_score = v_f
                    best_h = h_f.copy()
        except Exception:
            pass

    # ---- Phase 8: final sign-flip polish ----
    if time.time() < deadline - 0.3:
        try:
            h_ff, v_ff = _local_search_flip(best_h, max_passes=2,
                                            deadline=deadline - 0.1)
            if v_ff < best_score:
                best_score = v_ff
                best_h = h_ff.copy()
        except Exception:
            pass

    # Safety: never worse than baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
