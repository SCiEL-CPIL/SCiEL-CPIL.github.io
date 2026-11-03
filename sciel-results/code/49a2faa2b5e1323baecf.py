"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Strategy:
- Epigraph formulation with smooth log-sum-exp surrogate for the max constraint
- L-BFGS-B with analytic gradients (much faster than SLSQP for this problem)
- Support selection: try many sparse supports, optimize magnitudes on each
- Block-coordinate refinement and final polish
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1


def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))


def _conv_max(h):
    """Return max of autoconvolution (without DELTA factor)."""
    return np.max(np.convolve(h, h, mode='full'))


def _logsumexp(x, axis=None, beta=1.0):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    # Avoid inf - inf
    x_shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * x_shifted), axis=axis, keepdims=True)) / beta


def _smooth_max(x, beta=10.0):
    """Smooth approximation to max using log-sum-exp."""
    x_max = np.max(x)
    x_shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * x_shifted))) / beta


def _smooth_max_grad(x, beta=10.0):
    """Gradient of smooth max."""
    x_max = np.max(x)
    x_shifted = x - x_max
    exp_x = np.exp(beta * x_shifted)
    return exp_x / np.sum(exp_x)


def _solve_fixed_support(h_init, support, maxiter=500, ftol=1e-12, beta=20.0):
    """
    Optimize heights on a fixed support using L-BFGS-B with log-sum-exp smooth constraint.
    Variables: h_i for i in support (nonneg), t (epigraph upper bound on smooth max of conv).
    Minimize t subject to sum(h)=1, t >= smooth_max(conv(h)).
    """
    k = len(support)
    if k == 0:
        return None

    def objective(x):
        # x = [h_0, ..., h_{k-1}, t]
        return x[k]

    def objective_grad(x):
        grad = np.zeros(k + 1)
        grad[k] = 1.0
        return grad

    def get_h(x):
        h = np.zeros(N)
        h[support] = x[:k]
        return h

    def constraint_fun(x):
        h = get_h(x)
        conv = np.convolve(h, h, mode='full')
        return x[k] - _smooth_max(conv, beta)

    def constraint_grad(x):
        h = get_h(x)
        conv = np.convolve(h, h, mode='full')
        grad_t = 1.0
        grad_conv = _smooth_max_grad(conv, beta)
        # d(conv)/d(h_i) = 2 * conv shifted: conv[j] = sum_i h[i]*h[j-i]
        # d(conv[j])/d(h[i]) = h[j-i] + h[i-j] (with appropriate indexing)
        # Actually conv[j] = sum_{i} h[i] * h[j-i] where h is zero-padded
        # d(conv[j])/d(h[i]) = h[j-i] + h[i-(j-i)]... let me think more carefully.
        # conv = np.convolve(h, h, 'full'), conv[j] = sum_{i=0}^{N-1} h[i]*h[j-i]
        # d(conv[j])/d(h[i]) = h[j-i] + h[i-j] (both terms valid when indices in [0,N-1])
        # But since h is the same in both slots, it's 2*h[j-i] when j-i is valid,
        # except when j-i == i (i.e., j=2i), then it's just h[i]... 
        # Actually no: conv[j] = sum_i h[i] h[j-i]. d/d(h[i]) = h[j-i] + (if j-i != i) h[i-(j-i)]
        # Wait, let me be more careful. conv[j] = sum_{i=0}^{N-1} h[i] * h[j-i]
        # where h[m] = 0 for m outside [0, N-1].
        # d(conv[j])/d(h[i]) = h[j-i] + (contribution from the other term)
        # The term h[i]*h[j-i] gives derivative h[j-i] w.r.t. h[i].
        # The term h[j-i]*h[i] (when the index in the sum is j-i) gives derivative h[i] w.r.t. h[i].
        # So d(conv[j])/d(h[i]) = h[j-i] + h[i] if j-i != i, else 2*h[i] if j-i == i.
        # Wait no. Let me re-derive. conv[j] = sum_{a=0}^{N-1} h[a] * h[j-a].
        # d(conv[j])/d(h[i]) = sum_a [delta(a,i)*h[j-a] + h[a]*delta(j-a,i)]
        # = h[j-i] + h[j-i] if j-i is in [0,N-1] and i is in [0,N-1]... 
        # Actually: = h[j-i] (from a=i term) + h[j-i] (from j-a=i, i.e., a=j-i term)
        # But if j-i == i, these are the same term, so it's just h[i] (not 2*h[i]).
        # Hmm, let me just compute it directly for correctness.
        
        grad = np.zeros(k + 1)
        grad[k] = grad_t
        for idx, pos in enumerate(support):
            # d(conv)/d(h[pos]) = 2 * conv_shifted where conv_shifted[j] = h[j-pos] (with proper bounds)
            # More precisely: d(conv[j])/d(h[pos]) = h[j-pos] + h[pos-(j-pos)] if j-pos != pos, else 2*h[pos]
            # Let's just compute: for each j, d(conv[j])/d(h[pos]) = h[j-pos] + (h[pos - (j-pos)] if pos != j-pos else 0)
            # Wait, that's not right either. Let me just do it numerically-safe.
            # d(conv[j])/d(h[pos]): conv[j] = sum_a h[a]h[j-a]
            # = h[pos]h[j-pos] (a=pos) + h[j-pos]h[pos] (a=j-pos, if j-pos != pos)
            # So if j-pos != pos: d(conv[j])/d(h[pos]) = h[j-pos] + h[j-pos] = 2*h[j-pos]
            # If j-pos == pos (j=2pos): d(conv[j])/d(h[pos]) = h[pos] (only one term)
            
            # Vectorized: d_conv_j = 2*h[j-pos] for all j, except j=2pos where it's h[pos]
            # Actually this is getting complicated. Let me use the formula:
            # d(conv[j])/d(h[pos]) = h[j-pos] + h[pos-j+pos] = h[j-pos] + h[2pos-j]
            # where h[m] = 0 if m not in [0,N-1].
            
            # This is O(N) per pos, and we have k positions, so O(k*N) per gradient.
            # For k up to ~15 and N=50, that's fine.
            
            for j in range(NUM_CONV):
                m1 = j - pos
                m2 = 2 * pos - j
                dval = 0.0
                if 0 <= m1 < N:
                    dval += h[m1]
                if 0 <= m2 < N:
                    dval += h[m2]
                grad[idx] += grad_conv[j] * dval

        return grad

    # Bounds
    bounds = [(0.0, None)] * k + [(0.0, None)]

    # Initial point
    h_init_s = np.zeros(N)
    h_init_s[support] = h_init[support]
    if np.sum(h_init_s) < 1e-15:
        h_init_s[support] = 1.0 / k
    h_init_s = h_init_s / np.sum(h_init_s)

    conv_init = np.convolve(h_init_s, h_init_s, mode='full')
    t_init = _smooth_max(conv_init, beta) + 1e-4

    x0 = np.concatenate((h_init_s[support], [t_init]))

    constraints = [{'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_grad}]

    try:
        res = opt.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            jac=objective_grad,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8}
        )
        h_out = np.zeros(N)
        h_out[support] = np.clip(res.x[:k], 0, None)
        if np.sum(h_out) > 1e-15:
            h_out /= np.sum(h_out)
            return h_out
    except Exception:
        pass
    return None


def _solve_fixed_support_slsqp(h_init, support, maxiter=300, ftol=1e-10):
    """SLSQP version as backup/fallback."""
    k = len(support)
    if k == 0:
        return None

    def objective(x):
        return x[k]

    def objective_grad(x):
        grad = np.zeros(k + 1)
        grad[k] = 1.0
        return grad

    def get_h(x):
        h = np.zeros(N)
        h[support] = x[:k]
        return h

    def constraint_eq_fun(x):
        return np.sum(x[:k]) - 1.0

    def constraint_eq_jac(x):
        jac = np.zeros(k + 1)
        jac[:k] = 1.0
        return jac

    def constraint_ineq_fun(x):
        h = get_h(x)
        conv = np.convolve(h, h, mode='full')
        return x[k] - conv

    def constraint_ineq_jac(x):
        h = get_h(x)
        jac = np.zeros((NUM_CONV, k + 1))
        jac[:, k] = 1.0
        for idx, pos in enumerate(support):
            for j in range(NUM_CONV):
                m1 = j - pos
                m2 = 2 * pos - j
                dval = 0.0
                if 0 <= m1 < N:
                    dval += h[m1]
                if 0 <= m2 < N:
                    dval += h[m2]
                jac[j, idx] = -dval
        return jac

    constraints = [
        {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
        {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
    ]

    bounds = [(0.0, None)] * k + [(0.0, None)]

    h_init_s = np.zeros(N)
    h_init_s[support] = h_init[support]
    if np.sum(h_init_s) < 1e-15:
        h_init_s[support] = 1.0 / k
    h_init_s = h_init_s / np.sum(h_init_s)

    conv_init = np.convolve(h_init_s, h_init_s, mode='full')
    t_init = np.max(conv_init) + 1e-4

    x0 = np.concatenate((h_init_s[support], [t_init]))

    try:
        res = opt.minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            jac=objective_grad, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        h_out = np.zeros(N)
        h_out[support] = np.clip(res.x[:k], 0, None)
        if np.sum(h_out) > 1e-15:
            h_out /= np.sum(h_out)
            return h_out
    except Exception:
        pass
    return None


def _block_coordinate_refine(h_start, max_iter=40, deadline=None):
    """Refine by moving mass between pairs symmetric around the dominant peak."""
    h = h_start.copy()
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-15:
        return h
    h /= np.sum(h)

    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iter_count += 1

        conv = np.convolve(h, h, mode='full')
        k_star = np.argmax(conv)

        for i in range(N):
            j = k_star - i
            if j < 0 or j >= N or j == i:
                continue

            hi = h[i]
            hj = h[j]
            if hi + hj < 1e-15:
                continue

            best_t = 0.0
            best_local_score = _score_raw(h)

            # Try transferring mass from i to j and vice versa
            for t in np.linspace(-hi * 0.9, hj * 0.9, 20):
                h_test = h.copy()
                h_test[i] += t
                h_test[j] -= t
                if np.any(h_test < -1e-15):
                    continue
                h_test = np.clip(h_test, 0, None)
                if np.sum(h_test) < 1e-15:
                    continue
                h_test /= np.sum(h_test)
                s_test = _score_raw(h_test)
                if s_test > best_local_score + 1e-14:
                    best_local_score = s_test
                    best_t = t

            if best_t != 0.0:
                h[i] += best_t
                h[j] -= best_t
                h = np.clip(h, 0, None)
                h /= np.sum(h)
                improved = True

    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    # --- Generate diverse supports to try ---
    np.random.seed(42)
    supports = []

    # Full support
    supports.append(list(range(N)))

    # Sparse supports: random
    for trial in range(60):
        k = np.random.randint(3, 16)
        idx = np.random.choice(N, k, replace=False)
        supports.append(list(idx))

    # Structured sparse: clustered
    for _ in range(40):
        k = np.random.randint(3, 12)
        center = np.random.randint(5, N - 5)
        spread = np.random.randint(2, 8)
        idx = set()
        for _ in range(k * 3):
            pos = center + np.random.randint(-spread, spread + 1)
            if 0 <= pos < N:
                idx.add(pos)
            if len(idx) >= k:
                break
        if len(idx) >= 2:
            supports.append(list(idx))

    # Symmetric pairs around center
    for k in range(2, 10):
        for offset in range(1, 10):
            idx = set()
            center = N // 2
            for i in range(k):
                p1 = center - offset - i
                p2 = center + offset + i
                if 0 <= p1 < N:
                    idx.add(p1)
                if 0 <= p2 < N:
                    idx.add(p2)
            if len(idx) >= 2:
                supports.append(list(idx))

    # Alternating
    for start in range(N):
        for gap in [2, 3, 4, 5, 7]:
            idx = list(range(start, N, gap))
            if len(idx) >= 3:
                supports.append(idx)

    # Gaussian-like support (top-k of a gaussian)
    for c in np.linspace(5, N - 5, 8):
        for w in np.linspace(2, N / 3, 5):
            x_idx = np.arange(N)
            weights = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            k = np.random.randint(5, 15)
            top_idx = np.argsort(weights)[-k:]
            supports.append(list(top_idx))

    # Deduplicate supports (by sorted tuple)
    seen = set()
    unique_supports = []
    for s in supports:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
    supports = unique_supports

    # Sort supports by size (smaller first - faster to optimize)
    supports.sort(key=len)

    # --- Phase 1: Optimize on each support ---
    # Use L-BFGS-B with log-sum-exp for speed
    candidates = []

    for support in supports:
        if time.time() > deadline - 30:
            break

        k = len(support)
        # Initial: uniform on support
        h_init = np.zeros(N)
        h_init[support] = 1.0 / k

        # Try L-BFGS-B with log-sum-exp
        if k <= 20:  # Limit for speed
            h_cand = _solve_fixed_support(h_init, support, maxiter=300, ftol=1e-10, beta=15.0)
            if h_cand is not None:
                s_cand = _score_raw(h_cand)
                candidates.append((s_cand, h_cand))
                if s_cand > best_score:
                    best_score = s_cand
                    best_h = h_cand.copy()

        # Also try SLSQP for smaller supports
        if k <= 12 and time.time() < deadline - 20:
            h_cand2 = _solve_fixed_support_slsqp(h_init, support, maxiter=200, ftol=1e-10)
            if h_cand2 is not None:
                s_cand2 = _score_raw(h_cand2)
                candidates.append((s_cand2, h_cand2))
                if s_cand2 > best_score:
                    best_score = s_cand2
                    best_h = h_cand2.copy()

    # Sort candidates by score
    candidates.sort(key=lambda x: -x[0])

    # --- Phase 2: Refine top candidates with block-coordinate ---
    top_candidates = candidates[:20]
    for s_cand, h_cand in top_candidates:
        if time.time() > deadline - 15:
            break
        h_refined = _block_coordinate_refine(h_cand, max_iter=30, deadline=deadline - 10)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    # --- Phase 3: Refine best_h further ---
    if time.time() < deadline - 10:
        h_refined2 = _block_coordinate_refine(best_h, max_iter=50, deadline=deadline - 5)
        s_refined2 = _score_raw(h_refined2)
        if s_refined2 > best_score:
            best_score = s_refined2
            best_h = h_refined2.copy()

    # --- Phase 4: Final polish with tighter tolerances on best support ---
    if time.time() < deadline - 5:
        # Find support of best_h
        support_best = list(np.where(best_h > 1e-8)[0])
        if len(support_best) >= 2:
            h_polished = _solve_fixed_support(best_h, support_best, maxiter=500, ftol=1e-12, beta=50.0)
            if h_polished is not None:
                s_polished = _score_raw(h_polished)
                if s_polished > best_score:
                    best_score = s_polished
                    best_h = h_polished.copy()

            # SLSQP polish
            if time.time() < deadline - 3:
                h_polished2 = _solve_fixed_support_slsqp(best_h, support_best, maxiter=400, ftol=1e-12)
                if h_polished2 is not None:
                    s_polished2 = _score_raw(h_polished2)
                    if s_polished2 > best_score:
                        best_score = s_polished2
                        best_h = h_polished2.copy()

    # --- Phase 5: Final block-coordinate pass ---
    if time.time() < deadline - 2:
        h_final = _block_coordinate_refine(best_h, max_iter=30, deadline=deadline - 1)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
