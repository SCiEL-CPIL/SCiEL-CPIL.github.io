"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines Nesterov accelerated gradient descent on a smooth-max surrogate
with adaptive temperature, backtracking line search, projection onto the
constraint set, an epigraph formulation, and multi-stage coordinate
descent polish. Incorporates adaptive learning rate scheduling, more
structured initializations, and a high-beta final polish pass.
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


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)

    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp

    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)

    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp

    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)

    return val, grad_h, grad_t


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with adaptive learning rate and backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t = t_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _nesterov_epigraph(h0, beta, n_iter=500, lr=0.03):
    """
    Nesterov accelerated gradient descent using epigraph formulation.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    t_val = _score(h)

    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_mom_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_mom_new

        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)

        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)

        step_lr = lr_current
        h_new = None
        t_new_val = None
        for bt in range(max_backtracks):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)

            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                t_new_val = t_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y_h - lr_current * 0.5 * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr_current * 0.5 * grad_t)
            lr_current *= 0.5

        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _high_beta_polish(h0, n_iter=100, beta=200.0, lr=0.005):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    Uses plain projected gradient descent for stability at high beta.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_obj_grad(h, beta)

        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
            h = h_new
        else:
            h_new = h - lr * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            current_score = _score(h_new)
            if current_score < best_score:
                best_score = current_score
                best_h = h_new.copy()
                h = h_new
            else:
                break

    return best_h, best_score


def _coordinate_descent_polish(h0, max_outer=50, max_inner=N):
    """
    Coordinate descent polish: adjust one element at a time with
    local search around current value. Uses adaptive step sizes.
    """
    h = np.array(h0, dtype=np.float64).copy()
    h = _project(h, N / 2.0)
    best_score = _score(h)
    best_h = h.copy()

    best_step = np.full(N, 0.02)

    for outer in range(max_outer):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        for i in range(max_inner):
            if time.time() > _DEADLINE - 2:
                break
            old_val = h[i]
            step = best_step[i]

            candidates = set()
            for d in [step, step*0.5, step*0.25, step*0.125, step*0.0625, step*0.03125]:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            for d in [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            candidates.add(0.0)
            candidates.add(1.0)

            best_candidate_val = None
            best_candidate_score = best_score - 1e-12
            best_candidate_h = None

            for new_val in sorted(candidates):
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                if abs(new_val - old_val) < 1e-15:
                    continue
                h_test = h.copy()
                h_test[i] = new_val
                cur_sum = np.sum(h_test)
                if abs(cur_sum - N / 2.0) > 1e-10:
                    h_test = _project(h_test, N / 2.0)
                else:
                    h_test = np.clip(h_test, 0.0, 1.0)
                s_test = _score(h_test)
                if s_test < best_candidate_score:
                    best_candidate_score = s_test
                    best_candidate_val = new_val
                    best_candidate_h = h_test.copy()
                    move = abs(new_val - old_val)
                    if move > 0:
                        best_step[i] = max(0.00005, move * 0.5)

            if best_candidate_val is not None and best_candidate_h is not None:
                h = best_candidate_h
                best_score = best_candidate_score
                best_h = h.copy()
                improved = True

        if not improved:
            break

    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')

    np.random.seed(42)

    # Initializations: diverse starting points
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N),
        0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.random.randn(N),
        0.5 + 0.2 * np.sin(6 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.cos(3 * np.pi * np.arange(N) / N),
        0.5 + 0.1 * np.sin(8 * np.pi * np.arange(N) / N) + 0.1 * np.cos(5 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.cos(2 * np.pi * 3 * np.arange(N) / N),
        0.5 + 0.15 * np.sin(2 * np.pi * 5 * np.arange(N) / N),
    ]

    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Add random inits
    for _ in range(8):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Phase 1: Nesterov with adaptive beta (standard smooth max)
    beta = 8.0
    n_restarts = 5

    for r in range(n_restarts):
        if time.time() > _DEADLINE - 35:
            break

        for h_init in inits:
            if time.time() > _DEADLINE - 25:
                break

            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=350, lr=0.05)

            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

        beta *= 1.35

        # Also try a lower learning rate for finer convergence on top inits
        for h_init in inits[:4]:
            if time.time() > _DEADLINE - 25:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=250, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: Epigraph Nesterov with the best solution so far
    if best_h is not None and time.time() < _DEADLINE - 20:
        beta_ep = 12.0
        for r in range(3):
            if time.time() > _DEADLINE - 12:
                break
            start_points = [best_h]
            if len(inits) > 0:
                start_points.append(inits[0])
            if len(inits) > 1:
                start_points.append(inits[1])
            if len(inits) > 2:
                start_points.append(inits[2])

            for sp in start_points:
                if time.time() > _DEADLINE - 10:
                    break
                h_opt, s_opt = _nesterov_epigraph(sp, beta_ep, n_iter=250, lr=0.03)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            beta_ep *= 1.25

    # Phase 3: High-beta polish
    if best_h is not None and time.time() < _DEADLINE - 12:
        h_pol, s_pol = _high_beta_polish(best_h, n_iter=80, beta=150.0, lr=0.005)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()

    # Phase 4: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_pol2, s_pol2 = _coordinate_descent_polish(best_h, max_outer=50, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()

    # Phase 5: Final fine-grained coordinate descent with tighter tolerance
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_pol3, s_pol3 = _coordinate_descent_polish(best_h, max_outer=40, max_inner=N)
        if s_pol3 < best_score:
            best_score = s_pol3
            best_h = h_pol3.copy()

    # Phase 6: Ultra-fine polish - very small steps, many passes
    if best_h is not None and time.time() < _DEADLINE - 1.5:
        h_final = np.array(best_h, dtype=np.float64).copy()
        h_final = _project(h_final, N / 2.0)
        s_final = _score(h_final)
        for _ in range(20):
            if time.time() > _DEADLINE - 1.0:
                break
            any_improved = False
            for i in range(N):
                if time.time() > _DEADLINE - 1.0:
                    break
                old_val = h_final[i]
                for d in [0.0001, 0.00005, 0.00002, 0.00001]:
                    for sign in [1.0, -1.0]:
                        new_val = old_val + sign * d
                        if new_val < -1e-12 or new_val > 1 + 1e-12:
                            continue
                        new_val = np.clip(new_val, 0.0, 1.0)
                        h_test = h_final.copy()
                        h_test[i] = new_val
                        cur_sum = np.sum(h_test)
                        if abs(cur_sum - N / 2.0) > 1e-10:
                            h_test = _project(h_test, N / 2.0)
                        s_test = _score(h_test)
                        if s_test < s_final - 1e-12:
                            h_final = h_test
                            s_final = s_test
                            any_improved = True
                            break
                    if any_improved:
                        break
                if any_improved:
                    break
            if not any_improved:
                break
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
