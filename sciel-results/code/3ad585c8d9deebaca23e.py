"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
STRUCTURAL CHANGE: Fourier-domain power-spectrum optimization with
real-space box-constraint enforcement via iterative projection,
combined with stochastic perturbation (simulated annealing) for
landscape exploration, and smooth-max gradient refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 142


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


def _nesterov_accelerated_gradient(h0, beta, n_iter=400, lr=0.05):
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


def _fourier_power_spectrum_optimize(h0, n_iter=200, lr=0.01):
    """
    STRUCTURAL: Optimize directly in Fourier power-spectrum space.
    The overlap values C_k are determined by |H_k|^2 via inverse FFT.
    We parameterize by the power spectrum P_k = |H_k|^2 and use
    gradient descent on P with real-space feasibility enforcement.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    # Initial power spectrum
    H_full = np.fft.rfft(h, n=2*N)
    P = np.abs(H_full)**2  # power spectrum

    best_h = h.copy()
    best_score = _score(h)

    n_fft = 2 * N
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        # Reconstruct h from power spectrum (use current h's phases)
        H_current = np.fft.rfft(h, n=n_fft)
        # Update power spectrum with gradient
        cs = _linear_overlap_values(h)
        cs_max = np.max(cs)
        beta = 50.0
        exp_cs = np.exp(beta * (cs - cs_max))
        sum_exp = np.sum(exp_cs)
        w = exp_cs / sum_exp

        # Gradient of C_k w.r.t. h: grad C_k[i] = -(2/N) * (h[i+k] + h[i-k]) (circular)
        # In Fourier space, this is simpler: C_k depends on |H|^2
        # dC_k/dP_m involves the structure of the autocorrelation

        # Use direct gradient in h-space, then map to P-space
        val, grad_h = _smooth_max_obj_grad(h, beta)

        # Map gradient to power spectrum space
        # P = |H|^2, dP = 2*Re(conj(H) * dH)
        # dH = IDFT(grad_h) (approximately, for small steps)
        dH = np.fft.rfft(grad_h, n=n_fft)
        dP = 2.0 * np.real(np.conj(H_current) * dH)

        # Gradient descent on P (keep P >= 0)
        P_new = P - lr * dP
        P_new = np.maximum(P_new, 0.0)

        # Reconstruct H from P (keep phases from current H)
        phases = H_current / (np.abs(H_current) + 1e-15)
        H_new = np.sqrt(P_new) * phases
        # Ensure H_0 is real and = N/2 (DC constraint)
        H_new[0] = N / 2.0

        h_new = np.fft.irfft(H_new, n=n_fft)[:N]
        h_new = _project(h_new, N / 2.0)

        P = np.abs(np.fft.rfft(h_new, n=n_fft))**2

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()

    return best_h, best_score


def _stochastic_perturbation(h0, n_iter=500, temp_init=0.05, temp_decay=0.995):
    """
    STRUCTURAL: Simulated annealing / stochastic perturbation.
    Randomly perturb individual elements and accept/reject based on
    score improvement, with temperature annealing. This explores
    the landscape in a fundamentally different way than gradient methods.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score
    temp = temp_init

    rng = np.random.RandomState(123)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        # Pick a random element to perturb
        idx = rng.randint(N)
        # Perturbation size scales with temperature
        delta = rng.uniform(-temp, temp)
        h_test = h.copy()
        h_test[idx] = np.clip(h_test[idx] + delta, 0.0, 1.0)
        h_test = _project(h_test, N / 2.0)

        test_score = _score(h_test)

        # Accept if better, or with probability exp(-(test_score - current_score)/temp)
        if test_score < current_score:
            h = h_test
            current_score = test_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Metropolis criterion
            delta_s = test_score - current_score
            if delta_s < 0.01 * temp:  # Only consider small worsening
                accept_prob = np.exp(-delta_s / (temp + 1e-15))
                if rng.random() < accept_prob:
                    h = h_test
                    current_score = test_score

        temp *= temp_decay

    return best_h, best_score


def _coordinate_descent_polish(h0, max_outer=50, max_inner=N):
    """
    Coordinate descent polish: adjust one element at a time with
    local search around current value.
    """
    h = np.array(h0, dtype=np.float64).copy()
    best_score = _score(h)
    best_h = h.copy()

    best_step = np.full(N, 0.01)

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
            for d in [step, step/2, step/4, step/8, step/16, step/32]:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            for d in [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            candidates.add(0.0)
            candidates.add(1.0)

            best_candidate_val = None
            best_candidate_score = best_score - 1e-12

            for new_val in sorted(candidates):
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                h_test = h.copy()
                h_test[i] = new_val
                h_test = _project(h_test, N / 2.0)
                s_test = _score(h_test)
                if s_test < best_candidate_score:
                    best_candidate_score = s_test
                    best_candidate_val = new_val
                    move = abs(new_val - old_val)
                    if move > 0:
                        best_step[i] = max(0.00005, move * 0.5)

            if best_candidate_val is not None:
                h[i] = best_candidate_val
                h = _project(h, N / 2.0)
                best_score = best_candidate_score
                best_h = h.copy()
                improved = True

        if not improved:
            break

    return best_h, best_score


def _scipy_refine(h0, beta=20.0, max_iter=200):
    """Use scipy minimize with SLSQP for refinement."""
    h = np.array(h0, dtype=np.float64)

    def objective(x):
        hx = _project(x, N / 2.0)
        val, _ = _smooth_max_obj_grad(hx, beta)
        return val

    def gradient(x):
        hx = _project(x, N / 2.0)
        _, grad = _smooth_max_obj_grad(hx, beta)
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N

    try:
        result = scipy_minimize(
            objective, h, method='SLSQP', jac=gradient,
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        h_opt = _project(result.x, N / 2.0)
        s_opt = _score(h_opt)
        return h_opt, s_opt
    except:
        return h0, _score(h0)


def _high_beta_polish(h0, n_iter=300, beta=500.0, lr=0.001):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)
    current_lr = lr
    no_improve_count = 0

    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_obj_grad(h, beta)

        h_new = h - current_lr * grad
        h_new = _project(h_new, N / 2.0)

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                current_lr *= 0.8
                no_improve_count = 0

        h = h_new

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
        0.5 + 0.25 * np.cos(2 * np.pi * 2 * np.arange(N) / N) + 0.1 * np.cos(2 * np.pi * 5 * np.arange(N) / N),
        0.5 + 0.2 * np.sin(2 * np.pi * 3 * np.arange(N) / N) + 0.15 * np.sin(2 * np.pi * 7 * np.arange(N) / N),
    ]

    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    for _ in range(8):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Phase 1: Nesterov smooth-max with adaptive beta
    beta = 10.0
    n_restarts = 5

    for r in range(n_restarts):
        if time.time() > _DEADLINE - 55:
            break

        for h_init in inits:
            if time.time() > _DEADLINE - 45:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=300, lr=0.05)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

        beta *= 1.4

        for h_init in inits[:3]:
            if time.time() > _DEADLINE - 45:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=200, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: STRUCTURAL - Fourier power spectrum optimization
    if best_h is not None and time.time() < _DEADLINE - 35:
        h_opt, s_opt = _fourier_power_spectrum_optimize(best_h, n_iter=200, lr=0.01)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Phase 3: STRUCTURAL - Stochastic perturbation (simulated annealing)
    if best_h is not None and time.time() < _DEADLINE - 25:
        h_opt, s_opt = _stochastic_perturbation(best_h, n_iter=800, temp_init=0.03, temp_decay=0.998)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Phase 4: Nesterov after stochastic exploration
    if best_h is not None and time.time() < _DEADLINE - 18:
        for beta_n in [20.0, 40.0]:
            if time.time() > _DEADLINE - 14:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(best_h, beta_n, n_iter=200, lr=0.03)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 5: Scipy SLSQP refinement
    if best_h is not None and time.time() < _DEADLINE - 12:
        for beta_s in [15.0, 30.0]:
            if time.time() > _DEADLINE - 8:
                break
            h_opt, s_opt = _scipy_refine(best_h, beta=beta_s, max_iter=100)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 6: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 6:
        h_pol, s_pol = _coordinate_descent_polish(best_h, max_outer=40, max_inner=N)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()

    # Phase 7: Another stochastic pass at lower temperature
    if best_h is not None and time.time() < _DEADLINE - 4:
        h_opt, s_opt = _stochastic_perturbation(best_h, n_iter=500, temp_init=0.01, temp_decay=0.999)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Phase 8: High-beta projected-gradient polish
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_pol3, s_pol3 = _high_beta_polish(best_h, n_iter=200, beta=500.0, lr=0.001)
        if s_pol3 < best_score:
            best_score = s_pol3
            best_h = h_pol3.copy()

    # Phase 9: Final coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 1:
        h_pol2, s_pol2 = _coordinate_descent_polish(best_h, max_outer=20, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
