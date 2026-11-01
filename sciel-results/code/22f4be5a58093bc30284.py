import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Fixed global deadline (evaluator kills at 150s; keep a safety margin).
_T0 = time.time()
_DEADLINE = _T0 + 142.0


def _remaining():
    return _DEADLINE - time.time()


def _autocorr(h):
    """Linear aperiodic autocorrelation via FFT, length 2N."""
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)


def _norm_bound(ac, h):
    """Normalized C3 upper bound: max|sidelobe| / ac[0], with ac[0]=(sum h)^2."""
    peak = ac[0]
    if peak <= 1e-12:
        peak = 1e-12
    return float(np.max(np.abs(ac[1:N])) / peak)


def _sidelobe_abs(h):
    ac = _autocorr(h)
    return np.abs(ac[1:N]), ac[0]


def _objective_and_gradient(x, beta=12.0):
    """
    Smooth surrogate for the *normalized* max-sidelobe objective.

    We minimize  f = (max_k |ac_k|) / (sum x)^2  over the continuous relaxation x.
    Using a log-sum-exp soft-max for the numerator and a quadratic denominator
    gives a differentiable surrogate. The sign step is applied downstream via a
    straight-through estimator; here we optimize the continuous vector directly.
    """
    ac = _autocorr(x)
    side = np.abs(ac[1:N])
    S = x.sum()
    # Guard the denominator.
    if abs(S) < 1e-3:
        S = np.sign(S) * 1e-3 if S != 0 else 1e-3
    denom = S * S

    max_val = np.max(side)
    if max_val < 1e-12:
        return max_val / denom, np.zeros(N)

    shifted = beta * (side - max_val)
    exp_shifted = np.exp(shifted)
    Z = np.sum(exp_shifted)
    soft_num = max_val + np.log(Z) / beta
    obj = soft_num / denom

    # weights over sidelobes
    weights = exp_shifted / Z
    signs = np.sign(ac[1:N])

    # d soft_num / d x_i  =  sum_k weights_k * sign(ac_k) * d ac_k / d x_i
    # d ac_k / d x_i = x[i+k] + x[i-k]  (valid indices)
    num_grad = np.zeros(N)
    for k in range(1, N):
        w = weights[k - 1]
        if w < 1e-14:
            continue
        c = w * signs[k - 1]
        num_grad[0:N - k] += c * x[k:N]
        num_grad[k:N] += c * x[0:N - k]

    # d obj / d x_i = num_grad_i / denom  -  soft_num * (2 S / denom^2)
    obj_grad = num_grad / denom - (soft_num * (2.0 * S) / (denom * denom)) * np.ones(N)

    return obj, obj_grad


def _straight_through_sign(x):
    """Binary sign with a straight-through gradient (used conceptually)."""
    return np.where(x >= 0, 1.0, -1.0)


def _local_search(h, time_limit=5.0):
    """Discrete hill-climbing / annealing on sign flips of the normalized bound."""
    start = time.time()
    ac = _autocorr(h)
    cur = _norm_bound(ac, h)
    T0 = max(cur * 0.05, 1e-4)
    cool = 0.9995
    it = 0
    while time.time() - start < time_limit and it < 200000:
        it += 1
        i = np.random.randint(N)
        old = h[i]
        h[i] = -old
        ac_new = _autocorr(h)
        new = _norm_bound(ac_new, h)
        delta = new - cur
        if delta < 0:
            cur = new
            ac = ac_new
        else:
            T = T0 * (cool ** it)
            if T > 1e-16 and np.random.random() < np.exp(-delta / T):
                cur = new
                ac = ac_new
            else:
                h[i] = old
    return h, cur


def _structured_init(kind):
    if kind == 0:
        return np.random.choice([-1.0, 1.0], size=N)
    if kind == 1:
        # Alternating blocks to encourage large sum / low sidelobes.
        blk = 8
        b = np.tile(np.random.choice([-1.0, 1.0], size=N // blk + 1), N)
        return b[:N]
    if kind == 2:
        # Walsh-like: low-frequency dominant (flat high-freq).
        freq = np.linspace(0, 1, N)
        return np.sign(np.sin(2 * np.pi * freq * 1.0) + 0.01 * np.random.randn(N))
    # kind 3: mostly +1 with sparse -1 to maximize sum.
        h = np.ones(N)
        m = np.random.randint(0, N, size=int(0.25 * N))
        h[m] = -1.0
        return h


def construct():
    best_h = None
    best_val = float('inf')

    # Phase 1: continuous relaxation on the normalized soft-max objective.
    n_starts = 10
    for i in range(n_starts):
        if _remaining() < 25.0:
            break
        x0 = _structured_init(i % 4) * 0.9
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * N,
                           options={'maxiter': 120, 'ftol': 1e-7, 'gtol': 1e-7, 'maxls': 20})
        except Exception:
            continue
        # Straight-through: snap to binary signs.
        h_bin = _straight_through_sign(res.x)
        ac = _autocorr(h_bin)
        val = _norm_bound(ac, h_bin)
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()

        # Quick discrete polish on the current best.
        if best_h is not None and _remaining() > 20.0:
            h_ref, val_ref = _local_search(best_h.copy(), time_limit=min(4.0, _remaining() - 18.0))
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref

    # Phase 2: final high-precision discrete polish.
    if best_h is not None and _remaining() > 6.0:
        h_final, val_final = _local_search(best_h.copy(), time_limit=_remaining() - 4.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    if best_h is None:
        return np.ones(N)
    return best_h
