"""Optimized program: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s (evaluator kills at 150s, but we want buffer)


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si == 0:
        return np.inf
    return _autoconv_max_abs(h) / si


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0/1 values)."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    # Efficient generation using iterative doubling
    seq = np.array([0])
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def _smooth_surrogate_refine(h, max_iter=200, time_budget=None):
    """
    Smooth-surrogate refinement: minimize a log-sum-exp (soft-max) surrogate
    of max|autoconv| / (sum h)^2 using a projected subgradient step.

    The surrogate is differentiable, so we can move in the true descent
    direction (unlike pure sign-flip search) and then project back onto the
    sign domain.  This is a hypothesis for escaping the local minima that the
    discrete sign-flip search stalls in.
    """
    h = h.copy().astype(float)
    best_h = h.copy()
    best_score = _score(best_h)

    def lse_and_weights(x):
        ac = fftconvolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        m = float(np.max(abs_ac))
        if m == 0:
            return 0.0, np.zeros_like(ac), ac
        w = np.exp(abs_ac - m)
        sw = float(np.sum(w))
        lse = m + np.log(sw)
        return lse, w / sw, ac

    step = 0.05
    for it in range(max_iter):
        if time_budget is not None and time.time() - _START_TIME > time_budget:
            break
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break

        lse, w, ac = lse_and_weights(h)
        s = float(np.sum(h))
        if abs(s) < 1e-9:
            break

        # Subgradient of LSE(|ac|) w.r.t. x_i:
        #   sum_k w_k * sgn(ac_k) * (2*h_{k - i})   (ac_k = sum_j h_j h_{j+k})
        grad = np.zeros(len(h))
        for k in range(len(ac)):
            wk = w[k]
            if wk < 1e-12:
                continue
            sgn = 1.0 if ac[k] >= 0 else -1.0
            # indices j with 0 <= j < n and 0 <= (k - j) < n
            j = np.arange(max(0, k - len(h) + 1), min(len(h), k + 1))
            o = k - j
            mask = (o >= 0) & (o < len(h))
            if not np.any(mask):
                continue
            j = j[mask]
            o = o[mask]
            grad[j] += wk * sgn * 2.0 * h[o]

        # Gradient of 1/(s^2) term: d/dx_i [LSE/s^2] = LSE * d(1/s^2)/dx_i
        # d(1/s^2)/dx_i = -2/s^3
        grad = grad + lse * (-2.0 / (s ** 3))

        # Projected subgradient step, then re-project onto sign domain
        cand = h - step * grad
        cand = np.sign(cand)
        cand[cand == 0] = 1.0
        sc = _score(cand)
        if np.isfinite(sc) and sc < best_score:
            best_score = sc
            best_h = cand.copy()
            h = cand.copy()
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.7
            if step < 1e-4:
                break

    return best_h, best_score


def _square_wave_search(time_budget=None):
    """
    Structural two-block square-wave family (per directive).

    A sequence with one contiguous positive block (height +1, width a) and one
    contiguous negative block (height -1, width b), the rest zero, has a
    *closed-form* autoconvolution (a trapezoid) whose peak is exactly max(a, b).
    The objective is then max(a,b) / (a - b)^2, which we can evaluate in O(1)
    and sweep over (a, b) at fine resolution -- a search the discrete sign-flip
    loop cannot reach because it changes the continuous block widths.
    """
    best_h = None
    best_score = np.inf

    # Coarse-to-fine grid over (a, b) with a + b <= N and |a - b| >= 1
    for a in range(1, N):
        if time_budget is not None and time.time() - _START_TIME > time_budget:
            break
        # b in [1, N - a]; skip b == a (zero integral)
        for b in range(1, N - a + 1):
            if b == a:
                continue
            # Closed-form objective for the two-block square wave
            peak = float(max(a, b))
            net = a - b
            sc = peak / (net * net)
            if sc < best_score:
                best_score = sc
                # Build the sequence: a positive steps then b negative steps
                h = np.zeros(N)
                h[:a] = 1.0
                h[a:a + b] = -1.0
                best_h = h.copy()

    return best_h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    # Strategy: Thue-Morse based deterministic sequence with local optimization

    np.random.seed(42)

    best_h = None
    best_score = np.inf

    # Generate Thue-Morse sequence and convert to ±1
    tm = _thue_morse(N)
    h_tm = 1.0 - 2.0 * tm  # Convert 0/1 to ±1

    # Try multiple Thue-Morse variants (shifts, sign flips)
    candidates = []

    # Original Thue-Morse
    candidates.append(h_tm.copy())

    # Negated Thue-Morse
    candidates.append(-h_tm.copy())

    # Shifted Thue-Morse (circular shift)
    for shift in [1, 2, 3]:
        candidates.append(np.roll(h_tm, shift).copy())

    # Also try a few structured patterns for comparison
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    candidates.append(h_alt.copy())

    # Block pattern +1+1-1-1
    h_block = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    candidates.append(h_block.copy())

    # Random sparse 50%
    mask = np.random.rand(N) < 0.5
    h_rand = np.zeros(N)
    h_rand[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand.copy())

    # Random sparse 30%
    mask = np.random.rand(N) < 0.3
    h_rand2 = np.zeros(N)
    h_rand2[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand2.copy())

    # NEW: structural two-block square-wave candidate (closed-form family)
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        h_sq, sc_sq = _square_wave_search(time_budget=_TIME_LIMIT - 30)
        if h_sq is not None:
            candidates.append(h_sq)

    # Evaluate and optimize each candidate
    for trial, h in enumerate(candidates):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break

        # Quick score before optimization
        sc = _score(h)

        # Local sign flip optimization
        h_opt, sc_opt = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 10)

        if sc_opt < best_score:
            best_score = sc_opt
            best_h = h_opt.copy()

    # Smooth surrogate (log-sum-exp) refinement on the best candidate
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 12:
        best_h, best_score = _smooth_surrogate_refine(
            best_h, max_iter=100, time_budget=_TIME_LIMIT - 8
        )

    # Final local polish
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=50, time_budget=_TIME_LIMIT - 2
        )

    if best_h is None:
        best_h = np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
