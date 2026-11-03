"""Combined: Signed step heights minimizing max|autoconv| / (sum s)^2.

Strategy:
- Phase 0: Check the theoretical constant sequence candidate.
- Phase 1: Gradient-based optimization using a smooth log-sum-exp surrogate
  (from Inspiration 1) with multiple diverse restarts.
- Phase 2: Fast discrete local search (from Current Program) using O(N)
  incremental updates to refine the continuous solution into a precise one.
- Phase 3: Simulated annealing to escape local minima.
- Phase 4: Final high-precision polish.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve


def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / sq_int


def _soft_max_ac(s, K=128):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp on ac^2."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)


def _soft_objective(x, K=128):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, K) / sq_int


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 s safety margin

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # Phase 0: Check simple optimal candidates (Constant sequence)
    for candidate in [np.ones(N), -np.ones(N)]:
        sc = _score_sequence(candidate)
        if sc < best_score:
            best_score = sc
            best_s = candidate.copy()

    # Phase 1: Gradient-based optimization (Inspiration 1 style)
    n_restarts = 12
    for it in range(n_restarts):
        if _now() > deadline - 40:
            break

        if it == 0:
            # Start exactly at the constant sequence to see if gradient can improve
            x0 = np.ones(N)
        elif it < 4:
            # Near constant
            x0 = np.ones(N) + np.random.randn(N) * 0.05
        elif it < 8:
            # Diverse random
            x0 = np.random.randn(N) * 0.5
        else:
            # Perturb best found
            if best_s is not None:
                x0 = np.arctanh(np.clip(best_s, -0.99, 0.99)) + np.random.randn(N) * 0.1
            else:
                x0 = np.random.randn(N) * 0.5

        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(128,),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
                
            # Also check binarized version
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0
            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
        except Exception:
            continue

    # Phase 2: Discrete Local Search (Current Program style)
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _flip_eval(idx):
            val = cur[idx]
            new_ac = cur_ac.copy()
            start = idx
            end = idx + N
            new_ac[start:end] += -4.0 * val * cur
            new_ac[idx] += 4.0 * val * val
            new_sum = cur_sum - 2.0 * val
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        max_passes = 300
        for _ in range(max_passes):
            if _now() > deadline - 20:
                break
            best_j = -1
            best_js = cur_score
            best_new_ac = None
            best_new_sum = cur_sum

            for j in range(N):
                new_ac, new_sum, new_score = _flip_eval(j)
                if new_score < best_js:
                    best_js = new_score
                    best_j = j
                    best_new_ac = new_ac
                    best_new_sum = new_sum

            if best_j == -1 or best_js >= cur_score - 1e-12:
                break

            cur[best_j] = -cur[best_j]
            cur_ac = best_new_ac
            cur_sum = best_new_sum
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated Annealing
        T_start = 0.03
        sa_steps = 2000
        for it in range(sa_steps):
            if _now() > deadline - 20:
                break
            idx = int(np.random.randint(N))
            new_ac, new_sum, new_score = _flip_eval(idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), 1e-7)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur[idx] = -cur[idx]
                cur_ac, cur_sum, cur_score = new_ac, new_sum, new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
