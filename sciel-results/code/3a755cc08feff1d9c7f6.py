# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses linear programming (with sparse constraints) to find the optimal f under the
    constraint that M(x) = sum_k f(k) * floor(x/k) <= 1 for all x in [1, 10K],
    after the evaluator's normalization f(1) -= sum_k f(k)/k.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    t0 = time.time()
    DEADLINE = 600.0  # 630s kill, leave 30s margin

    def _elapsed():
        return time.time() - t0

    # Fallback: hardcoded baseline
    def _baseline():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Calculate score (objective a) for a given result
    def _calc_score(domain, values):
        # Clip
        v_clip = np.clip(values, -10, 10)
        d_float = domain.astype(np.float64)
        # S = sum f(k)/k
        S = np.sum(v_clip / d_float)
        # f(1) adjustment
        v_norm = v_clip.copy()
        if len(domain) > 0 and domain[0] == 1:
            v_norm[0] -= S
        # a = -sum f(k) log(k) / k
        a = -np.sum(v_norm * np.log(d_float) / d_float)
        return float(a)

    best_result = _baseline()
    best_a = _calc_score(best_result[0], best_result[1])

    # Try increasing domain sizes, keeping N small for speed
    # We try N = 50, 100, 150, 200, 300, 500 but break if time runs out
    for N in [50, 100, 150, 200, 300, 500]:
        if _elapsed() > DEADLINE - 120:
            break

        try:
            result, a_val = _solve_lp(N)
            if result is not None:
                # Verify feasibility
                if _verify_feasible(result[0], result[1]):
                    if a_val > best_a + 1e-9:
                        best_result = result
                        best_a = a_val
        except Exception:
            continue

    return best_result


def _solve_lp(N):
    """
    Solve the LP for domain {1, 2, ..., N}.
    Variables: f(2), f(3), ..., f(N)  (total n_vars = N-1)
    
    Objective: minimize sum_{k=2}^N f(k) * log(k) / k
    Constraints: For all x in 1..10N:
                 sum_{k=2}^N f(k) * [ floor(x/k) - x/k ] <= 1
                 
    Note: The constraint is derived from M(x) <= 1 where M(x) = sum_{k=1}^N f(k) floor(x/k).
    After normalization, f(1) = -sum_{k=2}^N f(k)/k.
    So M(x) = -x * sum_{k=2}^N f(k)/k + sum_{k=2}^N f(k) floor(x/k)
             = sum_{k=2}^N f(k) (floor(x/k) - x/k)
             = -sum_{k=2}^N f(k) {x/k}
    We require -sum f(k) {x/k} <= 1, which is equivalent to sum f(k) {x/k} >= -1.
    However, linprog handles <= constraints.
    Let's rewrite: -sum f(k) {x/k} <= 1.
    A_ub[i, j] = - (xs[i] % ks[j]) / ks[j]
    b_ub = 1
    """
    n_vars = N - 1
    if n_vars <= 0:
        return None, 0.0

    ks = np.arange(2, N + 1, dtype=np.float64)
    c = np.log(ks) / ks

    max_x = 10 * N
    xs = np.arange(1, max_x + 1, dtype=np.float64)

    # Construct A_ub efficiently using sparse matrix
    # A[i, j] = - (xs[i] % ks[j]) / ks[j]
    
    try:
        A_lil = lil_matrix((max_x, n_vars), dtype=np.float64)
        
        # Vectorize construction as much as possible
        for j in range(n_vars):
            k = int(ks[j])
            # Remainder of xs modulo k
            rem = (xs % k)
            A_lil[:, j] = -rem / k

        A_ub = A_lil.tocsr()
        b_ub = np.ones(max_x)
    except MemoryError:
        return None, 0.0

    bounds = [(-10.0, 10.0)] * n_vars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True})

    if not res.success:
        return None, 0.0

    f_rest = res.x
    f_1 = -np.sum(f_rest / ks)

    if abs(f_1) > 10.0 + 1e-9:
        return None, 0.0

    domain = np.arange(1, N + 1, dtype=np.int64)
    values = np.empty(N, dtype=np.float64)
    values[0] = f_1
    values[1:] = f_rest

    # Compute objective
    a = -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))

    return (domain, values), float(a)


def _verify_feasible(domain, values, tol=1.0001):
    """Verify M(x) <= tol for all x in [1, 10K]."""
    K = int(domain[-1])
    f = values
    # d[j] = sum_{k|j} f(k)
    d = np.zeros(K + 1, dtype=np.float64)
    for idx, k in enumerate(domain):
        kk = int(k)
        d[kk::kk] += f[idx]
    running = np.cumsum(d)
    return bool(running.max() <= tol)


# EVOLVE-BLOCK-END
