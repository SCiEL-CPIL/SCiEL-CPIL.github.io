# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses a Linear Program formulation to find the optimal f on a chosen domain.
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin

    # Domain size to use for the LP.
    # We want to balance constraint count (K * 10K = 10 K^2) with objective quality.
    # K=200 -> 40,000 constraints.
    # K=300 -> 90,000 constraints.
    # K=500 -> 250,000 constraints.
    # Let's try K=300 first. If it's too slow, we might need to reduce.
    K = 300

    domain = np.arange(1, K + 1, dtype=np.int64)

    # Objective: Maximize a = -sum f(k) * log(k) / k
    # This is equivalent to Minimize c^T f where c[k] = log(k+1) / (k+1)
    c = np.zeros(K)
    for k in range(1, K + 1):
        c[k - 1] = np.log(k) / k

    # Constraints:
    # 1. M(x) <= 1.0001 for x = 1..10K
    #    M(x) = sum_{k=1}^K f(k) floor(x/k)
    #    A_ub is a matrix of shape (10K, K) where A_ub[x-1, k-1] = floor(x/k)
    #    b_ub is a vector of shape (10K,) filled with 1.0001
    n_x = 10 * K
    A_ub = np.zeros((n_x, K))
    for x in range(1, n_x + 1):
        for k in range(1, K + 1):
            A_ub[x - 1, k - 1] = x // k

    b_ub = np.full(n_x, 1.0001)

    # 2. sum f(k) / k = 0
    #    This is an equality constraint: A_eq f = 0
    #    A_eq is a vector of shape (1, K) where A_eq[0, k-1] = 1/k
    A_eq = np.zeros((1, K))
    for k in range(1, K + 1):
        A_eq[0, k - 1] = 1.0 / k
    b_eq = np.array([0.0])

    # Bounds:
    # Values are clipped to [-10, 10] by the evaluator.
    # So we can bound f(k) to [-10, 10].
    bounds = [(-10.0, 10.0)] * K

    # Solve the LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to a simple heuristic if LP fails
            # Return the seed solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
        
        values = res.x
    except Exception:
        # Fallback to seed solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Check time
    if time.time() > deadline:
        # If we are running out of time, just return the current best (which is the LP solution)
        # But we should verify feasibility quickly? The evaluator will do it.
        # The LP guarantees feasibility (M(x) <= 1.0001 and sum f(k)/k = 0).
        # However, the evaluator clips values to [-10, 10] BEFORE checking constraints.
        # Since we bounded f(k) to [-10, 10], the clipping does nothing.
        # So the solution is valid.
        pass

    return domain, values


# EVOLVE-BLOCK-END
