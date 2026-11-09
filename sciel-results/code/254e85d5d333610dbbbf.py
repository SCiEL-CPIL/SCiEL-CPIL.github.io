# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 600.0  # safety margin before 630s kill

    # Start with a strong baseline
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0  # seed score

    # Try to improve via LP over a range of k values
    # The constraint is M(x) = sum_k f(k)*floor(x/k) <= 1 for x in [1, 10*K_max]
    # Objective: maximize -sum_k f(k)*log(k)/k for k >= 2 (f(1) is adjusted by evaluator)

    # We'll use scipy's linprog if available, otherwise fall back to baseline
    try:
        from scipy.optimize import linprog
        has_scipy = True
    except ImportError:
        has_scipy = False

    if not has_scipy:
        return domain, values

    # Try increasing K_max values
    for K_max in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > DEADLINE - 30:
            break

        # Variables: f(2), f(3), ..., f(K_max) -> K_max - 1 variables
        n_vars = K_max - 1
        if n_vars == 0:
            continue

        # Objective: maximize -sum_{k=2}^{K_max} f(k)*log(k)/k
        # linprog minimizes, so minimize sum f(k)*log(k)/k
        c = np.array([np.log(k) / k for k in range(2, K_max + 1)], dtype=np.float64)

        # Constraints: sum_k f(k)*floor(x/k) <= 1 for x = 1, 2, ..., 10*K_max
        # Note: for x < 2, floor(x/k) = 0 for all k >= 2, so those constraints are trivially satisfied
        # For x >= 2: sum_{k=2}^{K_max} f(k)*floor(x/k) <= 1
        # But we also need to account for the fact that f(1) will be adjusted.
        # Actually, the constraint is on the FULL sum including f(1), but f(1) is determined
        # by the sum constraint. Let me re-read the problem.
        #
        # The evaluator: clips values, then f(1) = f(1) - S where S = sum_k f(k)/k
        # So sum_k f(k)/k = 0 after adjustment.
        # Then M(x) = sum_k f(k)*floor(x/k) must be <= 1.
        #
        # Let's denote the raw values as g(k). After clipping to [-10,10] and adjusting f(1):
        # f(k) = g(k) for k >= 2 (clipped), f(1) = g(1) - sum_k g(k)/k (clipped)
        #
        # M(x) = sum_k f(k)*floor(x/k) = f(1)*floor(x/1) + sum_{k=2} f(k)*floor(x/k)
        #      = (g(1) - sum_k g(k)/k) * x + sum_{k=2} g(k)*floor(x/k)
        #
        # This is complex. Let me simplify by assuming g(1) = 0 initially and see.
        # Actually, let's just work with the constraint as stated and include f(1) as a variable too.

        # Let me restart: include f(1) as a variable
        # Variables: f(1), f(2), ..., f(K_max) -> K_max variables
        n_vars = K_max
        c_obj = np.zeros(n_vars, dtype=np.float64)
        for k in range(2, K_max + 1):
            c_obj[k-1] = np.log(k) / k  # minimize this, so maximize -sum

        # Constraints: sum_k f(k)*floor(x/k) <= 1 for x = 1, ..., 10*K_max
        # And: sum_k f(k)/k = 0 (this is enforced by the evaluator, but we can include it)
        # Actually, the evaluator enforces sum f(k)/k = 0 by adjusting f(1).
        # So we should NOT include that as a constraint in our LP; instead, we should
        # account for the fact that f(1) is determined by the other values.

        # Let me think differently. Let's define the problem after the evaluator's adjustment.
        # Let h(k) = f(k) for k >= 2 (our variables), and f(1) = -sum_{k=2} h(k)/k
        # Then M(x) = f(1)*x + sum_{k=2} h(k)*floor(x/k)
        #          = -x*sum_{k=2} h(k)/k + sum_{k=2} h(k)*floor(x/k)
        #          = sum_{k=2} h(k) * (floor(x/k) - x/k)
        #
        # So the constraint is: sum_{k=2}^{K_max} h(k) * (floor(x/k) - x/k) <= 1 for all x in [1, 10*K_max]
        # And the objective is to maximize -sum_{k=2}^{K_max} h(k)*log(k)/k

        # This is cleaner! Let's build the LP with this formulation.
        n_vars = K_max - 1  # h(2), ..., h(K_max)

        # Objective: minimize sum_{k=2}^{K_max} h(k)*log(k)/k
        c_lp = np.array([np.log(k) / k for k in range(2, K_max + 1)], dtype=np.float64)

        # Constraints: for each x in [1, 10*K_max]:
        # sum_{k=2}^{K_max} h(k) * (floor(x/k) - x/k) <= 1
        # Note: floor(x/k) - x/k is in [-1, 0] for all x, k
        # Actually, floor(x/k) <= x/k, so floor(x/k) - x/k <= 0
        # So the LHS is a sum of h(k) * (negative or zero terms)

        # We need to be careful: some of these coefficients are zero (when k > x)
        # Let's build the constraint matrix
        x_max = 10 * K_max
        n_constraints = x_max

        # Build A_ub matrix: n_constraints x n_vars
        # This could be large, so let's be smart about it
        # For a given x, the coefficient for h(k) is (floor(x/k) - x/k) if k <= x, else 0

        # Instead of building the full matrix, let's use a sparse approach or limit constraints
        # Actually, for K_max up to 200000, x_max = 2,000,000 which is too many constraints
        # Let's sample constraints or use a smarter approach

        # Alternative: note that M(x) is a step function that only changes at multiples of k
        # So we only need to check x values where M(x) could be maximum
        # M(x) = sum_{k=2} h(k)*(floor(x/k) - x/k)
        # This is linear in x between multiples, so the maximum occurs at the right endpoint
        # of each interval, i.e., at x = m*k - 1 for various m, k

        # This is getting complex. Let me just try small K_max with full constraints first.
        if K_max > 200:
            # Use a subset of constraints: check x = 1, 2, ..., K_max and x = 10*K_max
            # Actually, let's just check x values that are multiples of small k's
            # For now, let's limit to K_max <= 200 with full constraints
            continue

        # Build constraint matrix for small K_max
        A_rows = []
        b_vals = []
        for x in range(1, x_max + 1):
            row = np.zeros(n_vars, dtype=np.float64)
            for idx, k in enumerate(range(2, K_max + 1)):
                if k <= x:
                    row[idx] = np.floor(x / k) - x / k
                # else: row[idx] = 0 (already zero)
            A_rows.append(row)
            b_vals.append(1.0)

        A_ub = np.array(A_rows, dtype=np.float64)
        b_ub = np.array(b_vals, dtype=np.float64)

        # Bounds: f(k) is clipped to [-10, 10]
        bounds = [(-10.0, 10.0)] * n_vars

        # Solve the LP
        try:
            result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if time.time() - start_time > DEADLINE - 10:
                break

            if result.success:
                h_vals = result.x  # h(2), ..., h(K_max)
                # Compute f(1)
                f1 = -np.sum(h_vals / np.arange(2, K_max + 1))

                # Build the full domain and values
                all_domain = np.arange(1, K_max + 1, dtype=np.int64)
                all_values = np.zeros(K_max, dtype=np.float64)
                all_values[0] = f1
                all_values[1:] = h_vals

                # Clip to [-10, 10] as the evaluator does
                all_values = np.clip(all_values, -10.0, 10.0)

                # Verify the constraint
                # Recompute M(x) for all x
                max_M = 0.0
                for x in range(1, x_max + 1):
                    Mx = 0.0
                    for k in range(1, K_max + 1):
                        Mx += all_values[k-1] * np.floor(x / k)
                    if Mx > max_M:
                        max_M = Mx

                if max_M <= 1.0001:
                    # Compute the objective
                    a = -np.sum(all_values[1:] * np.log(np.arange(2, K_max + 1)) / np.arange(2, K_max + 1))
                    # Note: the evaluator computes a = -sum_k f(k)*log(k)/k for ALL k, including k=1
                    # But log(1) = 0, so f(1)*log(1)/1 = 0
                    # So a = -sum_{k=2} f(k)*log(k)/k
                    score = a
                    if score > best_score:
                        best_score = score
                        domain = all_domain
                        values = all_values
        except Exception as e:
            pass

    return domain, values

# EVOLVE-BLOCK-END
