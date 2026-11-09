import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute CW masks and weights
    cw_masks = np.zeros((D, n), dtype=np.int64)
    weights = np.zeros(D, dtype=np.int64)

    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        dist = (t - s) % n
        for k in range(dist):
            e = (s + k) % n
            cw_masks[i, e] = 1

    # Base load if all CW
    base_load = cw_masks.T @ weights

    # Contribution if switched to CCW
    # diff = CCW_mask - CW_mask = (1 - CW_mask) - CW_mask = 1 - 2*CW_mask
    diff_masks = 1 - 2 * cw_masks
    contrib = weights[:, np.newaxis] * diff_masks  # shape (D, n)

    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)

    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Phase 1: Brute force all 2^15 combinations using Gray code
    num_steps = 1 << D
    current_load = base_load.copy()
    current_routing = np.zeros(D, dtype=np.int64)

    # Check initial state (all CW)
    max_load = np.max(current_load)
    if max_load < best_max_load:
        best_max_load = max_load
        best_routing = current_routing.copy()

    for step in range(1, num_steps):
        if time.time() > deadline - 30:  # Leave 30s for Phase 2
            break

        prev_g = (step - 1) ^ ((step - 1) >> 1)
        curr_g = step ^ (step >> 1)
        diff_bits = prev_g ^ curr_g
        idx = diff_bits.bit_length() - 1

        if (curr_g >> idx) & 1:
            current_load += contrib[idx]
            current_routing[idx] = 1
        else:
            current_load -= contrib[idx]
            current_routing[idx] = 0

        max_load = np.max(current_load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing.copy()

    # Phase 2: Smooth optimization refinement using log-sum-exp
    # We use the brute-force result as a starting point and try to improve
    # via continuous optimization, then round and do local search.

    if time.time() < deadline - 10:
        # Log-sum-exp smooth approximation of max
        # f(x) = (1/alpha) * log(sum(exp(alpha * L_e)))
        # where L_e = base_load[e] + sum_i x_i * contrib[i, e]

        def smooth_max_load(x, alpha):
            # x is continuous, shape (D,)
            # Load on each edge
            load = base_load + x @ contrib
            # Soft max via log-sum-exp
            # For numerical stability, subtract max
            m = np.max(load)
            if m > 1e10:  # overflow protection
                return m
            return (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (load - m)))) + m

        def smooth_max_load_grad(x, alpha):
            load = base_load + x @ contrib
            m = np.max(load)
            if m > 1e10:
                # Gradient of max is one-hot at argmax
                idx = np.argmax(load)
                grad = np.zeros(D)
                # dL_e/dx_i = contrib[i, e]
                # grad_i = sum_e (dL_e/dx_i) * (1 if e==argmax else 0)
                # = contrib[i, argmax]
                grad = contrib[:, idx]
                return grad
            exp_load = np.exp(alpha * (load - m))
            softmax_weights = exp_load / np.sum(exp_load)
            # d/dx_i of (1/alpha)*log(sum(exp(alpha*L_e)))
            # = sum_e softmax_weights[e] * contrib[i, e]
            grad = contrib.T @ softmax_weights
            return grad

        # Try several alpha values and starting points
        alphas = [50.0, 100.0, 200.0, 500.0]
        start_points = [
            best_routing.astype(np.float64),
            np.zeros(D),
            np.ones(D),
            np.random.uniform(0, 1, D)
        ]

        for alpha in alphas:
            if time.time() > deadline - 5:
                break
            for x0 in start_points:
                if time.time() > deadline - 5:
                    break
                try:
                    res = minimize(
                        smooth_max_load,
                        x0,
                        args=(alpha,),
                        method='L-BFGS-B',
                        jac=lambda x, a: smooth_max_load_grad(x, a),
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    # Round the continuous solution
                    x_rounded = (res.x >= 0.5).astype(np.int64)
                    load = base_load + x_rounded @ contrib
                    max_l = np.max(load)
                    if max_l < best_max_load:
                        best_max_load = max_l
                        best_routing = x_rounded.copy()
                except Exception:
                    pass

        # Phase 3: Local search on the best binary solution found
        # Try all single-flip and double-flip moves
        current_best = best_routing.copy()
        current_max = best_max_load

        # Single flip
        for i in range(D):
            if time.time() > deadline - 2:
                break
            trial = current_best.copy()
            trial[i] = 1 - trial[i]
            load = base_load + trial @ contrib
            max_l = np.max(load)
            if max_l < current_max:
                current_max = max_l
                current_best = trial.copy()

        # Double flip (from the improved single-flip solution)
        for i in range(D):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, D):
                trial = current_best.copy()
                trial[i] = 1 - trial[i]
                trial[j] = 1 - trial[j]
                load = base_load + trial @ contrib
                max_l = np.max(load)
                if max_l < current_max:
                    current_max = max_l
                    current_best = trial.copy()

        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = current_best.copy()

    return best_routing.astype(np.float64)
