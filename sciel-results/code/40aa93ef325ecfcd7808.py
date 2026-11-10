# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _shannon_entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# Precomputed bin indices for fast vectorized entropy computation
_N = 10
_X, _Y = np.meshgrid(np.arange(_N), np.arange(_N))
_DIFF = _X - _Y  # values -9..9
_S1 = _X + _Y    # values 0..18
_S2 = _X + 2*_Y  # values 0..27

# Precompute bin index arrays (0-indexed)
_DIFF_BIN = _DIFF + (_N - 1)  # 0..18
_S1_BIN = _S1                  # 0..18
_S2_BIN = _S2                  # 0..27


def _compute_D_M(P):
    """Compute D = H_{-1} and M = max(H_0, H_1, H_2) for distribution P[y, x]."""
    n = _N

    # D: distribution of x - y (19 bins)
    diff_probs = np.zeros(2 * n - 1)
    np.add.at(diff_probs, _DIFF_BIN.ravel(), P.ravel())
    D = _shannon_entropy(diff_probs)

    # H_0: distribution of x (10 bins) - sum over y for each x
    x_probs = P.sum(axis=0)
    H0 = _shannon_entropy(x_probs)

    # H_1: distribution of x + y (19 bins)
    s1_probs = np.zeros(2 * n - 1)
    np.add.at(s1_probs, _S1_BIN.ravel(), P.ravel())
    H1 = _shannon_entropy(s1_probs)

    # H_2: distribution of x + 2y (28 bins)
    s2_probs = np.zeros(3 * n - 2)
    np.add.at(s2_probs, _S2_BIN.ravel(), P.ravel())
    H2 = _shannon_entropy(s2_probs)

    M = max(H0, H1, H2)
    return D, M


def _ratio_from_P(P):
    """Compute D/M ratio for a normalized distribution P."""
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s <= 1e-9:
        return 0.0
    P = P / s
    D, M = _compute_D_M(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _soft_max_entropy(theta, beta=50.0, n=10):
    """
    Smooth surrogate for max(H_0, H_1, H_2) using log-sum-exp.
    theta: (n, n) unnormalized weights
    Returns (soft_M, D) where soft_M approximates M smoothly.
    """
    theta_shifted = theta - np.max(theta)
    P_unnorm = np.exp(theta_shifted)
    P = P_unnorm / np.sum(P_unnorm)

    D, M = _compute_D_M(P)

    # Recompute individual entropies for soft-max
    x_probs = P.sum(axis=0)
    H0 = _shannon_entropy(x_probs)
    diff_probs = np.zeros(2 * n - 1)
    np.add.at(diff_probs, _DIFF_BIN.ravel(), P.ravel())
    D_val = _shannon_entropy(diff_probs)
    s1_probs = np.zeros(2 * n - 1)
    np.add.at(s1_probs, _S1_BIN.ravel(), P.ravel())
    H1 = _shannon_entropy(s1_probs)
    s2_probs = np.zeros(3 * n - 2)
    np.add.at(s2_probs, _S2_BIN.ravel(), P.ravel())
    H2 = _shannon_entropy(s2_probs)

    entropies = np.array([H0, H1, H2])
    # Soft max via log-sum-exp: beta * logsumexp(entropies / beta) -> max as beta -> inf
    scaled = entropies / beta
    m = np.max(scaled)
    soft_M = beta * (m + np.log(np.sum(np.exp(scaled - m))))

    return soft_M, D_val


def _objective_smooth(theta, n=10, beta=50.0):
    """
    Objective to minimize: -ln(D / soft_M) = -ln(D) + ln(soft_M)
    Using log-ratio for better conditioning.
    """
    theta = theta.reshape(n, n)
    soft_M, D = _soft_max_entropy(theta, beta=beta, n=n)
    if D <= 1e-15 or soft_M <= 1e-15:
        return 1e6
    return -np.log(D) + np.log(soft_M)


def _greedy_sparse_search(n=10, max_support=20, time_budget=30.0):
    """
    Greedy submodular-like search for sparse support sets.
    Iteratively adds the cell that maximizes marginal gain in D/M ratio.
    Returns a list of (support_set, P) pairs.
    """
    start_time = time.time()
    results = []
    
    # Precompute all possible cells
    all_cells = [(y, x) for y in range(n) for x in range(n)]
    
    # Start with several different initial seeds
    seed_sets = [
        [(4, 4)],
        [(0, 9), (9, 0)],
        [(0, 5), (5, 0)],
        [(2, 7), (7, 2)],
        [(3, 8), (8, 3)],
    ]
    
    for seed in seed_sets:
        if time.time() - start_time > time_budget:
            break
        
        support = set(seed)
        current_P = np.zeros((n, n))
        for (y, x) in support:
            current_P[y, x] = 1.0
        current_P /= current_P.sum()
        current_score = _ratio_from_P(current_P)
        
        # Iteratively add cells
        for _ in range(max_support - len(support)):
            if time.time() - start_time > time_budget:
                break
            
            best_gain = -np.inf
            best_cell = None
            best_new_P = None
            best_new_score = None
            
            for (y, x) in all_cells:
                if (y, x) in support:
                    continue
                
                new_support = support | {(y, x)}
                new_P = np.zeros((n, n))
                for (yy, xx) in new_support:
                    new_P[yy, xx] = 1.0
                new_P /= new_P.sum()
                new_score = _ratio_from_P(new_P)
                
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_cell = (y, x)
                    best_new_P = new_P.copy()
                    best_new_score = new_score
            
            if best_cell is None or best_gain < 1e-6:
                break
            
            support.add(best_cell)
            current_P = best_new_P
            current_score = best_new_score
            results.append((set(support).copy(), current_P.copy(), current_score))
    
    return results


def _objective(theta, n=10):
    """
    Objective to minimize: -D/M (exact, for final evaluation)
    theta: flattened parameter vector
    """
    theta = theta.reshape(n, n)
    theta_shifted = theta - np.max(theta)
    P_unnorm = np.exp(theta_shifted)
    P = P_unnorm / np.sum(P_unnorm)
    ratio = _ratio_from_P(P)
    return -ratio


def _objective_and_P(theta, n=10):
    """Return (objective_value, P) for debugging/inspection."""
    theta = theta.reshape(n, n)
    theta_shifted = theta - np.max(theta)
    P_unnorm = np.exp(theta_shifted)
    P = P_unnorm / np.sum(P_unnorm)
    ratio = _ratio_from_P(P)
    return -ratio, P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit

    # Initial best: uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_score = _ratio_from_P(P_uniform)

    # Build initializations
    initializations = []

    # 1. Uniform
    initializations.append(np.zeros((n, n)))

    # 2. Anti-diagonal emphasis: P[y, x] ~ exp(alpha * (x - y))
    for alpha in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
        theta = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                theta[y, x] = alpha * (x - y)
        initializations.append(theta.copy())

    # 3. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        theta = rng.randn(n, n) * 0.5
        initializations.append(theta.copy())

    # 4. Structured: emphasize cells where x-y is spread out
    for spread in [1.0, 2.0, 3.0]:
        theta = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                diff = x - y
                theta[y, x] = spread * abs(diff) / (n - 1)
        initializations.append(theta.copy())

    # 5. Two-anti-diagonal: concentrate on two parallel diagonals to spread x-y
    #    while keeping x, x+y, x+2y projections relatively concentrated
    for c1, c2, w in [(0, 4, 1.0), (2, 6, 1.0), (0, 5, 1.0), (1, 7, 1.0),
                       (0, 4, 2.0), (2, 6, 2.0), (0, 5, 2.0), (1, 7, 2.0),
                       (0, 3, 1.5), (2, 5, 1.5), (1, 6, 1.5), (3, 7, 1.5),
                       (-3, 3, 1.0), (-2, 4, 1.0), (-4, 2, 1.0), (-1, 5, 1.0)]:
        theta = np.full((n, n), -10.0)  # very low default
        for y in range(n):
            for x in range(n):
                d = x - y
                if d == c1 or d == c2:
                    theta[y, x] = w
        initializations.append(theta.copy())

    # 6. Three-anti-diagonal patterns
    for c1, c2, c3, w in [(0, 4, 8, 1.0), (1, 5, 9, 1.0), (-1, 3, 7, 1.0),
                           (-2, 2, 6, 1.0), (0, 5, 9, 1.0), (1, 4, 7, 1.0)]:
        theta = np.full((n, n), -10.0)
        for y in range(n):
            for x in range(n):
                d = x - y
                if d in (c1, c2, c3):
                    theta[y, x] = w
        initializations.append(theta.copy())

    # 7. Checkerboard-like patterns that spread x-y
    for phase in [0, 1]:
        theta = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x + y + phase) % 2 == 0:
                    theta[y, x] = 1.0
                else:
                    theta[y, x] = -1.0
        initializations.append(theta.copy())

    # 8. Gaussian mixture initializations: place weight on 2-3 diagonal clusters
    rng2 = np.random.RandomState(123)
    for _ in range(6):
        theta = np.full((n, n), -8.0)
        n_clusters = rng2.randint(2, 4)
        for _ in range(n_clusters):
            center_x = rng2.uniform(0, n)
            center_y = rng2.uniform(0, n)
            sigma = rng2.uniform(1.0, 3.0)
            for y in range(n):
                for x in range(n):
                    dist_sq = (x - center_x)**2 + (y - center_y)**2
                    theta[y, x] = max(theta[y, x], -dist_sq / (2 * sigma**2))
        initializations.append(theta.copy())

    # Phase 0: Greedy sparse search to find good sparse supports
    if time.time() < deadline - 120:
        try:
            greedy_results = _greedy_sparse_search(n=n, max_support=25, time_budget=60.0)
            for support_set, P_greedy, score_greedy in greedy_results:
                if score_greedy > best_score:
                    best_score = score_greedy
                    best_P = P_greedy.copy()
            
            # Add greedy results as additional initializations
            for support_set, P_greedy, score_greedy in greedy_results[-3:]:
                if score_greedy > 0:
                    theta_g = np.full((n, n), -10.0)
                    for (y, x) in support_set:
                        theta_g[y, x] = 0.0
                    initializations.append(theta_g.copy())
        except Exception:
            pass

    # Phase 1: Smooth optimization with soft-max surrogate (adaptive beta)
    for init_theta in initializations:
        if time.time() > deadline - 90:
            break

        try:
            # Try multiple beta values for the soft-max surrogate
            for beta in [20.0, 50.0, 100.0]:
                if time.time() > deadline - 90:
                    break
                result = minimize(
                    _objective_smooth,
                    init_theta.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50},
                    args=(n, beta),
                )
                theta_opt = result.x.reshape(n, n)
                theta_shifted = theta_opt - np.max(theta_opt)
                P_candidate = np.exp(theta_shifted)
                P_candidate = P_candidate / np.sum(P_candidate)
                score_candidate = _ratio_from_P(P_candidate)
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_P = P_candidate.copy()
        except Exception:
            continue

    # Phase 2: High-precision polish on the best solution with multiple methods
    if time.time() < deadline - 45:
        try:
            theta_best = np.zeros((n, n))
            log_P = np.log(best_P + 1e-300)
            theta_best = log_P - np.max(log_P)

            # L-BFGS-B polish
            result = minimize(
                _objective,
                theta_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100},
            )
            theta_opt = result.x.reshape(n, n)
            theta_shifted = theta_opt - np.max(theta_opt)
            P_candidate = np.exp(theta_shifted)
            P_candidate = P_candidate / np.sum(P_candidate)
            score_candidate = _ratio_from_P(P_candidate)
            if score_candidate > best_score:
                best_score = score_candidate
                best_P = P_candidate.copy()
        except Exception:
            pass

    # Phase 3: SLSQP polish
    if time.time() < deadline - 25:
        try:
            theta_best = np.log(best_P + 1e-300)
            theta_best = theta_best - np.max(theta_best)

            result = minimize(
                _objective,
                theta_best.flatten(),
                method='SLSQP',
                options={'maxiter': 600, 'ftol': 1e-14},
            )
            theta_opt = result.x.reshape(n, n)
            theta_shifted = theta_opt - np.max(theta_opt)
            P_candidate = np.exp(theta_shifted)
            P_candidate = P_candidate / np.sum(P_candidate)
            score_candidate = _ratio_from_P(P_candidate)
            if score_candidate > best_score:
                best_score = score_candidate
                best_P = P_candidate.copy()
        except Exception:
            pass

    # Phase 4: Final ultra-tight L-BFGS-B polish
    if time.time() < deadline - 10:
        try:
            theta_best = np.log(best_P + 1e-300)
            theta_best = theta_best - np.max(theta_best)

            result = minimize(
                _objective,
                theta_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200},
            )
            theta_opt = result.x.reshape(n, n)
            theta_shifted = theta_opt - np.max(theta_opt)
            P_candidate = np.exp(theta_shifted)
            P_candidate = P_candidate / np.sum(P_candidate)
            score_candidate = _ratio_from_P(P_candidate)
            if score_candidate > best_score:
                best_score = score_candidate
                best_P = P_candidate.copy()
        except Exception:
            pass

    # Phase 5: Final Nelder-Mead local refinement
    if time.time() < deadline - 3:
        try:
            theta_best = np.log(best_P + 1e-300)
            theta_best = theta_best - np.max(theta_best)

            result = minimize(
                _objective,
                theta_best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14},
            )
            theta_opt = result.x.reshape(n, n)
            theta_shifted = theta_opt - np.max(theta_opt)
            P_candidate = np.exp(theta_shifted)
            P_candidate = P_candidate / np.sum(P_candidate)
            score_candidate = _ratio_from_P(P_candidate)
            if score_candidate > best_score:
                best_score = score_candidate
                best_P = P_candidate.copy()
        except Exception:
            pass

    # Final validation and normalization
    P = np.clip(best_P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    # Ensure no NaN or inf
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END
