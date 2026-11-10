# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N, dtype=np.float64)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

# Precompute bin indices for each projection
# x - y ranges from -9 to 9 -> 19 bins
_IDX_D = (_XS - _YS).astype(np.int64) + 9
# x ranges 0..9 -> 10 bins
_IDX_H0 = _XS.astype(np.int64)
# x + y ranges 0..18 -> 19 bins
_IDX_H1 = (_XS + _YS).astype(np.int64)
# x + 2y ranges 0..27 -> 28 bins
_IDX_H2 = (_XS + 2 * _YS).astype(np.int64)

_N_BINS_D = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28

_ALL_CELLS = np.arange(_N * _N)

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector, ignoring zeros."""
    s = counts.sum()
    if s <= 1e-30:
        return 0.0
    p = counts / s
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _compute_value(P):
    """Compute D/M ratio for a given 10x10 distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0
    p = p / total

    # D: distribution of x - y
    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    # H_0: distribution of x
    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    # H_1: distribution of x + y
    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    # H_2: distribution of x + 2y
    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m

def _negative_value(theta):
    """Objective to minimize: -D/M. theta is a 100-dim vector of log-probabilities."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    if val <= 1e-9:
        return 1e6
    return -val

def _uniform_init():
    return np.zeros(_N * _N, dtype=np.float64)

def _random_init():
    return np.random.randn(_N * _N) * 0.1

def _compute_entropies_on_support(weights, support):
    """
    Compute entropies H_D, H_0, H_1, H_2 for weights distributed on support.
    weights: array of length len(support), sum 1.
    support: array of flat indices.
    """
    # D
    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D[support], weights)
    hd = _entropy_from_counts(d_counts)
    
    # H0
    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0[support], weights)
    h0 = _entropy_from_counts(h0_counts)
    
    # H1
    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1[support], weights)
    h1 = _entropy_from_counts(h1_counts)
    
    # H2
    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2[support], weights)
    h2 = _entropy_from_counts(h2_counts)
    
    return hd, h0, h1, h2

def _optimize_weights_on_support(support, deadline, max_iter=200, n_restarts=3):
    """
    Optimize weights on a fixed support using L-BFGS-B.
    Returns best ratio and the full 10x10 array P.
    """
    K = len(support)
    if K == 0:
        return 0.0, None
    
    best_ratio = 0.0
    best_P = None
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if i == 0:
            theta0 = np.zeros(K)
        else:
            theta0 = rng.standard_normal(K) * 0.5
            
        def obj(theta):
            z = theta - np.max(theta)
            exp_z = np.exp(z)
            s = exp_z.sum()
            if s < 1e-30:
                return 1e6
            w = exp_z / s
            hd, h0, h1, h2 = _compute_entropies_on_support(w, support)
            m = max(h0, h1, h2)
            if m <= 1e-9:
                return 1e6
            return -(hd / m)
            
        try:
            res = minimize(
                obj,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if res.fun is not None:
                ratio = -res.fun
                if ratio > best_ratio:
                    best_ratio = ratio
                    # Reconstruct P
                    z = res.x - np.max(res.x)
                    exp_z = np.exp(z)
                    w = exp_z / exp_z.sum()
                    P = np.zeros(_N * _N)
                    P[support] = w
                    P = P.reshape(_N, _N)
                    best_P = P
        except Exception:
            pass
            
    return best_ratio, best_P

def _greedy_support_search(deadline, max_steps=15, penalty=0.5, seed=0):
    """
    Greedily expand support to maximize D/M.
    Starts with a random small support, then adds cells that improve the score.
    """
    rng = np.random.default_rng(seed)
    
    # Initial support: random 3-5 cells
    init_size = rng.integers(3, 6)
    support = rng.choice(_ALL_CELLS, size=init_size, replace=False)
    support_set = set(support.tolist())
    
    # Compute initial score with uniform weights
    w = np.ones(len(support)) / len(support)
    hd, h0, h1, h2 = _compute_entropies_on_support(w, support)
    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0, None
    current_score = hd / m
    
    best_score = current_score
    best_support = support.copy()
    best_w = w.copy()
    
    for step in range(max_steps):
        if time.time() > deadline - 10.0:
            break
            
        best_gain = -np.inf
        best_cell = None
        best_new_w = None
        best_new_score = current_score
        
        # Evaluate adding each candidate cell
        for cell in _ALL_CELLS:
            if cell in support_set:
                continue
                
            new_support = np.append(support, cell)
            new_K = len(new_support)
            # Uniform weights on new support
            new_w = np.ones(new_K) / new_K
            
            hd, h0, h1, h2 = _compute_entropies_on_support(new_w, new_support)
            m = max(h0, h1, h2)
            if m <= 1e-9:
                continue
            new_score = hd / m
            
            # Gain is increase in score. We can also look at delta_HD - penalty * delta_M
            # But simple score increase is a good heuristic.
            gain = new_score - current_score
            
            if gain > best_gain:
                best_gain = gain
                best_cell = cell
                best_new_w = new_w
                best_new_score = new_score
                
        if best_cell is None or best_gain <= 1e-6:
            break
            
        # Add the best cell
        support = np.append(support, best_cell)
        support_set.add(best_cell)
        current_score = best_new_score
        w = best_new_w
        
        if current_score > best_score:
            best_score = current_score
            best_support = support.copy()
            best_w = w.copy()
            
    # Now optimize weights on the best support found
    ratio, P = _optimize_weights_on_support(best_support, deadline, max_iter=100, n_restarts=2)
    if ratio > best_score:
        return ratio, P
    else:
        # If optimization didn't improve, use the uniform weights on best_support
        P = np.zeros(_N * _N)
        P[best_support] = best_w
        P = P.reshape(_N, _N)
        return best_score, P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s

    best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)
    best_score = _compute_value(best_P)

    # 1. Try Greedy Support Search with different seeds
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 60.0:
            break
        try:
            score, P = _greedy_support_search(deadline, max_steps=20, seed=i)
            if P is not None and score > best_score:
                best_score = score
                best_P = P
        except Exception:
            pass

    # 2. Try Full Grid Optimization (original approach) if time permits
    if time.time() < deadline - 120.0:
        n_restarts = 10
        for i in range(n_restarts):
            if time.time() > deadline - 30.0:
                break

            if i == 0:
                theta0 = _uniform_init()
            else:
                theta0 = _random_init()

            try:
                result = minimize(
                    _negative_value,
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                theta_opt = result.x
                theta_shifted = theta_opt - theta_opt.max()
                exp_theta = np.exp(theta_shifted)
                p = exp_theta / exp_theta.sum()
                P = p.reshape(_N, _N)

                P = np.clip(P, 0.0, None)
                s = P.sum()
                if s <= 1e-9:
                    continue
                P = P / s

                score = _compute_value(P)
                if score > best_score:
                    best_score = score
                    best_P = P.copy()

            except Exception:
                continue

    # 3. Final Polish: Optimize weights on the support of the best_P found
    if time.time() < deadline - 10.0:
        support = np.where(best_P.ravel() > 1e-6)[0]
        if len(support) > 0:
            # Extract weights
            w = best_P.ravel()[support]
            w = w / w.sum()
            
            # Use the optimized weights as a starting point for a high-precision run
            # We can try to refine the support slightly or just re-optimize weights
            # Let's just re-optimize weights on the current support with tighter tolerance
            def obj_refine(theta):
                z = theta - np.max(theta)
                exp_z = np.exp(z)
                s = exp_z.sum()
                if s < 1e-30:
                    return 1e6
                new_w = exp_z / s
                P_temp = np.zeros(_N * _N)
                P_temp[support] = new_w
                P_temp = P_temp.reshape(_N, _N)
                return -_compute_value(P_temp)
                
            # Initialize theta from current weights
            theta_init = np.log(np.maximum(w, 1e-15))
            
            try:
                res = minimize(
                    obj_refine,
                    theta_init,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                if res.fun is not None:
                    z = res.x - np.max(res.x)
                    exp_z = np.exp(z)
                    new_w = exp_z / exp_z.sum()
                    P_refined = np.zeros(_N * _N)
                    P_refined[support] = new_w
                    P_refined = P_refined.reshape(_N, _N)
                    P_refined = np.clip(P_refined, 0.0, None)
                    s = P_refined.sum()
                    if s > 1e-9:
                        P_refined = P_refined / s
                        score = _compute_value(P_refined)
                        if score > best_score:
                            best_score = score
                            best_P = P_refined
            except Exception:
                pass

    # Final cleanup: ensure validity
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    return best_P

# EVOLVE-BLOCK-END
