"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, roots_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute transformation matrix from monomial basis to Laguerre basis L_k^(1)
    # L_k^(1)(x) = sum_{j=0}^k a_{k,j} x^j
    # We need c such that sum c_k L_k^(1)(x) = P(x) = sum p_j x^j
    # This implies p = A^T c, where A[k,j] is coeff of x^j in L_k^(1)
    # So c = (A^T)^-1 p
    
    A = np.zeros((m+1, m+1))
    for k in range(m+1):
        for j in range(k+1):
            # L_k^(alpha)(x) = sum_{j=0}^k (-1)^j binom(k+alpha, k-j) x^j / j!
            # alpha=1
            # coeff of x^j is (-1)^j * binom(k+1, k-j) / j!
            # binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Precompute inverse of A^T
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(1, m+2) # k+1 for k=0..m

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod(x-r_i) in the Laguerre basis.
        """
        p = np.array([1.0])
        for r in roots:
            p = np.convolve(p, np.array([1.0, -r]))
        
        if len(p) < m+1:
            p = np.pad(p, (0, m+1-len(p)))
        
        c = A_T_inv @ p
        norm_sq = np.dot(c**2, weights)
        return norm_sq

    def objective(y):
        """
        Minimize -norm_sq to maximize norm_sq.
        y: log of roots.
        """
        roots = np.exp(y)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e100
        try:
            nsq = compute_norm_sq(roots)
            if nsq <= 0:
                return 1e100
            return -nsq
        except Exception:
            return 1e100

    best_val = -1e100
    best_y = None

    # Start 1: Geometric progression (seed)
    y_seed = np.log([4.0 * (2.0 ** i) for i in range(m)])
    
    # Start 2: Roots of Laguerre polynomial L_m^(n)
    # Note: roots_genlaguerre returns a tuple (roots, weights)
    roots_lag, _ = roots_genlaguerre(m, n)
    y_lag = np.log(roots_lag.real)

    # Start 3: Uniformly spaced
    y_unif = np.log(np.linspace(0.1, 10.0, m))

    starts = [y_seed, y_lag, y_unif]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(10):
        y_rand = np.random.uniform(-2.0, 3.0, m)
        starts.append(y_rand)
        
    # Add geometric progression perturbations (based on knowledge that geometric roots are optimal)
    # The seed was 4.0 * 2^i. Let's try varying the base and ratio slightly.
    for base in [2.0, 3.0, 5.0, 8.0]:
        for ratio in [1.5, 1.8, 2.0, 2.2, 2.5]:
            y_geo = np.log([base * (ratio ** i) for i in range(m)])
            starts.append(y_geo)

    for y0 in starts:
        if not time_ok():
            break
        try:
            # Use L-BFGS-B with tighter tolerances and higher maxiter for better convergence
            res = minimize(objective, y0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_y = res.x
        except Exception:
            continue

    # Polish
    if best_y is not None and time_ok():
        try:
            res2 = minimize(objective, best_y, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_y = res2.x
        except Exception:
            pass

    if best_y is None:
        return [4.0 * (2.0 ** i) for i in range(m)]

    best_roots = np.exp(best_y)
    best_roots = sorted(best_roots)
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
