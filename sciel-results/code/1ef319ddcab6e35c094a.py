# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _shannon_entropy(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _project_entropy(P, r):
    """Compute Shannon entropy of the distribution of x + r*y over the grid."""
    n = P.shape[0]
    # P[y, x] where y is row index (0..n-1), x is column index (0..n-1)
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    vals = xs + r * ys
    
    # Use bincount for efficient and safe pooling
    # We need to map vals to unique indices. 
    # Since r can be float, we can't directly bincount float vals.
    # We must use unique/inverse or sort.
    # For small n=10, unique is fine and safe.
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    proj = np.bincount(inverse.flatten(), weights=P.flatten(), minlength=len(unique_vals))
    
    return _shannon_entropy(proj)


def _objective(params):
    """Objective to maximize D/M. We minimize -D/M."""
    n = 10
    P = params.reshape(n, n)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 1e6
    P = P / s
    
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return 1e6
    if D < 1e-12:
        return 1e6
    
    return -D / M


def _make_initial_patterns(n):
    """Generate structured initial points targeting concentrated-line patterns."""
    patterns = []
    
    # Uniform
    patterns.append(np.ones((n, n)) / (n * n))
    
    # Concentrate on diagonal x = y
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1.0 / n
    patterns.append(P)
    
    # Concentrate on anti-diagonal x + y = n-1
    P = np.zeros((n, n))
    for i in range(n):
        P[i, n - 1 - i] = 1.0 / n
    patterns.append(P)
    
    # Concentrate on a few parallel lines with slope -1 (x - y = const)
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # Random sparse patterns
    rng = np.random.RandomState(42)
    for _ in range(10):
        P = np.zeros((n, n))
        k = rng.randint(5, 15)
        indices = rng.choice(n * n, size=k, replace=False)
        P.flatten()[indices] = 1.0
        P /= P.sum()
        patterns.append(P)
        
    # Add some random dense patterns
    for _ in range(5):
        P = rng.rand(n, n)
        P /= P.sum()
        patterns.append(P)

    return patterns


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses multi-start Nelder-Mead optimization.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Generate structured initial points
    initial_patterns = _make_initial_patterns(n)
    
    best_P = initial_patterns[0].copy()
    best_score = -_objective(best_P.flatten())
    
    # Multi-start Nelder-Mead
    # We will run NM from several starting points
    # To save time, we limit the number of starts and iterations
    
    num_starts = min(len(initial_patterns), 15)
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        x0 = initial_patterns[i].flatten().copy()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = -_objective(P_candidate.flatten())
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
                    
        except Exception:
            continue
            
    # If we have time, do a few more random restarts or polish the best
    if time.time() < deadline - 60:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb best_P
            noise = rng.rand(n, n) * 0.1
            P_start = best_P + noise
            P_start = np.clip(P_start, 0, None)
            s = P_start.sum()
            if s > 1e-12:
                P_start /= s
            else:
                continue
                
            try:
                result = minimize(
                    _objective,
                    P_start.flatten().copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                P_candidate = result.x.reshape(n, n)
                P_candidate = np.clip(P_candidate, 0, None)
                s = P_candidate.sum()
                if s > 1e-12:
                    P_candidate /= s
                    score = -_objective(P_candidate.flatten())
                    if score > best_score:
                        best_score = score
                        best_P = P_candidate.copy()
            except Exception:
                continue

    # Final normalization and clipping
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-12:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s
    
    return best_P


# EVOLVE-BLOCK-END
