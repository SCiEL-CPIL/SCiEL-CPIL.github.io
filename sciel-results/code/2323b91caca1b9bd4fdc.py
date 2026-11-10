import numpy as np
import time

# Precompute bin indices for fast entropy calculation
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _score_from_weights(w):
    """
    Fast score computation assuming w is a flat array of non-negative weights summing to 1.
    """
    d_probs = np.zeros(_N_D)
    np.add.at(d_probs, _d_vals, w)
    h0_probs = np.zeros(_N_H0)
    np.add.at(h0_probs, _h0_vals, w)
    h1_probs = np.zeros(_N_H1)
    np.add.at(h1_probs, _h1_vals, w)
    h2_probs = np.zeros(_N_H2)
    np.add.at(h2_probs, _h2_vals, w)
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten() / s
    return _score_from_weights(w)

def _generate_quadratic_distribution(a, b, cx, cy):
    """
    Generate a 10x10 distribution of the form:
    P[y, x] proportional to exp( -a * (x - y - cx)^2 - b * (x + y - cy)^2 )
    
    Parameters:
    a: weight for the (x-y) direction (spread along anti-diagonal)
    b: weight for the (x+y) direction (spread along diagonal)
    cx: center offset for (x-y)
    cy: center offset for (x+y)
    """
    x = _x_idx
    y = _y_idx
    
    # We work in the transformed coordinates
    u = x - y - cx
    v = x + y - cy
    
    # Calculate unnormalized log-probabilities
    log_p = -a * u**2 - b * v**2
    
    # Convert to probabilities
    # Shift to prevent underflow/overflow
    log_p_max = np.max(log_p)
    p_unnorm = np.exp(log_p - log_p_max)
    
    total = p_unnorm.sum()
    if total < 1e-15:
        return np.ones((10, 10)) / 100.0
        
    P = p_unnorm / total
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    start_time = time.time()
    timeout = 630
    safety_margin = 10
    
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _compute_ratio(best_P)
    
    # Strategy 1: Grid search over the quadratic family parameters
    # a controls spread along x-y (we want this large -> small a)
    # b controls spread along x+y (we want this small -> large b)
    # cx, cy control centering
    
    # Log-scale for a and b to cover wide range
    # a in [0.01, 10], b in [0.01, 10]
    # If a is small, x-y is wide. If b is large, x+y is narrow.
    
    a_vals = np.logspace(-2, 1, 30)
    b_vals = np.logspace(-2, 1, 30)
    
    # Centers: x-y can be -9..9, x+y can be -9..9 (shifted 0..18)
    # Let's search a few center points
    cx_vals = np.linspace(-4, 4, 9)
    cy_vals = np.linspace(-4, 4, 9)
    
    # To save time, we'll do a coarse grid first
    coarse_a = a_vals[::5]
    coarse_b = b_vals[::5]
    coarse_cx = cx_vals[::3]
    coarse_cy = cy_vals[::3]
    
    for a in coarse_a:
        for b in coarse_b:
            for cx in coarse_cx:
                for cy in coarse_cy:
                    if time.time() - start_time > timeout - safety_margin:
                        break
                    P = _generate_quadratic_distribution(a, b, cx, cy)
                    score = _compute_ratio(P)
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
    
    # Refine around the best found parameters
    # We need to track which params gave the best score
    # For simplicity, we will run a finer grid around the center if the best score came from center,
    # or just run a global fine grid if time permits.
    
    # Let's do a fine grid over a, b for a few likely centers (0,0) and near best
    # Since we didn't track best params, we'll just run a fine grid on a, b with center 0,0
    # and a few other centers if time allows.
    
    fine_a = np.logspace(-2, 1, 100)
    fine_b = np.logspace(-2, 1, 100)
    
    # We will iterate through a, b and check a few centers
    centers_to_check = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    
    for a in fine_a:
        if time.time() - start_time > timeout - safety_margin - 5:
            break
        for b in fine_b:
            for cx, cy in centers_to_check:
                P = _generate_quadratic_distribution(a, b, cx, cy)
                score = _compute_ratio(P)
                if score > best_score:
                    best_score = score
                    best_P = P.copy()
    
    # Strategy 2: Try the specific "anti-diagonal" line approximation with Gaussian smoothing
    # This is covered by the quadratic family with small 'a' and large 'b'
    
    # Strategy 3: Local polish on the best P found using small random perturbations
    # This helps if the parametric family is slightly off
    if time.time() - start_time < timeout - safety_margin:
        current_P = best_P.copy()
        current_score = best_score
        np.random.seed(42)
        
        # Simple hill climbing on the 100-dim space, but very limited steps
        # Only if we have significant time left
        if timeout - (time.time() - start_time) > 20:
            step_size = 0.05
            for _ in range(500):
                if time.time() - start_time > timeout - safety_margin:
                    break
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                new_P = current_P.copy()
                new_P[i, j] += delta
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                new_score = _compute_ratio(new_P)
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
    
    return best_P
