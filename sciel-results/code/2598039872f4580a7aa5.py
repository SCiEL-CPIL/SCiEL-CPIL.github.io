# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _shannon_entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = np.array(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    P = np.clip(P, 0, None)
    total = P.sum()
    if total < 1e-9:
        return 0.0
    P = P / total
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    # D = H(x - y)
    vals_dy = X - Y
    unique_dy, counts_dy = np.unique(vals_dy, return_counts=True)
    probs_dy = counts_dy / counts_dy.sum()
    H_dy = _shannon_entropy(probs_dy)
    
    # H0 = H(x)
    vals_x = X
    unique_x, counts_x = np.unique(vals_x, return_counts=True)
    probs_x = counts_x / counts_x.sum()
    H_x = _shannon_entropy(probs_x)
    
    # H1 = H(x + y)
    vals_xy = X + Y
    unique_xy, counts_xy = np.unique(vals_xy, return_counts=True)
    probs_xy = counts_xy / counts_xy.sum()
    H_xy = _shannon_entropy(probs_xy)
    
    # H2 = H(x + 2y)
    vals_x2y = X + 2 * Y
    unique_x2y, counts_x2y = np.unique(vals_x2y, return_counts=True)
    probs_x2y = counts_x2y / counts_x2y.sum()
    H_x2y = _shannon_entropy(probs_x2y)
    
    M = max(H_x, H_xy, H_x2y)
    if M < 1e-9:
        return 0.0
    
    return H_dy / M


def _optimize_weights_for_support(support_mask, P_init, max_iter=5000, lr=0.1):
    """Optimize weights on a given support mask to maximize D/M."""
    n = 10
    P = P_init.copy()
    P[~support_mask] = 0
    
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Simple gradient-free optimization: random perturbations with annealing
    np.random.seed(42)
    for iteration in range(max_iter):
        # Random perturbation
        delta = np.random.randn(n, n) * lr
        delta[~support_mask] = 0
        P_new = P + delta
        P_new = np.clip(P_new, 0, None)
        
        # Normalize
        total = P_new.sum()
        if total < 1e-9:
            continue
        P_new = P_new / total
        
        ratio = _compute_ratio(P_new)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_new.copy()
            P = P_new
        else:
            # Sometimes accept worse solutions (simulated annealing)
            if np.random.rand() < 0.1:
                P = P_new
        
        # Decrease learning rate
        lr = max(0.01, 0.1 * (0.999 ** iteration))
    
    return best_P, best_ratio


def _try_pattern(pattern_type, P_init):
    """Try a specific support pattern and optimize weights."""
    n = 10
    support_mask = np.zeros((n, n), dtype=bool)
    
    if pattern_type == "uniform":
        support_mask[:] = True
    elif pattern_type == "diagonal_x_minus_y":
        # Focus on cells that spread out x-y values
        for y in range(n):
            for x in range(n):
                support_mask[y, x] = True
    elif pattern_type == "sparse_diagonals":
        # Use only certain diagonals to concentrate x-y mass
        for d in range(-9, 10, 2):  # Every other diagonal
            for y in range(n):
                x = y + d
                if 0 <= x < n:
                    support_mask[y, x] = True
    elif pattern_type == "corners":
        # Use corner cells
        for y in [0, 1, 8, 9]:
            for x in [0, 1, 8, 9]:
                support_mask[y, x] = True
    elif pattern_type == "anti_diagonal":
        # Anti-diagonal: x + y = constant
        for y in range(n):
            x = n - 1 - y
            support_mask[y, x] = True
    elif pattern_type == "cross":
        # Cross pattern: middle row and column
        support_mask[4, :] = True
        support_mask[5, :] = True
        support_mask[:, 4] = True
        support_mask[:, 5] = True
    elif pattern_type == "sparse_random":
        # Random sparse pattern
        np.random.seed(123)
        support_mask = np.random.rand(n, n) < 0.3
        # Ensure at least some cells
        if support_mask.sum() < 5:
            support_mask[0, 0] = True
            support_mask[9, 9] = True
    elif pattern_type == "x_minus_y_concentrated":
        # Concentrate on cells where x-y takes extreme values
        for y in range(n):
            for x in range(n):
                diff = abs(x - y)
                if diff >= 7:  # Extreme differences
                    support_mask[y, x] = True
    elif pattern_type == "banded":
        # Banded pattern: x close to y
        for y in range(n):
            for x in range(n):
                if abs(x - y) <= 2:
                    support_mask[y, x] = True
    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        for y in range(n):
            for x in range(n):
                if (x + y) % 2 == 0:
                    support_mask[y, x] = True
    
    if support_mask.sum() == 0:
        return P_init, 0.0
    
    # Initialize weights uniformly on support
    P_init_support = np.zeros((n, n))
    P_init_support[support_mask] = 1.0 / support_mask.sum()
    
    P_opt, ratio = _optimize_weights_for_support(support_mask, P_init_support, max_iter=3000, lr=0.05)
    return P_opt, ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_ratio = _compute_ratio(P_uniform)
    
    # Try various patterns
    patterns = [
        "uniform",
        "diagonal_x_minus_y",
        "sparse_diagonals",
        "corners",
        "anti_diagonal",
        "cross",
        "sparse_random",
        "x_minus_y_concentrated",
        "banded",
        "checkerboard",
    ]
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        P_candidate, ratio = _try_pattern(pattern, P_uniform)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_candidate.copy()
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total < 1e-9:
        # Fallback to uniform
        best_P = P_uniform
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
