# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_score(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    n = len(support)
    if n == 0:
        return 0.0
    # Normalize weights
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    s = support.astype(np.float64)
    
    # Compute all pairwise sums s_i + s_j
    # For efficiency, use broadcasting but be careful with memory
    # n^2 pairwise sums
    if n > 500:
        # Chunked computation to avoid memory issues
        P = 0.0
        chunk_size = max(1, 1000000 // n)
        for k_start in range(0, n, chunk_size):
            k_end = min(k_start + chunk_size, n)
            sk = s[k_start:k_end]
            wk = w[k_start:k_end]
            # For each k in this chunk, compute sum over l of w_l * (sum over i,j with s_i+s_j < 2*s_l - s_k)
            for k_idx in range(k_start, k_end):
                target = 2.0 * s[k_idx] - s[k_idx]  # = s[k_idx]
                # We need sum over l of w_l * F(2*s_l - s_k) where F(t) = P[X1+X2 < t]
                # Actually: P = sum_{k,l} w_k w_l * P[X1+X2 < 2*s_l - s_k]
                # Let's compute this more carefully
                pass
        # This approach is getting complicated. Let's use a vectorized approach for moderate n.
        # For large n, we need a smarter approach.
    
    # Vectorized approach for n up to ~2000
    # P = sum_{k,l} w_k * w_l * P[X1+X2 < 2*s_l - s_k]
    # Let's compute pairwise sums first
    ss = s[:, None] + s[None, :]  # n x n
    ww = w[:, None] * w[None, :]  # n x n
    
    # For each (k, l), we need sum_{i,j: ss[i,j] < 2*s[l] - s[k]} ww[i,j]
    # This is a 4D problem. Let's think differently.
    
    # Alternative: P = E[P[X1+X2 < 2*X4 - X3 | X3, X4]]
    # = sum_{k,l} w_k w_l * F(2*s_l - s_k)
    # where F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i w_j
    
    # Precompute all pairwise sums and their weights
    # For each possible threshold t = 2*s_l - s_k, we need F(t)
    
    # Efficient approach: sort pairwise sums, then for each threshold, binary search
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    # Sort pair_sums
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # Now for each (k, l), threshold = 2*s_l - s_k
    # Find number of pair sums < threshold using binary search
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find count of pair sums strictly less than threshold
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                prob = cum_weights[idx - 1]
            else:
                prob = 0.0
            P += w[k] * w[l] * prob
    
    return P


def _compute_score_fast(support, weights):
    """Faster computation using sorting and vectorization."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    s = support.astype(np.float64)
    
    # Compute all pairwise sums
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # For each (k, l), threshold = 2*s_l - s_k
    # We can vectorize this
    # Create matrix of thresholds: shape (n, n) where entry [k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n_k, n_l)
    
    # For each threshold, find the cumulative weight of pair sums < threshold
    # Use searchsorted
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    # Get cumulative weights at idx-1 (or 0 if idx=0)
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k * w_l * probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a mixture distribution that places mass at 0 and at a large value,
    with carefully chosen proportions to maximize the probability.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Initial baseline: uniform on {0, 1, ..., 31}
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    best_score = _compute_score_fast(support_base, weights_base)
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    
    # Try various structured distributions
    candidates = []
    
    # 1. Two-point distributions: mass p at 0, mass (1-p) at M
    for M in [1, 2, 3, 5, 10, 20, 50, 100]:
        for p in np.linspace(0.01, 0.99, 99):
            support = np.array([0.0, M], dtype=np.float64)
            weights = np.array([p, 1.0 - p], dtype=np.float64)
            candidates.append((support, weights))
    
    # 2. Three-point distributions: mass p1 at 0, p2 at a, p3 at b
    for a in [1, 2, 5]:
        for b in [2, 5, 10, 20]:
            if a >= b:
                continue
            for p1 in np.linspace(0.05, 0.8, 16):
                for p2 in np.linspace(0.05, 0.8, 16):
                    p3 = 1.0 - p1 - p2
                    if p3 < 0.01:
                        continue
                    support = np.array([0.0, a, b], dtype=np.float64)
                    weights = np.array([p1, p2, p3], dtype=np.float64)
                    candidates.append((support, weights))
    
    # 3. Geometric-like distributions
    for r in [0.5, 0.7, 0.8, 0.9]:
        for max_n in [5, 10, 20]:
            support = np.arange(max_n, dtype=np.float64)
            weights = r ** support
            weights /= weights.sum()
            candidates.append((support, weights))
    
    # 4. Uniform on sparse sets
    for n_sparse in [4, 6, 8, 10]:
        for gap in [1, 2, 3, 5]:
            support = np.arange(n_sparse) * gap
            weights = np.ones(n_sparse) / n_sparse
            candidates.append((support, weights))
    
    # Evaluate all candidates
    for support, weights in candidates:
        if time.time() > deadline - 30.0:
            break
        try:
            score = _compute_score_fast(support, weights)
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = weights.copy()
        except Exception:
            continue
    
    # Try local optimization on the best candidate found
    if time.time() < deadline - 60.0:
        # Gradient-free local search
        current_support = best_support.copy()
        current_weights = best_weights.copy()
        current_score = best_score
        
        n = len(current_support)
        step_sizes = [0.1, 0.05, 0.01, 0.005]
        
        for step_size in step_sizes:
            improved = True
            while improved and time.time() < deadline - 60.0:
                improved = False
                for i in range(n):
                    # Try perturbing support
                    for delta in [-step_size, step_size]:
                        new_support = current_support.copy()
                        new_support[i] += delta
                        if new_support[i] < -1e-9:
                            continue
                        try:
                            score = _compute_score_fast(new_support, current_weights)
                            if score > current_score + 1e-10:
                                current_support = new_support
                                current_score = score
                                improved = True
                        except Exception:
                            pass
                    
                    # Try perturbing weights
                    for delta in [-step_size, step_size]:
                        new_weights = current_weights.copy()
                        new_weights[i] += delta
                        if np.any(new_weights < -1e-9):
                            continue
                        if new_weights.sum() <= 1e-9:
                            continue
                        try:
                            score = _compute_score_fast(current_support, new_weights)
                            if score > current_score + 1e-10:
                                current_weights = new_weights
                                current_score = score
                                improved = True
                        except Exception:
                            pass
                
                if current_score > best_score:
                    best_score = current_score
                    best_support = current_support.copy()
                    best_weights = current_weights.copy()
        
        # Final validation
        if best_score > 0 and len(best_support) >= 1:
            # Ensure validity
            if np.any(best_support < -1e-9):
                best_support = np.clip(best_support, 0, None)
            if np.any(best_weights < -1e-9):
                best_weights = np.clip(best_weights, 0, None)
            if best_weights.sum() <= 1e-9:
                # Fallback to baseline
                best_support = support_base
                best_weights = weights_base
                best_score = _compute_score_fast(best_support, best_weights)
    
    # Normalize weights to sum to 1 for cleanliness
    total_w = best_weights.sum()
    if total_w > 1e-9:
        best_weights = best_weights / total_w
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
