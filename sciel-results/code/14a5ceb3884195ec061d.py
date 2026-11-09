import numpy as np
from scipy.optimize import minimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    timeout = 600.0
    margin = 30.0  # Safety margin
    
    def compute_probability(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] exactly.
        Vectorized approach:
        1. Compute all pairwise sums S2 = s_i + s_j and their weights W2 = w_i w_j.
        2. Sort S2.
        3. Compute cumulative sum of W2.
        4. For each k, l, threshold T = 2*s_l - s_k.
        5. Use searchsorted to find count of sums < T.
        6. Sum w_k w_l * cum_weight[count].
        """
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        
        # Compute all pairwise sums and weights
        # Sums: s_i + s_j for all i, j
        s_i = support[:, None]
        s_j = support[None, :]
        sums = s_i + s_j
        
        # Weights: w_i * w_j
        w_i = w[:, None]
        w_j = w[None, :]
        weights_pair = w_i * w_j
        
        # Flatten and sort sums
        sums_flat = sums.ravel()
        weights_flat = weights_pair.ravel()
        
        # Sort sums and compute cumulative weights
        order = np.argsort(sums_flat, kind='stable')
        sums_sorted = sums_flat[order]
        weights_sorted = weights_flat[order]
        cum_weights = np.cumsum(weights_sorted)
        
        # For each k, l, we need sum of w_i w_j where s_i + s_j < 2*s_l - s_k
        # Thresholds: T_kl = 2*s_l - s_k
        # We can vectorize this over k and l
        # T is an n x n matrix
        T = 2.0 * support[None, :] - support[:, None]
        
        # For each threshold T_kl, find index in sums_sorted where sums < T_kl
        # searchsorted with side='left' gives the index of the first element >= T_kl
        # So the number of elements < T_kl is idx
        idx = np.searchsorted(sums_sorted, T, side='left')
        
        # We need cum_weights[idx-1] if idx > 0, else 0
        # Initialize result array for inner sums
        inner_sums = np.zeros((n, n))
        
        # Mask for idx > 0
        mask = idx > 0
        inner_sums[mask] = cum_weights[idx[mask] - 1]
        
        # Total probability: sum over k, l of w_k w_l * inner_sums[k, l]
        # w_k w_l is outer product of w and w
        prob_matrix = np.outer(w, w) * inner_sums
        return float(np.sum(prob_matrix))

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Candidate supports
    # These are chosen to be "spread out" to maximize the chance that X1+X2+X3 is small
    # while X4 is large.
    candidates = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 1.0, 4.0]),
        np.array([0.0, 1.0, 5.0]),
        np.array([0.0, 1.0, 6.0]),
        np.array([0.0, 1.0, 8.0]),
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0, 1.0, 15.0]),
        np.array([0.0, 2.0, 5.0]),
        np.array([0.0, 2.0, 8.0]),
        np.array([0.0, 3.0, 10.0]),
        np.array([0.0, 1.0, 3.0, 9.0]),
        np.array([0.0, 1.0, 4.0, 13.0]),
        np.array([0.0, 1.0, 5.0, 16.0]),
        np.array([0.0, 2.0, 6.0, 20.0]),
        np.array([0.0, 1.0, 3.0, 7.0, 15.0]),
        np.array([0.0, 1.0, 4.0, 9.0, 20.0]),
        np.array([0.0, 1.0, 5.0, 11.0, 25.0]),
        np.array([0.0, 2.0, 7.0, 16.0, 30.0]),
    ]
    
    # Add some random geometric-like supports
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = rng.integers(3, 6)
        # Geometric progression with random ratio
        ratio = rng.uniform(1.5, 4.0)
        s = np.array([ratio ** i for i in range(n)])
        # Normalize to start at 0? No, support must be >= 0.
        # We can shift it? No, s_i >= 0.
        # Let's just use the geometric progression starting from 1 or 0.
        # If we start from 0, it's not geometric.
        # Let's try: 0, 1, r, r^2 ...
        s = np.concatenate(([0.0, 1.0], ratio ** np.arange(1, n-1)))
        s = np.sort(s)
        candidates.append(s)

    def objective(weights):
        # Weights are already positive due to bounds in L-BFGS-B or transformation
        # Here we assume weights are passed as they are, and we handle normalization inside compute_probability
        # But for optimizer, it's better to optimize unconstrained or bounded.
        # Let's use L-BFGS-B with bounds [1e-6, 1]
        return -compute_probability(current_support, weights)

    for support in candidates:
        if time.time() > start_time + timeout - margin:
            break
            
        n = len(support)
        current_support = support
        
        # Initial guess: uniform weights
        initial_weights = np.ones(n) / n
        
        # Optimize weights
        try:
            bounds = [(1e-6, 1.0) for _ in range(n)]
            
            res = minimize(
                objective,
                initial_weights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            best_w = res.x
            # Normalize to ensure sum is 1 (L-BFGS-B doesn't enforce sum=1, just bounds)
            # The compute_probability function normalizes, so this is fine.
            # But let's normalize for the final output anyway.
            best_w = best_w / np.sum(best_w)
            
            score = compute_probability(support, best_w)
            
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = best_w.copy()
                
        except Exception as e:
            pass

    # If no optimization worked or score is bad, fallback
    if best_support is None or best_score < 0.5:
        best_support = np.array([0.0, 1.0, 3.0])
        best_weights = np.array([0.5, 0.3, 0.2])
        best_weights = best_weights / np.sum(best_weights)
        s = compute_probability(best_support, best_weights)
        if s > best_score:
            best_score = s
            best_support = best_support.copy()
            best_weights = best_weights.copy()

    return best_support, best_weights
