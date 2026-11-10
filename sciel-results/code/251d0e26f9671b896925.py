# EVOLVE-BLOCK-START
import numpy as np
import time


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s kill

    def compute_P(support, weights):
        """
        Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pair sums.
        P = sum_{l,k} w_l w_k * (sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j)
        """
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support

        # Precompute all pairwise sums and their weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        pair_weights = w[i_idx] * w[j_idx]

        # Sort pair sums for efficient threshold queries
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)

        P = 0.0
        for l in range(n):
            target = 2.0 * s[l]
            thresholds = target - s
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            P += w[l] * np.sum(w * inner_sums)

        return P

    def compute_gradient(support, weights):
        """
        Compute gradient of P w.r.t weights.
        P = sum_{l,k,i,j: s_i+s_j < 2s_l-s_k} w_l w_k w_i w_j
        dP/dw_m = 4 * sum_{l,k,i,j: cond} (delta_ml + delta_mk + delta_mi + delta_mj) w_l w_k w_i w_j
        = 4 * [sum_{k,i,j: s_i+s_j < 2s_m-s_k} w_m w_k w_i w_j
             + sum_{l,i,j: s_i+s_j < 2s_l-s_m} w_l w_m w_i w_j
             + sum_{l,k,j: s_m+s_j < 2s_l-s_k} w_l w_k w_m w_j
             + sum_{l,k,i: s_i+s_m < 2s_l-s_k} w_l w_k w_i w_m]
        By symmetry, all four terms are equal, so:
        dP/dw_m = 4 * 4 * sum_{k,i,j: s_i+s_j < 2s_m-s_k} w_m w_k w_i w_j
                = 16 * w_m * sum_{k,i,j: s_i+s_j < 2s_m-s_k} w_k w_i w_j
        """
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return np.zeros(n)
        w = weights / w_sum
        s = support

        # Precompute pair sums and weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        pair_weights = w[i_idx] * w[j_idx]

        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)

        grad = np.zeros(n)
        for m in range(n):
            # Compute sum_{k,i,j: s_i+s_j < 2s_m-s_k} w_k w_i w_j
            # = sum_k w_k * (sum_{i,j: s_i+s_j < 2s_m-s_k} w_i w_j)
            inner_sum = 0.0
            for k in range(n):
                threshold = 2.0 * s[m] - s[k]
                idx = np.searchsorted(sorted_sums, threshold, side='left')
                if idx > 0:
                    inner_sum += w[k] * cum_weights[idx - 1]
            grad[m] = 16.0 * w[m] * inner_sum

        return grad

    def projected_gradient_ascent(support, w_init, n_iter=5000, lr=0.01):
        """
        Projected gradient ascent to optimize weights for fixed support.
        """
        n = len(support)
        w = w_init.copy()
        best_w = w.copy()
        best_score = compute_P(support, w)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            # Compute gradient
            grad = compute_gradient(support, w)

            # Gradient ascent step
            w_new = w + lr * grad

            # Project onto simplex {w >= 0, sum(w) = 1}
            w_new = np.maximum(w_new, 0.0)
            s = np.sum(w_new)
            if s < 1e-15:
                w_new = np.ones(n) / n
            else:
                w_new = w_new / s

            score_new = compute_P(support, w_new)

            if score_new > best_score:
                best_score = score_new
                best_w = w_new.copy()

            w = w_new

        return best_w, best_score

    def simulated_annealing_weights(support, w_init, n_iter=3000, T_start=0.01, T_end=0.001):
        """Simulated annealing to optimize weights for fixed support."""
        n = len(support)
        w = w_init.copy()
        best_w = w.copy()
        best_score = compute_P(support, w)

        T = T_start
        cooling = (T_end / T_start) ** (1.0 / n_iter)

        for it in range(n_iter):
            T *= cooling
            if time.time() > deadline - 5.0:
                break

            # Random perturbation: move mass from one index to another
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = np.random.uniform(-0.1, 0.1)
            w_new = w.copy()
            w_new[i] -= delta
            w_new[j] += delta
            w_new = np.maximum(w_new, 0.0)
            s = np.sum(w_new)
            if s < 1e-15:
                continue
            w_new = w_new / s

            score_new = compute_P(support, w_new)
            delta_score = score_new - best_score

            if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
                w = w_new
                if score_new > best_score:
                    best_score = score_new
                    best_w = w_new.copy()

        return best_w, best_score

    best_score = 0.0
    best_support = None
    best_weights = None

    # Try various support configurations
    support_configs = [
        np.arange(5, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        np.arange(10, dtype=np.float64),
        np.arange(12, dtype=np.float64),
        np.arange(15, dtype=np.float64),
        np.arange(20, dtype=np.float64),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 25], dtype=np.float64),
        np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], dtype=np.float64),
        np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.float64),
        np.array([0, 0.5, 1, 2, 3, 5, 8, 13, 21], dtype=np.float64),
        np.array([0, 1, 2, 4, 7, 11, 16, 22], dtype=np.float64),
    ]

    for support in support_configs:
        if time.time() > deadline - 30.0:
            break

        n = len(support)
        w0 = np.ones(n) / n

        # First try projected gradient ascent
        w_pga, score_pga = projected_gradient_ascent(support, w0, n_iter=3000, lr=0.01)
        if score_pga > best_score:
            best_score = score_pga
            best_support = support.copy()
            best_weights = w_pga.copy()

        # Then try simulated annealing starting from the best weights found
        w_sa, score_sa = simulated_annealing_weights(support, w_pga, n_iter=2000)
        if score_sa > best_score:
            best_score = score_sa
            best_support = support.copy()
            best_weights = w_sa.copy()

    # If no better found, fall back to uniform n=32
    if best_score == 0.0 or best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64)
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64)
        elif np.sum(best_weights) <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64)

    return best_support, best_weights


# EVOLVE-BLOCK-END
