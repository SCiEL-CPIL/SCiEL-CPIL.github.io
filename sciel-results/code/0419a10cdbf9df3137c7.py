"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import interp1d
import time


def fit(X_train, y_train):
    """Fit using local polynomial regression (LOESS) for 1D data,
    with a global polynomial fallback for multi-dimensional inputs."""
    t0 = time.time()
    deadline = 320.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # For 1D inputs, use LOESS-style local regression
    if d == 1:
        x = X[:, 0].copy()
        # Sort by x for efficient neighbor lookup
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Try several bandwidth fractions via leave-one-out CV
        best_score = np.inf
        best_bw = 0.3
        n_samples = len(x_sorted)

        # Candidate bandwidths (fraction of range)
        candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

        for bw_frac in candidates:
            if time.time() > deadline:
                break
            # Number of neighbors
            k = max(5, int(bw_frac * n_samples))
            k = min(k, n_samples)

            # Compute LOO prediction scores
            # Use vectorized approach: for each point, fit local quadratic to k nearest neighbors
            # For speed, use a sliding window approximation
            scores = np.zeros(n_samples)
            for i in range(n_samples):
                # Find k nearest neighbors (excluding self for LOO)
                dists = np.abs(x_sorted - x_sorted[i])
                dists[i] = np.inf  # exclude self
                idx = np.argpartition(dists, k - 1)[:k]
                if len(idx) < 3:
                    continue
                xi = x_sorted[idx]
                yi = y_sorted[idx]
                # Local quadratic fit
                A = np.column_stack([np.ones_like(xi), xi, xi**2])
                try:
                    w_local, *_ = np.linalg.lstsq(A, yi, rcond=None)
                    pred = A @ w_local
                    scores[i] = np.mean((pred - yi)**2)
                except:
                    scores[i] = np.inf

            mean_score = np.nanmean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_bw = bw_frac

        # Build final LOESS predictor
        final_k = max(5, int(best_bw * n_samples))
        final_k = min(final_k, n_samples)

        def predict_loess(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            xt = X_test[:, 0]
            preds = np.zeros(len(xt))
            for i in range(len(xt)):
                dists = np.abs(x_sorted - xt[i])
                idx = np.argpartition(dists, min(final_k, n_samples) - 1)[:min(final_k, n_samples)]
                xi = x_sorted[idx]
                yi = y_sorted[idx]
                A = np.column_stack([np.ones_like(xi), xi, xi**2])
                try:
                    w_local, *_ = np.linalg.lstsq(A, yi, rcond=None)
                    preds[i] = A[0] @ w_local if len(xi) > 0 else np.mean(y)
                except:
                    preds[i] = np.mean(y)
            return preds

        return predict_loess

    else:
        # Multi-dimensional fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)

        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, d)
            return features(X_test) @ w

        return predict_poly

# EVOLVE-BLOCK-END
