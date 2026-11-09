"""Symbolic regression with Sum-of-Sines Kernel Ridge Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs to [0, 1] range or similar to make frequency selection more uniform?
    # Or just use raw values. The ranges are known roughly.
    # Let's use raw values but ensure frequencies are appropriate.
    # Ranges: [1,3], [1,2], [1,5], [1,10].
    # A frequency of 1.0 gives period ~6.28.
    # For range [1,10], we need higher frequencies to capture details.
    # Let's use a log-spaced frequency grid.
    
    freqs = np.logspace(-1, np.log10(20), num=12) # 0.1 to 20.0
    
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        cols = [np.ones(len(Xr))]
        
        # Single variable basis: sin/cos for each freq and each dim
        for j in range(d):
            for f in freqs:
                cols.append(np.sin(f * Xr[:, j]))
                cols.append(np.cos(f * Xr[:, j]))
        
        # Interaction terms: Use only the lowest few frequencies to avoid explosion
        # We use the first 2 frequencies for interactions
        inter_freqs = freqs[:2]
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    for f1 in inter_freqs:
                        for f2 in inter_freqs:
                            # sin(f1*xi) * sin(f2*xj)
                            cols.append(np.sin(f1 * Xr[:, i]) * np.sin(f2 * Xr[:, j]))
                            # cos(f1*xi) * cos(f2*xj)
                            cols.append(np.cos(f1 * Xr[:, i]) * np.cos(f2 * Xr[:, j]))
                            # sin(f1*xi) * cos(f2*xj)
                            cols.append(np.sin(f1 * Xr[:, i]) * np.cos(f2 * Xr[:, j]))
                            # cos(f1*xi) * sin(f2*xj)
                            cols.append(np.cos(f1 * Xr[:, i]) * np.sin(f2 * Xr[:, j]))
        
        return np.column_stack(cols)

    F = make_features(X)
    
    # Ridge Regression
    y_mean = y.mean()
    y_centered = y - y_mean
    
    best_w = None
    best_score = np.inf
    
    # Grid of lambda values
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    for lam in lambdas:
        if time.time() > DEADLINE - 10:
            break
        A = F.T @ F
        b = F.T @ y_centered
        # Add regularization
        A += lam * np.eye(F.shape[1])
        
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y_centered, rcond=None)[0]
            
        pred = F @ w + y_mean
        score = np.mean((pred - y) ** 2)
        
        if score < best_score:
            best_score = score
            best_w = w

    if best_w is None:
        return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)

    def predict(Xraw):
        Fr = make_features(Xraw)
        return Fr @ best_w + y_mean

    return predict
# EVOLVE-BLOCK-END
