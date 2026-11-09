"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve


def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Generate a dense grid of frequency components for the Sum-of-Sines kernel.
    # We use frequencies up to a certain max to capture variations in the bounded domains.
    # The input ranges are generally [1, 5] or [1, 10], so frequencies of 0.1 to ~3.0 should cover
    # most periodic/non-periodic smooth functions well.
    
    # Create a grid of frequencies for each dimension.
    # To keep the number of basis functions manageable but expressive, we use a moderate grid.
    # Let's use frequencies from 0.2 to 4.0 in steps of 0.2 for each dimension.
    # This gives about 20 frequencies per dimension.
    # For d dimensions, the total number of basis functions will be the sum over all combinations.
    # Actually, a sum-of-sines kernel usually means K(x, x') = sum_i sum_j c_ij sin(f_i x_j) sin(f_k x_l) ...
    # A simpler and effective approach is to use a feature map:
    # phi(x) = [1, sin(f_1 x_1), cos(f_1 x_1), ..., sin(f_m x_d), cos(f_m x_d), ...]
    # But to capture interactions, we might need products. However, KRR with a good kernel handles this.
    # Let's use a kernel: K(x, x') = sum_{j=1}^d sum_{f in F} [sin(f x_j) sin(f x'_j) + cos(f x_j) cos(f x'_j)]
    # This is equivalent to using features: for each dim j and freq f, sin(f x_j) and cos(f x_j).
    # This is a linear combination of sine and cosine waves for each input dimension.
    # This captures additive nonlinearities. For interactions, we might need cross terms.
    # Let's add cross terms for pairs of dimensions: sin(f x_i) cos(g x_j) etc.
    # To keep it efficient, let's start with a moderate number of frequencies and include cross terms for small d.
    
    # Frequencies: let's use a logarithmic or linear spacing.
    # Given ranges up to 10, a max frequency of ~5 should capture up to 2-3 periods.
    freqs = np.arange(0.2, 5.0, 0.2)  # 24 frequencies
    n_freqs = len(freqs)
    
    # Build the feature matrix for training data.
    # Features:
    # 1. Bias term
    # 2. For each dimension j and each frequency f: sin(f * x_j), cos(f * x_j)
    # 3. For each pair of dimensions (i, j) with i < j and each pair of frequencies (f, g):
    #    sin(f * x_i) * cos(g * x_j), cos(f * x_i) * sin(g * x_j), etc.
    #    To keep the number of features reasonable, we'll limit cross terms.
    
    # Let's first try with just single-dimension sine/cosine features.
    # If d is large, the number of features is d * 2 * n_freqs.
    # For d=3, that's 3 * 2 * 24 = 144 features. Plus bias = 145.
    # This is manageable.
    
    # Build features for X_train
    features = [np.ones(n)]
    for j in range(d):
        for f in freqs:
            features.append(np.sin(f * X_train[:, j]))
            features.append(np.cos(f * X_train[:, j]))
    
    # Add cross terms for pairs of dimensions to capture interactions.
    # For each pair (i, j) with i < j, add sin(f_i x_i) * cos(f_j x_j) and cos(f_i x_i) * sin(f_j x_j)
    # and maybe sin*sin and cos*cos. Let's add a few cross terms.
    # To limit the number of features, we'll use a subset of frequencies for cross terms.
    # Let's use every other frequency for cross terms to keep it manageable.
    cross_freqs = freqs[::2]  # 12 frequencies
    for i in range(d):
        for j in range(i + 1, d):
            for f in cross_freqs:
                for g in cross_freqs:
                    features.append(np.sin(f * X_train[:, i]) * np.cos(g * X_train[:, j]))
                    features.append(np.cos(f * X_train[:, i]) * np.sin(g * X_train[:, j]))
                    features.append(np.sin(f * X_train[:, i]) * np.sin(g * X_train[:, j]))
                    features.append(np.cos(f * X_train[:, i]) * np.cos(g * X_train[:, j]))
    
    X_feat = np.column_stack(features)
    
    # Kernel Ridge Regression: solve (X_feat X_feat^T + alpha * I) w = y
    # Or directly solve the regularized least squares: (X^T X + alpha I) w = X^T y
    # Let's use the direct formulation for efficiency.
    alpha = 1e-3  # Regularization parameter
    XtX = X_feat.T @ X_feat
    Xty = X_feat.T @ y_train
    w = solve(XtX + alpha * np.eye(X_feat.shape[1]), Xty, assume_a='pos')
    
    def predict(X):
        n_test = X.shape[0]
        features_test = [np.ones(n_test)]
        for j in range(d):
            for f in freqs:
                features_test.append(np.sin(f * X[:, j]))
                features_test.append(np.cos(f * X[:, j]))
        for i in range(d):
            for j in range(i + 1, d):
                for f in cross_freqs:
                    for g in cross_freqs:
                        features_test.append(np.sin(f * X[:, i]) * np.cos(g * X[:, j]))
                        features_test.append(np.cos(f * X[:, i]) * np.sin(g * X[:, j]))
                        features_test.append(np.sin(f * X[:, i]) * np.sin(g * X[:, j]))
                        features_test.append(np.cos(f * X[:, i]) * np.cos(g * X[:, j]))
        X_feat_test = np.column_stack(features_test)
        return X_feat_test @ w
    
    return predict
# EVOLVE-BLOCK-END
