"""Rewritten program for symbolic regression over 4 hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np


def fit(X_train, y_train):
    """
    Fit a flexible model using a rich set of basis functions and
    L1-regularized regression for feature selection.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def build_basis(X):
        cols = [np.ones(len(X))]
        
        # Polynomial terms up to degree 4
        for deg in range(1, 5):
            for i in range(d):
                cols.append(X[:, i] ** deg)
        
        # Cross terms
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(X[:, i] * X[:, j])
                    cols.append(X[:, i] ** 2 * X[:, j])
                    cols.append(X[:, i] * X[:, j] ** 2)
        
        # Trigonometric
        for i in range(d):
            cols.append(np.sin(X[:, i]))
            cols.append(np.cos(X[:, i]))
            cols.append(np.sin(2 * X[:, i]))
            cols.append(np.cos(2 * X[:, i]))
        
        # Exponential (bounded)
        for i in range(d):
            cols.append(np.exp(np.clip(X[:, i], -3, 3)))
            cols.append(np.exp(np.clip(-X[:, i], -3, 3)))
        
        # Logarithmic (for positive inputs)
        for i in range(d):
            log_term = np.zeros(len(X))
            pos_mask = X[:, i] > 1e-6
            log_term[pos_mask] = np.log(X[:, i][pos_mask])
            cols.append(log_term)
        
        # Inverse
        for i in range(d):
            inv_term = np.zeros(len(X))
            nonzero = np.abs(X[:, i]) > 1e-6
            inv_term[nonzero] = 1.0 / X[:, i][nonzero]
            cols.append(inv_term)
        
        basis = np.column_stack(cols)
        
        # Remove invalid columns
        valid_cols = []
        for j in range(basis.shape[1]):
            col = basis[:, j]
            if np.all(np.isfinite(col)) and np.std(col) > 1e-10:
                valid_cols.append(j)
        basis = basis[:, valid_cols]
        
        # Standardize non-constant columns
        if basis.shape[1] > 1:
            means = basis[:, 1:].mean(axis=0)
            stds = basis[:, 1:].std(axis=0)
            stds[stds < 1e-10] = 1.0
            basis[:, 1:] = (basis[:, 1:] - means) / stds
        
        return basis, (means if basis.shape[1] > 1 else None, stds if basis.shape[1] > 1 else None)
    
    # Build basis and save normalization params
    basis_train, (means, stds) = build_basis(X)
    
    # L1-regularized regression via iterative reweighted least squares
    A = basis_train
    n_feat = A.shape[1]
    
    # Initialize with ridge
    alpha = 1e-2
    A_reg = np.vstack([A, np.sqrt(alpha) * np.eye(n_feat)])
    y_reg = np.concatenate([y, np.zeros(n_feat)])
    w, *_ = np.linalg.lstsq(A_reg, y_reg, rcond=None)
    
    # IRLS for L1
    A_ta = A.T @ A
    A_ty = A.T @ y
    lam = 0.005
    eps = 1e-8
    
    for it in range(30):
        sign_w = np.sign(w)
        sign_w[np.abs(w) < eps] = 1.0
        M = A_ta + 2 * lam * np.diag(sign_w)
        try:
            w_new = np.linalg.solve(M, A_ty)
        except np.linalg.LinAlgError:
            w_new = w
        if np.linalg.norm(w_new - w) < 1e-7:
            w = w_new
            break
        w = w_new
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        basis_test, _ = build_basis(X_test)
        
        if basis_test.shape[1] != basis_train.shape[1]:
            # Fallback: simple polynomial regression
            def poly_features(X):
                cols = [np.ones(len(X))]
                for deg in range(1, 5):
                    for i in range(X.shape[1]):
                        cols.append(X[:, i] ** deg)
                return np.column_stack(cols)
            A_train = poly_features(X)
            A_test = poly_features(X_test)
            w_poly, *_ = np.linalg.lstsq(A_train, y, rcond=None)
            return A_test @ w_poly
        
        return basis_test @ w
    
    return predict
# EVOLVE-BLOCK-END
