"""Search-based symbolic regression with time budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Define a set of basis functions that can be applied to any X
    # Each element is a tuple (name, lambda_function)
    # We will generate features for each dimension and their pairwise products
    
    def get_basis_funcs():
        """Return a list of functions that take a 1D array and return a 1D array of values."""
        return [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x ** 2,
            lambda x: x ** 3,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(-x),
            lambda x: np.log1p(x),
            lambda x: np.abs(x),
        ]
    
    basis_funcs = get_basis_funcs()
    
    # Generate feature columns on training data
    # Feature 0 is always 1 (bias)
    # Features 1..d*len(basis_funcs) are single-variable basis functions
    # Features d*len(basis_funcs)+1 .. are pairwise products
    
    train_features = [np.ones(n)]
    feature_generators = [lambda X: np.ones(X.shape[0])]
    
    # Single variable features
    for j in range(d):
        xj = X_train[:, j]
        for f in basis_funcs:
            val = f(xj)
            # Check for numerical stability
            if not np.all(np.isfinite(val)):
                continue
            train_features.append(val)
            # Capture j and f in closure
            feature_generators.append(lambda X, j=j, f=f: f(X[:, j]))
    
    # Pairwise product features
    # Only include products if we have time and d is small enough
    # For d=1, products are just powers (already covered)
    # For d=2, 3, etc., we add x_i * x_j type terms
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and d > 1:
        for i in range(d):
            for j in range(i, d):
                xi = X_train[:, i]
                xj = X_train[:, j]
                for fi in basis_funcs:
                    for fj in basis_funcs:
                        # Skip some redundant or expensive combinations if needed
                        # But let's try a reasonable subset first
                        val_i = fi(xi)
                        val_j = fj(xj)
                        
                        if not np.all(np.isfinite(val_i)) or not np.all(np.isfinite(val_j)):
                            continue
                        
                        val = val_i * val_j
                        if not np.all(np.isfinite(val)):
                            continue
                        
                        train_features.append(val)
                        feature_generators.append(lambda X, i=i, j=j, fi=fi, fj=fj: fi(X[:, i]) * fj(X[:, j]))
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # If we ran out of time generating features, fall back to simple polynomial
        # But we likely finished generation fast. Let's proceed to fitting.
        pass

    if len(train_features) == 1:
        # Only bias term? Should not happen if d>=1
        return lambda X: np.full(X.shape[0], y_mean)

    feat_matrix = np.column_stack(train_features)
    
    # Solve least squares
    # Use rcond=None for stability
    try:
        w, residuals, rank, s = np.linalg.lstsq(feat_matrix, y_train, rcond=None)
    except Exception:
        # Fallback to mean if lstsq fails
        return lambda X: np.full(X.shape[0], y_mean)
    
    pred_train = feat_matrix @ w
    mse = np.mean((pred_train - y_train) ** 2)
    
    # Define the predict function
    def predict(X):
        # Build feature matrix for X
        cols = []
        for gen in feature_generators:
            col = gen(X)
            cols.append(col)
        feat_mat = np.column_stack(cols)
        return feat_mat @ w
    
    # Verify prediction on training set to ensure no errors
    try:
        _ = predict(X_train)
    except Exception:
        # If prediction fails, return a simple constant predictor
        return lambda X: np.full(X.shape[0], y_mean)
    
    return predict

# EVOLVE-BLOCK-END
