"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    x_min, x_max = np.min(X), np.max(X)
    x_range = x_max - x_min
    if x_range <= 1e-10:
        x_range = 1.0
        
    # Normalize X to [0, 1] for stable basis functions
    X_norm = (X - x_min) / x_range
    
    # Normalize Y for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Candidate Basis Functions
    # 1. Polynomials (degree 0-6)
    # 2. Fourier (low frequency sines/cosines)
    # 3. RBFs (Gaussian bumps at quantiles of X)
    
    def build_basis(X_val, rbf_centers, rbf_widths):
        X_flat = X_val.flatten()
        
        # Polynomials
        basis = [np.ones_like(X_flat)]
        for d in range(1, 7):
            basis.append((X_flat ** d))
            
        # Fourier (frequencies 1, 2, 3)
        for f in [1, 2, 3]:
            basis.append(np.sin(2 * np.pi * f * X_flat))
            basis.append(np.cos(2 * np.pi * f * X_flat))
            
        # RBFs
        for c, w in zip(rbf_centers, rbf_widths):
            basis.append(np.exp(-((X_flat - c) / w) ** 2))
            
        return np.column_stack(basis)

    # Define RBF centers and widths
    # Centers at 10 quantiles
    rbf_centers = np.quantile(X_norm, np.linspace(0.1, 0.9, 9))
    # Widths: 3 scales (wide, medium, narrow)
    rbf_widths = [x_range * 0.5, x_range * 0.1, x_range * 0.02]
    
    # Build the full training basis matrix
    # Note: We use normalized X for the basis, but we need to be careful with scaling in predict
    # Actually, let's just use the raw X but scale the RBF parameters accordingly?
    # No, it's better to stick to one normalized space for the basis construction to avoid overflow in polynomials.
    # However, the prediction must handle raw X.
    # So, inside predict, we will normalize X_pred using the same x_min, x_range.
    
    B_train = build_basis(X_norm, rbf_centers, rbf_widths)
    
    # We will search for the best regularization parameter lambda (alpha)
    # Objective: Minimize MSE on a validation split (or LOOCV if fast enough, but 200 points is small)
    # Let's use a simple 5-fold cross-validation to pick alpha.
    
    # Since n=200, we can afford a decent grid search.
    # Regularization: min ||B w - y||^2 + alpha * ||w||^2
    # Solution: w = (B^T B + alpha I)^-1 B^T y
    
    # Precompute B^T B to speed up
    BTB = B_train.T @ B_train
    
    # Grid of alpha values (log spaced)
    # Small alpha = interpolation, Large alpha = smoothing
    alphas = np.logspace(-4, 4, 20)
    
    best_alpha = 1.0
    best_val_score = np.inf
    
    # Simple K-Fold CV
    k = 5
    indices = np.arange(n)
    np.random.seed(42) # Fixed seed for determinism in CV split
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for alpha in alphas:
        val_errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            B_tr = B_train[train_idx]
            y_tr = y_norm[train_idx]
            B_val = B_train[val_idx]
            y_val = y_norm[val_idx]
            
            # Solve (B^T B + alpha I) w = B^T y
            # Note: BTB is for the whole set. For CV, we need BTB for the subset?
            # No, BTB depends on the data points used. So we must recompute or use a different strategy.
            # Recomputing BTB for 160 points x ~20 features is very fast.
            
            BTB_tr = B_tr.T @ B_tr
            BTy_tr = B_tr.T @ y_tr
            
            try:
                w = np.linalg.solve(BTB_tr + alpha * np.eye(B_train.shape[1]), BTy_tr)
                y_pred_val = B_val @ w
                mse = np.mean((y_pred_val - y_val) ** 2)
                val_errors.append(mse)
            except np.linalg.LinAlgError:
                val_errors.append(1e10)
        
        mean_val_score = np.mean(val_errors)
        if mean_val_score < best_val_score:
            best_val_score = mean_val_score
            best_alpha = alpha
            
    # Final fit with best alpha
    BTB_full = B_train.T @ B_train
    BTy_full = B_train.T @ y_norm
    
    try:
        w_final = np.linalg.solve(BTB_full + best_alpha * np.eye(B_train.shape[1]), BTy_full)
    except np.linalg.LinAlgError:
        # Fallback: simple linear regression on raw x
        w_final = np.zeros(B_train.shape[1])
        w_final[0] = np.mean(y_norm) # Intercept
        
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)
        # Normalize X_pred
        X_pred_norm = (X_pred - x_min) / x_range
        
        B_pred = build_basis(X_pred_norm, rbf_centers, rbf_widths)
        
        y_pred_norm = B_pred @ w_final
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
        
        return y_pred.flatten()

    return predict
# EVOLVE-BLOCK-END
