import numpy as np

def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(training_x)
    degree = 5
    
    # Handle edge cases
    if n == 0:
        return np.zeros(6)
    if n < degree + 1:
        # Fall back to lower degree
        deg_eff = min(degree, n - 1)
        try:
            # Use np.polyfit (descending powers) and reverse
            coeff_desc = np.polyfit(training_x, training_subgrid_tendency, deg_eff)
            coeff_asc = np.zeros(6)
            coeff_asc[:len(coeff_desc)] = coeff_desc[::-1]
            return coeff_asc
        except:
            return np.zeros(6)

    # Standardize x to improve numerical stability of Vandermonde matrix
    x_mean = np.mean(training_x)
    x_std = np.std(training_x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (training_x - x_mean) / x_std
    
    # Build standardized design matrix
    V = np.vander(x_norm, N=degree + 1, increasing=True)
    y = training_subgrid_tendency
    
    # Pre-compute Gram matrix G = V^T V and cross-product b = V^T y
    # This is constant for all lambda and all folds if we were doing full data, 
    # but for CV we need fold-specific G and b.
    # However, we can compute V^T V for the full set to estimate scale for lambda selection.
    G_full = V.T @ V
    trace_G = np.trace(G_full)
    scale = trace_G / (degree + 1) if trace_G > 0 else 1.0
    
    # Lambda selection
    # We want to cover a range of regularization strengths.
    # The eigenvalues of G_full give a sense of the scale.
    # Let's use logspace relative to the median eigenvalue or trace.
    lambdas = np.logspace(-10, 10, 50) * scale
    
    # K-fold CV
    k = 10
    if n < 2 * k:
        k = max(2, n // 2)
    
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, k)
    
    best_mse = np.inf
    best_lam = 0.0
    
    # For each lambda, we perform k-fold CV
    for lam in lambdas:
        mse_sum = 0.0
        for i in range(k):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            # Ridge regression: (V_train^T V_train + lam * R) w = V_train^T y_train
            # R is identity with 0 at (0,0) to not penalize intercept
            G_train = V_train.T @ V_train
            b_train = V_train.T @ y_train
            
            R = np.eye(degree + 1)
            R[0, 0] = 0.0
            
            A_reg = G_train + lam * R
            
            try:
                # Use Cholesky decomposition for stability and speed
                # A_reg should be SPD for lam > 0
                L = np.linalg.cholesky(A_reg)
                w_cv = np.linalg.solve(L.T, np.linalg.solve(L, b_train))
            except np.linalg.LinAlgError:
                # Fallback to pinv if Cholesky fails (should be rare with lam > 0)
                w_cv = np.linalg.pinv(A_reg) @ b_train
                
            pred_val = V_val @ w_cv
            mse = np.mean((pred_val - y_val) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / k
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lam = lam
            
    # Final fit with best lambda on all data
    G_full = V.T @ V
    b_full = V.T @ y
    R = np.eye(degree + 1)
    R[0, 0] = 0.0
    A_reg = G_full + best_lam * R
    
    try:
        L = np.linalg.cholesky(A_reg)
        w_norm = np.linalg.solve(L.T, np.linalg.solve(L, b_full))
    except np.linalg.LinAlgError:
        w_norm = np.linalg.pinv(A_reg) @ b_full
        
    # Transform coefficients from normalized x space to original x space
    # y = sum(c_i * ((x - mean)/std)^i)
    # We need coefficients for y = sum(d_j * x^j)
    # This requires expanding the polynomial.
    # Let u = (x - mu)/sigma. x = sigma*u + mu.
    # We have w_norm for u. We want w_orig for x.
    # V_orig(x) @ w_orig = V_norm(u) @ w_norm
    # This is a linear transformation of the coefficient vector.
    # Alternatively, we can just return the coefficients in the normalized space?
    # NO, the task requires coefficients in ascending powers of LOCAL SLOW STATE x.
    # So we must transform back.
    
    # Transformation:
    # ( (x-mu)/sigma )^i = sum_{j=0}^i binom(i, j) (x-mu)^j sigma^{-(i-j)} ... this is getting complicated.
    # Easier: The basis functions in normalized space are phi_i(u) = u^i.
    # We want to express phi_i(u) in terms of x^j.
    # u = (x-mu)/sigma = x/sigma - mu/sigma.
    # Let a = 1/sigma, b = -mu/sigma. u = a*x + b.
    # u^i = (a*x + b)^i = sum_{j=0}^i binom(i, j) (a*x)^j b^{i-j} = sum_{j=0}^i binom(i, j) a^j b^{i-j} x^j.
    # So, y = sum_{i=0}^D w_i u^i = sum_{i=0}^D w_i sum_{j=0}^i binom(i, j) a^j b^{i-j} x^j
    #       = sum_{j=0}^D ( sum_{i=j}^D w_i binom(i, j) a^j b^{i-j} ) x^j
    # Thus, d_j = sum_{i=j}^D w_i binom(i, j) a^j b^{i-j}.
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    w_orig = np.zeros(degree + 1)
    for j in range(degree + 1):
        s = 0.0
        for i in range(j, degree + 1):
            # binom(i, j)
            # Calculate binomial coefficient
            # i! / (j! (i-j)!)
            # Using scipy.special.comb or manual calculation
            # Since degree is small (5), manual is fine.
            binom_val = 1.0
            for k in range(1, j + 1):
                binom_val *= (i - k + 1) / k
            s += w_norm[i] * binom_val * (a ** j) * (b ** (i - j))
        w_orig[j] = s
        
    return w_orig
