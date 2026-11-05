import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    x = np.array(training_x, dtype=np.float64)
    y = np.array(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.flatten()
    y = y.flatten()
    
    n_samples = len(x)
    if n_samples == 0:
        return np.zeros(6)
        
    # If data is extremely small, just do a simple polyfit
    if n_samples < 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except:
            return np.zeros(6)

    # Standardization
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build Chebyshev basis matrix for degree 5
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2z T_{k-1}(z) - T_{k-2}(z)
    T = np.zeros((n_samples, 6))
    T[:, 0] = 1.0
    if n_samples > 0:
        T[:, 1] = x_scaled
    for k in range(2, 6):
        T[:, k] = 2.0 * x_scaled * T[:, k-1] - T[:, k-2]
        
    # Cross-validation to select lambda
    # Split data into 5 folds
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    lambdas = np.logspace(-4, 2, 10)
    best_lambda = 0.0
    best_cv_mse = np.inf
    
    for lam in lambdas:
        if time.time() > deadline - 20.0:
            break
            
        mses = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            T_train = T[train_idx]
            y_train = y[train_idx]
            T_val = T[val_idx]
            y_val = y[val_idx]
            
            # Solve Ridge: (T^T T + lambda * D) c = T^T y
            # D is diag(0, 1, 1, 1, 1, 1)
            TtT = T_train.T @ T_train
            Tty = T_train.T @ y_train
            
            # Add penalty
            # Do not penalize the intercept (index 0)
            TtT_p = TtT.copy()
            TtT_p[1:, 1:] += lam * np.eye(5)
            
            try:
                c_cheb = np.linalg.solve(TtT_p, Tty)
            except np.linalg.LinAlgError:
                c_cheb = np.linalg.lstsq(TtT_p, Tty, rcond=None)[0]
                
            # Predict on val
            y_pred = T_val @ c_cheb
            mse = np.mean((y_val - y_pred)**2)
            mses.append(mse)
            
        mean_mse = np.mean(mses)
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_lambda = lam
            
    # Refit with best lambda on all data
    TtT = T.T @ T
    Tty = T.T @ y
    TtT_p = TtT.copy()
    TtT_p[1:, 1:] += best_lambda * np.eye(5)
    
    try:
        c_cheb = np.linalg.solve(TtT_p, Tty)
    except np.linalg.LinAlgError:
        c_cheb = np.linalg.lstsq(TtT_p, Tty, rcond=None)[0]
        
    # Convert Chebyshev coefficients (in z) to Monomial coefficients (in z)
    # T_k(z) = sum_{j=0}^k d_{k,j} z^j
    # We need d_{k,j} for k=0..5
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    # T_4(z) = 8z^4 - 8z^2 + 1
    # T_5(z) = 16z^5 - 20z^3 + 5z
    
    # d[k, j] is coeff of z^j in T_k(z)
    d = np.zeros((6, 6))
    d[0, 0] = 1
    d[1, 1] = 1
    d[2, 0] = -1; d[2, 2] = 2
    d[3, 1] = -3; d[3, 3] = 4
    d[4, 0] = 1; d[4, 2] = -8; d[4, 4] = 8
    d[5, 1] = 5; d[5, 3] = -20; d[5, 5] = 16
    
    # c_mon_z[j] = sum_{k=j}^5 c_cheb[k] * d[k, j]
    c_mon_z = np.zeros(6)
    for j in range(6):
        s = 0.0
        for k in range(j, 6):
            s += c_cheb[k] * d[k, j]
        c_mon_z[j] = s
        
    # Transform coefficients from scaled domain (z) to original domain (x)
    # z = (x - x_mean) / x_std
    # P(z) = sum c_mon_z_i z^i
    # We want Q(x) = P((x - x_mean) / x_std)
    # Let a = 1/x_std, b = -x_mean/x_std
    # Q(x) = P(a*x + b)
    # c_mon_z are coefficients for z.
    # Q(x) = sum_{k=0}^5 c_mon_z_k (a*x + b)^k
    #      = sum_{k=0}^5 c_mon_z_k sum_{j=0}^k binom(k, j) (a*x)^j b^{k-j}
    #      = sum_{j=0}^5 ( sum_{k=j}^5 c_mon_z_k binom(k, j) a^j b^{k-j} ) x^j
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    c_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for k in range(j, 6):
            binom_k_j = math.comb(k, j)
            term = c_mon_z[k] * binom_k_j * (a**j) * (b**(k-j))
            s += term
        c_orig[j] = s
        
    return c_orig
