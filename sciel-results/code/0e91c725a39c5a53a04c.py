import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    x = np.asarray(training_x, dtype=float).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=float).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # The task specifies a degree-five polynomial in ascending powers of x.
    # We fit coefficients c_0..c_5 such that pred = sum(c_i * x**i).
    
    # We exploit the physical antisymmetry of the subgrid tendency.
    # In the Lorenz-96 model, the subgrid forcing is an odd function of x.
    # Therefore, we restrict the polynomial to odd powers: x^1, x^3, x^5.
    # This reduces the effective dimensionality and improves generalization.
    # The returned coefficients will have c_0=c_2=c_4=0.
    
    # Construct the design matrix for odd powers only: [x, x^3, x^5]
    V = np.column_stack([x, x**3, x**5])
    
    # We use ridge regression with cross-validation to select the regularization parameter.
    # This helps prevent overfitting and improves performance on the unseen F=10 regime.
    
    # Candidate regularization parameters (log-spaced)
    lambdas = np.logspace(-10, 2, 25)
    
    # k-fold cross-validation
    k = 5
    if n < k:
        k = max(2, n)
    
    # Shuffle indices for CV
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
    if n % k != 0:
        folds[-1] = np.concatenate([folds[-1], indices[k*fold_size:]])
    
    best_lambda = lambdas[0]
    best_cv_score = np.inf
    
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
        
        cv_errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            # Ridge regression: solve (V^T V + lambda * I) c = V^T y
            try:
                A = V_train.T @ V_train + lam * np.eye(3)
                b = V_train.T @ y_train
                c = np.linalg.solve(A, b)
                
                pred_val = V_val @ c
                mse = np.mean((pred_val - y_val)**2)
                cv_errors.append(mse)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_lambda = lam
    
    # Final fit on all data with best lambda
    A = V.T @ V + best_lambda * np.eye(3)
    b = V.T @ y
    try:
        c_odd = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_odd = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Ensure finite
    c_odd = np.nan_to_num(c_odd, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Construct the full 6-coefficient vector in ascending powers
    # c = [c_0, c_1, c_2, c_3, c_4, c_5]
    # Odd powers only: c_1, c_3, c_5 are non-zero
    c = np.zeros(6)
    c[1] = c_odd[0]  # x^1
    c[3] = c_odd[1]  # x^3
    c[5] = c_odd[2]  # x^5
    
    return c
