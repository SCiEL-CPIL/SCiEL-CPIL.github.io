import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    
    if n < 6 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.zeros(6)
        
    # Split data for hyperparameter selection
    # Use a fixed split for reproducibility
    idx = np.arange(n)
    # Shuffle? The task says "flattened paired samples from two independent... trajectories".
    # They might be ordered in time. Shuffling is good for unbiased validation.
    # However, the validation set in the *score* is independent.
    # For internal selection, shuffling is better.
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    
    n_val = max(1, int(0.2 * n))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]
    
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_trn = x[trn_idx]
    y_trn = y[trn_idx]
    
    n_trn = len(x_trn)
    
    best_score = np.inf
    best_params = {'alpha': 0.0, 'h': None} # h=None means no NW
    
    # Determine if we can afford Nadaraya-Watson
    use_nw = n_trn < 3000
    
    # Candidate Alphas
    alphas = np.logspace(-6, 3, 10)
    # Candidate Bandwidths (relative to std of x)
    std_x = np.std(x_trn)
    if std_x == 0:
        std_x = 1.0
    h_factors = np.logspace(-1, 1, 10) # 0.1 to 10 times std
    
    # Pre-compute Vandermonde for val
    V_val = np.vander(x_val, N=6, increasing=True)
    
    # Search Loop
    for h_factor in h_factors:
        if time.time() > deadline - 10:
            break
            
        h = h_factor * std_x
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
                
            y_trn_smooth = y_trn
            if use_nw:
                try:
                    # Compute NW smoothed y for training set
                    diff = x_trn[:, None] - x_trn[None, :]
                    weights = np.exp(-0.5 * (diff / h) ** 2)
                    sums = np.sum(weights, axis=1)
                    sums[sums == 0] = 1e-10
                    y_trn_smooth = np.sum(weights * y_trn[None, :], axis=1) / sums
                except:
                    continue
            else:
                # If not using NW, we only need to check one alpha per "h" iteration?
                # No, if use_nw is False, we should break out of h loop or just run once.
                # Let's restructure: if not use_nw, only run alpha loop once.
                if h_factor != h_factors[0]:
                    continue
            
            # Fit polynomial
            V_trn = np.vander(x_trn, N=6, increasing=True)
            
            if alpha == 0.0:
                try:
                    c = np.linalg.lstsq(V_trn, y_trn_smooth, rcond=None)[0]
                except:
                    continue
            else:
                # Ridge
                A = V_trn.T @ V_trn
                b = V_trn.T @ y_trn_smooth
                A_reg = A + alpha * np.eye(6)
                try:
                    c = np.linalg.solve(A_reg, b)
                except:
                    try:
                        c = np.linalg.lstsq(V_trn, y_trn_smooth, rcond=None)[0]
                    except:
                        continue
            
            if not np.all(np.isfinite(c)):
                continue
            
            # Validate
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_params = {'alpha': alpha, 'h': h if use_nw else None}
                
    # If use_nw was False, we might have only tested the first h_factor.
    # If use_nw was True, we tested all.
    
    # Final Fit on ALL data using best_params
    alpha = best_params['alpha']
    h = best_params['h']
    
    y_fit = y
    if h is not None:
        # Compute NW smoothed y for FULL training set
        # This might be slow if n is large, but we only do it once.
        # If n > 3000, we skipped NW in search, so h is None.
        # So this block only executes if n < 3000.
        diff = x[:, None] - x[None, :]
        weights = np.exp(-0.5 * (diff / h) ** 2)
        sums = np.sum(weights, axis=1)
        sums[sums == 0] = 1e-10
        y_fit = np.sum(weights * y[None, :], axis=1) / sums
        
    V_full = np.vander(x, N=6, increasing=True)
    
    if alpha == 0.0:
        try:
            final_coeff = np.linalg.lstsq(V_full, y_fit, rcond=None)[0]
        except:
            final_coeff = np.zeros(6)
    else:
        A = V_full.T @ V_full
        b = V_full.T @ y_fit
        A_reg = A + alpha * np.eye(6)
        try:
            final_coeff = np.linalg.solve(A_reg, b)
        except:
            try:
                final_coeff = np.linalg.lstsq(V_full, y_fit, rcond=None)[0]
            except:
                final_coeff = np.zeros(6)
                
    # Safety check
    if not np.all(np.isfinite(final_coeff)):
        # Fallback to simple LS on raw data
        try:
            final_coeff = np.linalg.lstsq(np.vander(x, N=6, increasing=True), y, rcond=None)[0]
        except:
            final_coeff = np.zeros(6)
            
    return final_coeff
