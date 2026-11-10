import numpy as np
import time
import itertools

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([np.mean(y) if n > 0 else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Robust scaling: use percentiles to avoid outlier influence on the range
    # This is a structural change from the current mean/std standardization
    p_low, p_high = np.percentile(x, [1, 99])
    if p_high - p_low < 1e-12:
        p_low, p_high = 0.0, 1.0
    
    # Map x to t in [-1, 1]
    # t = 2 * (x - p_low) / (p_high - p_low) - 1
    scale = (p_high - p_low) / 2.0
    center = (p_high + p_low) / 2.0
    
    t = (x - center) / scale

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_deg = 5
    best_alpha = 1e-4
    best_cheb_coeffs = None

    max_deg = 5

    # Precompute Chebyshev polynomials T_k(t) for k=0..max_deg
    # T_0 = 1, T_1 = t, T_{k+1} = 2t T_k - T_{k-1}
    T = np.ones((n, max_deg + 1), dtype=np.float64)
    if max_deg >= 1:
        T[:, 1] = t
    for k in range(2, max_deg + 1):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]

    def _fit_cheb_ridge(T_sub, y_sub, degree, alpha):
        V = T_sub[:, :degree + 1]
        if alpha > 0:
            A = V.T @ V
            b = V.T @ y_sub
            reg = alpha * np.eye(degree + 1)
            # Add small regularization to diagonal to ensure invertibility if needed
            try:
                c = np.linalg.solve(A + reg, b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
        else:
            try:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
            except Exception:
                c = np.zeros(degree + 1)
        return c

    def _cv_mse(T, y, degree, alpha, k_folds=5):
        n_samples = len(T)
        if n_samples < (degree + 2) * k_folds:
            k_folds = 2
        
        # Use fixed random state for reproducibility in CV
        idx = np.random.RandomState(42).permutation(n_samples)
        fold_size = n_samples // k_folds
        mse_sum = 0.0
        
        for k in range(k_folds):
            lo = k * fold_size
            hi = lo + fold_size
            if k == k_folds - 1:
                hi = n_samples
            
            val_idx = idx[lo:hi]
            train_idx = np.concatenate([idx[:lo], idx[hi:]])
            
            T_train = T[train_idx]
            y_train = y[train_idx]
            T_val = T[val_idx]
            y_val = y[val_idx]
            
            c = _fit_cheb_ridge(T_train, y_train, degree, alpha)
            pred = T_val[:, :degree + 1] @ c
            mse_sum += np.mean((pred - y_val)**2)
        
        return mse_sum / k_folds

    # Grid search over degree and regularization
    degrees = [1, 2, 3, 4, 5]
    # Chebyshev fits are generally more stable, so we can use smaller alphas
    alphas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]

    # First pass: coarse grid
    for deg in degrees:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse(T, y, deg, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_deg = deg
                    best_alpha = alpha
            except Exception:
                continue

    # Second pass: refine around the best degree with finer alphas
    if time.time() < deadline - 10.0:
        # Logspace for alpha refinement
        # Center around best_alpha
        log_alpha = np.log10(max(1e-10, best_alpha))
        refined_alphas = 10 ** (np.linspace(log_alpha - 2, log_alpha + 2, 20))
        for deg in [max(1, best_deg - 1), best_deg, min(5, best_deg + 1)]:
            if time.time() > deadline:
                break
            for alpha in refined_alphas:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse(T, y, deg, float(alpha))
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_deg = deg
                        best_alpha = float(alpha)
                except Exception:
                    continue

    # Final fit on all data with best settings
    c_cheb = _fit_cheb_ridge(T, y, best_deg, best_alpha)

    # Now we need to convert Chebyshev coefficients in t to Monomial coefficients in x
    # P(t) = sum_{i=0}^{deg} c_i T_i(t)
    # We want P(x) = sum_{j=0}^{5} a_j x^j
    # t = (x - center) / scale = x/scale - center/scale
    # Let t = u*x + v where u = 1/scale, v = -center/scale
    # T_i(u*x + v) is a polynomial in x of degree i.
    
    # We can compute the monomial coefficients of T_i(t) in terms of x directly.
    # Or, expand the Chebyshev polynomial in t to monomials in t, then substitute t.
    
    # Step 1: Convert Chebyshev coeffs (c_i) to monomial coeffs in t (d_k)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    # Let's build a conversion matrix M_cheb_to_monom_t of size (max_deg+1, max_deg+1)
    # M[i, j] is the coefficient of t^j in T_i(t)
    M_cheb_to_monom_t = np.zeros((max_deg + 1, max_deg + 1))
    
    # Recursive definition of Chebyshev polynomials in monomial form
    # T_0 = 1
    M_cheb_to_monom_t[0, 0] = 1.0
    if max_deg >= 1:
        M_cheb_to_monom_t[1, 1] = 1.0
    for i in range(2, max_deg + 1):
        # T_i = 2t T_{i-1} - T_{i-2}
        # 2t T_{i-1} shifts coefficients of T_{i-1} by 1 and multiplies by 2
        M_cheb_to_monom_t[i, 1:] = 2.0 * M_cheb_to_monom_t[i-1, :-1]
        M_cheb_to_monom_t[i, :] -= M_cheb_to_monom_t[i-2, :]

    # Get monomial coefficients in t: d = c_cheb @ M_cheb_to_monom_t
    # d_k is coeff of t^k in P(t)
    d = np.dot(c_cheb[:best_deg+1], M_cheb_to_monom_t[:best_deg+1, :best_deg+1])
    
    # Now substitute t = u*x + v
    # P(x) = sum_{k=0}^{deg} d_k (u*x + v)^k
    # (u*x + v)^k = sum_{j=0}^k binom(k, j) (u*x)^j v^{k-j}
    #             = sum_{j=0}^k binom(k, j) u^j v^{k-j} x^j
    
    # a_j = sum_{k=j}^{deg} d_k * binom(k, j) * u^j * v^{k-j}
    
    u = 1.0 / scale
    v = -center / scale
    
    a = np.zeros(6, dtype=np.float64)
    
    for k in range(best_deg + 1):
        dk = d[k]
        if dk == 0:
            continue
        for j in range(k + 1):
            # Calculate binom(k, j)
            if k == 0:
                binom_kj = 1.0
            else:
                # Use multiplicative formula to avoid large factorials
                # binom(k, j) = k! / (j! (k-j)!)
                # We can compute iteratively or use simple loop
                # Since max_deg is 5, simple calculation is fine
                if j == 0 or j == k:
                    binom_kj = 1.0
                else:
                    mm = min(j, k-j)
                    num = 1
                    den = 1
                    for t_idx in range(1, mm + 1):
                        num *= (k - mm + t_idx)
                        den *= t_idx
                    binom_kj = num / den
            
            term = dk * binom_kj * (u ** j) * (v ** (k - j))
            a[j] += term

    # Ensure finite
    if not np.all(np.isfinite(a)):
        a = np.zeros(6, dtype=np.float64)
        a[0] = np.mean(y)

    return a
