"""Strategy: Orthogonal polynomial basis (Legendre) for stable degree-5 fit, with NW denoising and ridge."""
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 20:
        return np.zeros(6)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 20:
        return np.zeros(6)

    # Center and scale x to [-1, 1] range for Legendre polynomials
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Further scale to [-1, 1] for better Legendre conditioning
    x_range = np.max(np.abs(x_scaled))
    if x_range > 0:
        x_leg = x_scaled / x_range
    else:
        x_leg = x_scaled

    # Deterministic train/val split
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_frac = 0.2
    val_size = max(100, int(n * val_frac))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train_scaled = x_scaled[train_idx]
    y_train = y[train_idx]
    x_val_scaled = x_scaled[val_idx]
    y_val = y[val_idx]

    x_train_leg = x_leg[train_idx]
    x_val_leg = x_leg[val_idx]

    def _legendre_design(x_vals):
        """Build Legendre polynomial design matrix P(x) for x in [-1,1]."""
        n_pts = len(x_vals)
        P = np.zeros((n_pts, 6))
        P[:, 0] = 1.0
        P[:, 1] = x_vals
        for k in range(2, 6):
            P[:, k] = ((2*k - 1) * x_vals * P[:, k-1] - (k - 1) * P[:, k-2]) / k
        return P

    def _legendre_to_monomial_coeff(leg_coeff, x_mean, x_std, x_range):
        """Convert Legendre coefficients to monomial coefficients in original x scale.
        
        x_leg = (x - x_mean) / (x_std * x_range)
        So x = x_mean + x_std * x_range * x_leg
        We need c_i such that sum(c_i * x^i) = sum(d_k * P_k(x_leg))
        
        P_k(x_leg) = sum_{j=0}^k a_{k,j} * x_leg^j
        x_leg^j = (x - x_mean)^j / (x_std * x_range)^j
        (x - x_mean)^j = sum_{m=0}^j binom(j,m) * x^m * (-x_mean)^(j-m)
        """
        # First, convert Legendre to x_leg monomial basis
        # P_k(x) = sum_{j=0}^k a_{k,j} x^j
        # Build the transformation matrix
        n_pts = 200
        x_test = np.linspace(-1, 1, n_pts)
        P_mat = _legendre_design(x_test)  # (n_pts, 6)
        
        # Get monomial coefficients in x_leg basis by fitting
        # Actually, let's use the explicit formula for Legendre polynomial coefficients
        # P_k(x) has known coefficients. Let's compute them numerically.
        
        # Alternative: use the fact that P_k(x) = sum_{j=0}^{floor(k/2)} (-1)^j * (2k-2j)! / (2^k * j! * (k-j)! * (k-2j)!) * x^{k-2j}
        # This gives P_k in terms of x^{k}, x^{k-2}, x^{k-4}, ...
        
        leg_to_xleg = np.zeros((6, 6))  # leg_to_xleg[k, j] = coeff of x_leg^j in P_k
        for k in range(6):
            for j in range(0, k+1, 2):
                # coefficient of x^j in P_k
                # P_k(x) = 2^{-k} * sum_{m=0}^{floor(k/2)} (-1)^m * binom(k, m) * binom(2k-2m, k) * x^{k-2m}
                m = (k - j) // 2
                if (k - j) % 2 == 0:
                    # binom(k, m) * binom(2k-2m, k) / 2^k * (-1)^m
                    def _binom(n, r):
                        if r < 0 or r > n:
                            return 0
                        result = 1
                        for i in range(r):
                            result *= (n - i)
                            result //= (i + 1)
                        return result
                    coeff = ((-1)**m * _binom(k, m) * _binom(2*k - 2*m, k)) / (2.0**k)
                    leg_to_xleg[k, j] = coeff
        
        # Now x_leg = (x - x_mean) / (x_std * x_range)
        # Let s = x_std * x_range, so x_leg = (x - x_mean)/s = x/s - x_mean/s
        # x_leg^j = (x - x_mean)^j / s^j = sum_{m=0}^j binom(j,m) x^m (-x_mean)^{j-m} / s^j
        
        monomial_coeff = np.zeros(6)
        for k in range(6):
            d_k = leg_coeff[k]
            if abs(d_k) < 1e-15:
                continue
            for j in range(0, k+1):
                a_kj = leg_to_xleg[k, j]
                if abs(a_kj) < 1e-15:
                    continue
                s = x_std * x_range
                for m in range(0, j+1):
                    if m < 6:
                        # binom(j, m) * (-x_mean)^(j-m) / s^j
                        def _binom2(n, r):
                            if r < 0 or r > n:
                                return 0
                            result = 1
                            for i in range(r):
                                result *= (n - i)
                                result //= (i + 1)
                            return result
                        binom_jm = _binom2(j, m)
                        monomial_coeff[m] += d_k * a_kj * binom_jm * ((-x_mean)**(j-m)) / (s**j)
        
        return np.real(monomial_coeff)

    def _score_candidate(leg_coeff):
        """Compute normalized MSE on validation set using Legendre basis."""
        P_val = _legendre_design(x_val_leg)
        pred_val = P_val @ leg_coeff
        mse = np.mean((pred_val - y_val) ** 2)
        var_y = np.var(y_val)
        if var_y < 1e-10:
            return 1e10
        return mse / var_y

    best_leg_coeff = None
    best_score = np.inf

    P_train = _legendre_design(x_train_leg)

    # Strategy 1: NW denoising in original space, then fit in Legendre basis
    def _nadaraya_watson(x_data, y_data, bandwidth):
        diff = x_data[:, None] - x_data[None, :]
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        wsum = np.sum(weights, axis=1)
        wsum[wsum < 1e-30] = 1e-30
        y_est = np.sum(weights * y_data[None, :], axis=1) / wsum
        return y_est

    bandwidths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for bw in bandwidths:
        if time.time() > deadline - 30:
            break
        try:
            y_den = _nadaraya_watson(x_train_scaled, y_train, bw)
            # Fit Legendre coefficients: minimize ||P_train @ c - y_den||^2
            A = P_train.T @ P_train
            B = P_train.T @ y_den
            leg_coeff = np.linalg.solve(A, B)
            if np.all(np.isfinite(leg_coeff)):
                score = _score_candidate(leg_coeff)
                if score < best_score:
                    best_score = score
                    best_leg_coeff = leg_coeff.copy()
        except:
            pass

    # Strategy 2: Ridge regression in Legendre basis with various lambdas
    lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
        try:
            A = P_train.T @ P_train + lam * np.eye(6)
            B = P_train.T @ y_train
            leg_coeff = np.linalg.solve(A, B)
            if np.all(np.isfinite(leg_coeff)):
                score = _score_candidate(leg_coeff)
                if score < best_score:
                    best_score = score
                    best_leg_coeff = leg_coeff.copy()
        except:
            pass

    # Strategy 3: Direct fit on raw data in Legendre basis
    try:
        A = P_train.T @ P_train
        B = P_train.T @ y_train
        leg_coeff = np.linalg.solve(A, B)
        if np.all(np.isfinite(leg_coeff)):
            score = _score_candidate(leg_coeff)
            if score < best_score:
                best_score = score
                best_leg_coeff = leg_coeff.copy()
    except:
        pass

    # Strategy 4: Try fitting on full dataset in Legendre basis
    P_full = _legendre_design(x_leg)
    try:
        A = P_full.T @ P_full
        B = P_full.T @ y
        leg_coeff_full = np.linalg.solve(A, B)
        if np.all(np.isfinite(leg_coeff_full)):
            P_val_check = _legendre_design(x_val_leg)
            pred_check = P_val_check @ leg_coeff_full
            mse_check = np.mean((pred_check - y_val) ** 2)
            var_y = np.var(y_val)
            score_check = mse_check / (var_y + 1e-10)
            if score_check < best_score:
                best_score = score_check
                best_leg_coeff = leg_coeff_full.copy()
    except:
        pass

    # Strategy 5: Weighted fit using kernel density weights
    for bw in [0.2, 0.3, 0.5]:
        if time.time() > deadline - 10:
            break
        try:
            diff = x_train_scaled[:, None] - x_train_scaled[None, :]
            dens = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
            weights = dens / (dens.sum() + 1e-10)
            W = np.diag(weights)
            A = P_train.T @ W @ P_train
            B = P_train.T @ W @ y_train
            leg_coeff = np.linalg.solve(A, B)
            if np.all(np.isfinite(leg_coeff)):
                score = _score_candidate(leg_coeff)
                if score < best_score:
                    best_score = score
                    best_leg_coeff = leg_coeff.copy()
        except:
            pass

    # Fallback
    if best_leg_coeff is None or not np.all(np.isfinite(best_leg_coeff)):
        try:
            A = P_full.T @ P_full
            B = P_full.T @ y
            best_leg_coeff = np.linalg.solve(A, B)
        except:
            return np.zeros(6)

    # Convert to monomial coefficients in original x scale
    coeff_original = _legendre_to_monomial_coeff(best_leg_coeff, x_mean, x_std, x_range)

    if not np.all(np.isfinite(coeff_original)):
        return np.zeros(6)

    return coeff_original
