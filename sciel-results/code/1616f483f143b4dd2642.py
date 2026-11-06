"""Strategy: SEARCH approach - generate many varied candidate answers, keep the best."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 20:
        return np.zeros(6)

    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 20:
        return np.zeros(6)

    # Center and scale x for better conditioning of the orthogonal basis
    # We map x to [-1, 1] for Legendre polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

    # Split into train/val for candidate selection
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_frac = 0.2
    val_size = max(100, int(n * val_frac))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train = x_scaled[train_idx]
    y_train = y[train_idx]
    x_val = x_scaled[val_idx]
    y_val = y[val_idx]

    # Precompute Legendre basis matrices for train and val
    # P_k(x) are Legendre polynomials of degree k
    def _legendre_basis(x_data, degree):
        """Compute Legendre polynomial values up to 'degree' for given x_data."""
        # x_data is (N,)
        # Returns (N, degree+1) matrix where column k is P_k(x)
        n_pts = len(x_data)
        basis = np.zeros((n_pts, degree + 1))
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = x_data
            for k in range(2, degree + 1):
                # Recurrence: (k) P_k(x) = (2k-1) x P_{k-1}(x) - (k-1) P_{k-2}(x)
                basis[:, k] = ((2 * k - 1) * x_data * basis[:, k - 1] - (k - 1) * basis[:, k - 2]) / k
        return basis

    L_train = _legendre_basis(x_train, 5)
    L_val = _legendre_basis(x_val, 5)

    def _score_candidate(coeff_leg):
        """Compute normalized MSE on validation set using Legendre coefficients."""
        pred_val = L_val @ coeff_leg
        mse = np.mean((pred_val - y_val) ** 2)
        var_y = np.var(y_val)
        if var_y < 1e-10:
            return 1e10
        return mse / var_y

    def _convert_leg_to_monomial(coeff_leg, x_min, x_max):
        """
        Convert Legendre coefficients (on [-1,1]) to monomial coefficients on original x.
        P_k(2t-1) is a polynomial in t. We need coefficients in (x - x_min)/range * 2 - 1.
        Actually, it's easier to just evaluate the polynomial conversion.
        Let z = (x - x_min) / (x_max - x_min) * 2 - 1.
        x = (z + 1) / 2 * (x_max - x_min) + x_min.
        So x = A*z + B, where A = (x_max - x_min)/2, B = x_min + A.
        We have f(z) = sum c_k P_k(z).
        We want f(x) = sum d_j x^j.
        This is a linear transformation.
        Since degree is low (5), we can construct the transformation matrix.
        Let M be a 6x6 matrix such that d = M @ c.
        P_k(z) expressed in powers of z: p_k(z) = sum_{j=0}^k m_{k,j} z^j.
        Then f(z) = sum_k c_k sum_j m_{k,j} z^j.
        Then substitute z = (x - B)/A.
        z^j = (1/A^j) * (x - B)^j = (1/A^j) * sum_{i=0}^j binom(j,i) x^i (-B)^{j-i}.
        This is getting complicated. 
        Alternative: Use numpy.polyfit on the converted points? No, that loses precision.
        
        Better approach: 
        Construct the monomial coefficients directly from the Legendre recurrence in terms of x.
        Or, simpler: 
        We know the mapping z = (2*x - x_min - x_max) / (x_max - x_min).
        Let's just build the basis polynomials in x directly? No, we already have L_train.
        
        Let's use a simple interpolation/fitting trick for the conversion since degree is 5.
        We can evaluate the polynomial f(z) at 6 points in z, then fit those to x.
        But we need exact coefficients.
        
        Let's use the property:
        P_k(z) = sum_{j=0}^k a_{k,j} z^j.
        We can compute the matrix A_leg_to_z_monomial.
        Then z = (x - B)/A.
        z^j = (x - B)^j / A^j.
        (x - B)^j = sum_{i=0}^j binom(j, i) x^i (-B)^(j-i).
        
        Let's precompute the transformation matrix T such that c_monomial = T @ c_leg.
        """
        # Compute coefficients of P_k(z) in monomial basis z^j
        # P_0 = 1
        # P_1 = z
        # P_2 = (3z^2 - 1)/2
        # P_3 = (5z^3 - 3z)/2
        # P_4 = (35z^4 - 30z^2 + 3)/8
        # P_5 = (63z^5 - 70z^3 + 15z)/8
        
        # Matrix M where M[k, j] is coeff of z^j in P_k(z)
        M_leg_z = np.zeros((6, 6))
        M_leg_z[0, 0] = 1.0
        M_leg_z[1, 1] = 1.0
        M_leg_z[2, 2] = 1.5
        M_leg_z[2, 0] = -0.5
        M_leg_z[3, 3] = 2.5
        M_leg_z[3, 1] = -1.5
        M_leg_z[4, 4] = 35.0 / 8.0
        M_leg_z[4, 2] = -30.0 / 8.0
        M_leg_z[4, 0] = 3.0 / 8.0
        M_leg_z[5, 5] = 63.0 / 8.0
        M_leg_z[5, 3] = -70.0 / 8.0
        M_leg_z[5, 1] = 15.0 / 8.0
        
        # Now convert from z monomials to x monomials.
        # z = (x - B) / A
        # z^j = (1/A^j) * sum_{i=0}^j binom(j, i) x^i (-B)^(j-i)
        # Let T_z_x[j, i] be coeff of x^i in z^j.
        A_val = (x_max - x_min) / 2.0
        B_val = x_min + A_val
        
        T_z_x = np.zeros((6, 6))
        for j in range(6):
            for i in range(j + 1):
                binom = 1.0
                for k in range(j - i):
                    binom *= (j - k) / (k + 1)
                # Term: binom * x^i * (-B)^(j-i) * (1/A^j)
                T_z_x[j, i] = binom * ((-B_val) ** (j - i)) / (A_val ** j)
                
        # Total transformation: c_monomial = T_z_x @ M_leg_z @ c_leg
        T_total = T_z_x @ M_leg_z
        coeff_monomial = T_total @ coeff_leg
        return coeff_monomial

    best_coeff_leg = None
    best_score = np.inf
    best_type = None
    best_param = None

    # Strategy 1: Nadaraya-Watson denoising with various bandwidths
    # Note: NW is performed in the scaled x space ([-1, 1])
    
    def _nadaraya_watson(x_data, y_data, bandwidth):
        """Compute NW estimate at x_data points."""
        # To save memory/time, we can do this in chunks or just accept O(N^2) if N is not huge.
        # N is likely around 1000-10000. N^2 is 1e4-1e8. 1e8 floats is 800MB. Too much.
        # We need a more efficient NW or smaller N.
        # Let's subsample for NW if N is large, or use a different denoising.
        # Actually, the original code did O(N^2) on train_idx. If N_train is 8000, that's 64M ops. 
        # With numpy vectorization, it's fast but memory heavy.
        # Let's use a chunked approach or just rely on the fact that the original code worked.
        # The original code used `x_train[:, None] - x_train[None, :]`.
        # If n_train > 5000, this creates a 5000x5000 array (25M floats, 200MB). 
        # It might be tight. Let's limit NW to smaller bandwidths or subsample.
        
        # For safety, let's subsample the training data for NW denoising if it's too large.
        n_sub = min(len(x_data), 2000)
        if n_sub < len(x_data):
            sub_idx = np.random.RandomState(1).choice(len(x_data), n_sub, replace=False)
            x_sub = x_data[sub_idx]
            y_sub = y_data[sub_idx]
        else:
            x_sub = x_data
            y_sub = y_data
            
        diff = x_sub[:, None] - x_sub[None, :]
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        wsum = np.sum(weights, axis=1)
        wsum[wsum < 1e-30] = 1e-30
        y_est_sub = np.sum(weights * y_sub[None, :], axis=1) / wsum
        
        # Interpolate back to full x_data? Or just fit the polynomial to the subsampled denoised data?
        # Fitting to subsampled data is fine as it's just a regularization.
        return x_sub, y_est_sub

    bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for bw in bandwidths:
        if time.time() > deadline - 20:
            break
        try:
            x_den, y_den = _nadaraya_watson(x_train, y_train, bw)
            # Fit Legendre polynomial to denoised data
            L_den = _legendre_basis(x_den, 5)
            # Solve least squares
            # L_den.T @ L_den is 6x6
            A = L_den.T @ L_den
            B = L_den.T @ y_den
            coeff_leg = np.linalg.solve(A, B)
            if np.all(np.isfinite(coeff_leg)):
                score = _score_candidate(coeff_leg)
                if score < best_score:
                    best_score = score
                    best_coeff_leg = coeff_leg.copy()
                    best_type = 'nw'
                    best_param = bw
        except:
            pass

    # Strategy 2: Ridge regression with different regularization strengths
    # Use full training data
    A_full = L_train.T @ L_train
    B_full = L_train.T @ y_train
    
    lambdas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    for lam in lambdas:
        if time.time() > deadline - 15:
            break
        try:
            A_reg = A_full + lam * np.eye(6)
            coeff_leg = np.linalg.solve(A_reg, B_full)
            if np.all(np.isfinite(coeff_leg)):
                score = _score_candidate(coeff_leg)
                if score < best_score:
                    best_score = score
                    best_coeff_leg = coeff_leg.copy()
                    best_type = 'ridge'
                    best_param = lam
        except:
            pass

    # Strategy 3: Plain least squares (no regularization)
    try:
        coeff_leg = np.linalg.solve(A_full, B_full)
        if np.all(np.isfinite(coeff_leg)):
            score = _score_candidate(coeff_leg)
            if score < best_score:
                best_score = score
                best_coeff_leg = coeff_leg.copy()
                best_type = 'ls'
                best_param = None
    except:
        pass

    # Final Polish: Re-fit the best candidate type on the FULL dataset
    if best_coeff_leg is not None:
        if best_type == 'nw':
            # Re-run NW on full data with best bandwidth
            try:
                x_den_full, y_den_full = _nadaraya_watson(x_scaled, y, best_param)
                L_den_full = _legendre_basis(x_den_full, 5)
                A_full_den = L_den_full.T @ L_den_full
                B_full_den = L_den_full.T @ y_den_full
                coeff_leg_final = np.linalg.solve(A_full_den, B_full_den)
                if np.all(np.isfinite(coeff_leg_final)):
                    best_coeff_leg = coeff_leg_final
            except:
                pass
        elif best_type == 'ridge':
            # Re-run ridge on full data (already did this, but maybe with more data points if we used subsampling? No, we used full train_idx)
            # The 'ridge' strategy above used L_train which is train_idx. 
            # Let's re-fit on the entire x_scaled, y.
            A_all = _legendre_basis(x_scaled, 5).T @ _legendre_basis(x_scaled, 5)
            # Wait, computing basis twice is wasteful.
            L_all = _legendre_basis(x_scaled, 5)
            A_all = L_all.T @ L_all
            B_all = L_all.T @ y
            A_reg_all = A_all + best_param * np.eye(6)
            coeff_leg_final = np.linalg.solve(A_reg_all, B_all)
            if np.all(np.isfinite(coeff_leg_final)):
                best_coeff_leg = coeff_leg_final
        elif best_type == 'ls':
            L_all = _legendre_basis(x_scaled, 5)
            A_all = L_all.T @ L_all
            B_all = L_all.T @ y
            coeff_leg_final = np.linalg.solve(A_all, B_all)
            if np.all(np.isfinite(coeff_leg_final)):
                best_coeff_leg = coeff_leg_final

    # Fallback
    if best_coeff_leg is None or not np.all(np.isfinite(best_coeff_leg)):
        # Simple LS on full data
        try:
            L_all = _legendre_basis(x_scaled, 5)
            A_all = L_all.T @ L_all
            B_all = L_all.T @ y
            best_coeff_leg = np.linalg.solve(A_all, B_all)
        except:
            return np.zeros(6)

    # Convert to original monomial coefficients
    coeff_original = _convert_leg_to_monomial(best_coeff_leg, x_min, x_max)

    if not np.all(np.isfinite(coeff_original)):
        return np.zeros(6)

    return coeff_original

# EVOLVE-BLOCK-END
