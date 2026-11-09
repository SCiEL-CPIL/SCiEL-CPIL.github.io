"""Symbolic regression: Random Fourier Features ridge regression with
cross-validated hyperparameters and a strict time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 315.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0

    def standardize(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        return (Xr - mu) / sd

    Xs = standardize(X)
    y_mean = y.mean()
    yc = y - y_mean
    yvar = max(np.var(y), 1e-12)

    # ---- Random Fourier Features kernel ------------------------------------
    # Approximates an RBF kernel: K(x1,x2) ≈ cos(Phi x1)^T cos(Phi x2)
    # where Phi rows are drawn from N(0, 1/l^2) per dimension.
    # We build a hybrid: RFF features + polynomial features + linear features.

    def build_features(X_std, freq_scale, n_freq, include_poly=True, poly_deg=2):
        """Build feature matrix from standardized input.
        
        freq_scale: lengthscale parameter (smaller = more frequencies = more local)
        n_freq: number of random Fourier features per dimension
        include_poly: whether to include polynomial terms
        poly_deg: degree of polynomial terms
        """
        rng = np.random.RandomState(42)
        n_pts = X_std.shape[0]
        
        # Random Fourier Features
        # For RBF with lengthscale l, sample Omega ~ N(0, I/l^2)
        # Feature: cos(X @ Omega.T) and sin(X @ Omega.T)
        total_freq = n_freq * d
        Omega = rng.randn(total_freq, d) / freq_scale
        # Center the frequencies for better approximation
        phase = rng.uniform(0, 2 * np.pi, size=(n_pts, total_freq))
        # Actually, standard RFF uses no phase, just cos
        X_omega = X_std @ Omega.T  # (n_pts, total_freq)
        feat_rff_cos = np.cos(X_omega)
        feat_rff_sin = np.sin(X_omega)
        
        feats = [np.ones((n_pts, 1)), feat_rff_cos, feat_rff_sin]
        
        if include_poly:
            # Polynomial features up to poly_deg
            if poly_deg >= 1:
                feats.append(X_std)
            if poly_deg >= 2:
                # All pairwise products
                for i in range(d):
                    for j in range(i, d):
                        feats.append((X_std[:, i] * X_std[:, j]).reshape(-1, 1))
        
        F = np.hstack(feats)
        return F

    # ---- Candidate hyperparameters ------------------------------------------
    # freq_scale is the lengthscale l in the RBF approximation
    # Smaller l -> larger freq_scale denominator -> higher frequencies
    cand_freq_scales = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5]
    cand_n_freq = [10, 20, 40, 60]
    cand_alin = [0.0, 0.5, 1.0, 2.0]
    cand_alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    # ---- Cross-validation ---------------------------------------------------
    rng_cv = np.random.RandomState(0)
    idx = np.arange(n)
    perm = rng_cv.permutation(n)
    n_folds = 5
    folds = np.array_split(perm, n_folds)

    def evaluate_hyp(freq_scale, n_freq, alpha, include_poly=True, poly_deg=2):
        """Return mean CV normalized MSE."""
        total = 0.0
        cnt = 0
        for fo in folds:
            tr = np.setdiff1d(idx, fo, assume_unique=False)
            va = fo
            if len(tr) < 5:
                continue
            Xtr_s = Xs[tr]
            ytr_c = yc[tr]
            Xva_s = Xs[va]
            yva_c = yc[va]
            
            Ftr = build_features(Xtr_s, freq_scale, n_freq, include_poly, poly_deg)
            Fva = build_features(Xva_s, freq_scale, n_freq, include_poly, poly_deg)
            
            # Ridge regression: (F^T F + alpha * I) w = F^T y
            n_feat = Ftr.shape[1]
            reg_matrix = Ftr.T @ Ftr + alpha * np.eye(n_feat)
            try:
                w = np.linalg.solve(reg_matrix, Ftr.T @ ytr_c)
            except np.linalg.LinAlgError:
                try:
                    w = np.linalg.lstsq(reg_matrix, Ftr.T @ ytr_c, rcond=None)[0]
                except Exception:
                    return 1e12
            pred = Fva @ w
            mse = np.mean((pred - yva) ** 2)
            total += mse / yvar
            cnt += 1
        return total / max(cnt, 1)

    # ---- Hyperparameter search ----------------------------------------------
    best_cv = np.inf
    best_params = (1.0, 20, 1e-4, True, 2)

    # Coarse grid search
    for freq_scale in cand_freq_scales:
        for n_freq in cand_n_freq:
            for alpha in cand_alpha[:3]:
                if time.time() - t0 > DEADLINE - 60:
                    break
                cv = evaluate_hyp(freq_scale, n_freq, alpha)
                if cv < best_cv:
                    best_cv = cv
                    best_params = (freq_scale, n_freq, alpha, True, 2)
            if time.time() - t0 > DEADLINE - 60:
                break
        if time.time() - t0 > DEADLINE - 60:
            break

    # Refine around best
    fs_b, nf_b, al_b, ip_b, pd_b = best_params
    # Try more alpha values
    for alpha in cand_alpha:
        if time.time() - t0 > DEADLINE - 40:
            break
        cv = evaluate_hyp(fs_b, nf_b, alpha, ip_b, pd_b)
        if cv < best_cv:
            best_cv = cv
            best_params = (fs_b, nf_b, alpha, ip_b, pd_b)

    # Try nearby freq scales
    for fs in [fs_b * 0.7, fs_b * 0.85, fs_b * 1.15, fs_b * 1.3]:
        if time.time() - t0 > DEADLINE - 30:
            break
        cv = evaluate_hyp(fs, nf_b, best_params[2], best_params[3], best_params[4])
        if cv < best_cv:
            best_cv = cv
            best_params = (fs, nf_b, best_params[2], best_params[3], best_params[4])

    freq_scale, n_freq, alpha, include_poly, poly_deg = best_params

    # ---- Final fit on all data ----------------------------------------------
    Ftr = build_features(Xs, freq_scale, n_freq, include_poly, poly_deg)
    n_feat = Ftr.shape[1]
    reg_matrix = Ftr.T @ Ftr + alpha * np.eye(n_feat)
    try:
        w = np.linalg.solve(reg_matrix, Ftr.T @ yc)
    except np.linalg.LinAlgError:
        try:
            w = np.linalg.lstsq(reg_matrix, Ftr.T @ yc, rcond=None)[0]
        except Exception:
            return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)

    def predict(Xraw):
        Xr = standardize(Xraw)
        Fva = build_features(Xr, freq_scale, n_freq, include_poly, poly_deg)
        return Fva @ w + y_mean

    return predict
# EVOLVE-BLOCK-END
