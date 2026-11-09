import time
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model using ensemble of diverse basis functions
    with adaptive model selection via cross-validation.
    """
    start_time = time.time()
    deadline = start_time + 285  # Safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # Generate diverse feature sets
    def make_features(x_vec, feature_type):
        """Generate features for a given feature type."""
        x_vec = np.asarray(x_vec, dtype=np.float64)
        n_pts = len(x_vec)
        cols = []

        if feature_type == 'poly':
            # Polynomial basis up to degree 12
            for p in range(0, 13):
                cols.append(x_vec ** p)

        elif feature_type == 'fourier':
            # Fourier series
            cols.append(np.ones(n_pts))
            for k in range(1, 21):
                cols.append(np.sin(k * x_vec))
                cols.append(np.cos(k * x_vec))

        elif feature_type == 'exp':
            # Exponential basis
            cols.append(np.ones(n_pts))
            for b in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                cols.append(np.exp(b * x_vec))
                cols.append(np.exp(-b * x_vec))

        elif feature_type == 'rational':
            # Rational-like smooth functions
            cols.append(np.ones(n_pts))
            cols.append(x_vec)
            cols.append(x_vec ** 2)
            for a in [0.5, 1.0, 2.0, 5.0]:
                cols.append(1.0 / (x_vec ** 2 + a))
                cols.append(x_vec / (x_vec ** 2 + a))
                cols.append(x_vec / (x_vec ** 2 + a ** 2))

        elif feature_type == 'tanh':
            # Hyperbolic tangent and sigmoid
            cols.append(np.ones(n_pts))
            for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
                cols.append(np.tanh(s * x_vec))
                cols.append(1.0 / (1.0 + np.exp(-s * x_vec)))

        elif feature_type == 'sqrt_abs':
            # Square root and absolute value based
            cols.append(np.ones(n_pts))
            cols.append(x_vec)
            cols.append(np.abs(x_vec))
            cols.append(np.sqrt(np.abs(x_vec)))
            cols.append(np.sign(x_vec) * np.sqrt(np.abs(x_vec)))
            cols.append(x_vec * np.tanh(x_vec))
            cols.append(x_vec * np.abs(x_vec))

        elif feature_type == 'mixed1':
            # Mixed: low-order poly + some trig
            cols.append(np.ones(n_pts))
            for p in range(1, 6):
                cols.append(x_vec ** p)
            for k in [1, 2, 3]:
                cols.append(np.sin(k * x_vec))
                cols.append(np.cos(k * x_vec))

        elif feature_type == 'mixed2':
            # Mixed: exp + rational
            cols.append(np.ones(n_pts))
            cols.append(x_vec)
            for b in [0.5, 1.0, 2.0]:
                cols.append(np.exp(b * x_vec))
                cols.append(np.exp(-b * x_vec))
            for a in [1.0, 2.0]:
                cols.append(1.0 / (x_vec ** 2 + a))

        elif feature_type == 'log_positive':
            # Only valid if x > 0
            if np.min(x_vec) > 0.01:
                cols.append(np.ones(n_pts))
                cols.append(x_vec)
                cols.append(np.log(x_vec))
                cols.append(np.log(x_vec + 1.0))
                cols.append(1.0 / x_vec)
                cols.append(1.0 / x_vec ** 2)
                cols.append(np.sqrt(x_vec))
                cols.append(x_vec * np.log(x_vec))
            else:
                cols.append(np.ones(n_pts))
                cols.append(x_vec)

        elif feature_type == 'gaussian_rbf':
            # Gaussian RBF at several centers
            cols.append(np.ones(n_pts))
            # Place centers across the range
            x_min, x_max = np.min(x_vec), np.max(x_vec)
            if x_max > x_min:
                centers = np.linspace(x_min, x_max, 8)
            else:
                centers = [x_min]
            widths = [0.5, 1.0, 2.0]
            for c in centers:
                for w in widths:
                    cols.append(np.exp(-((x_vec - c) ** 2) / (2 * w ** 2)))

        # Stack and clean
        Phi = np.column_stack(cols)
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=1e6, neginf=-1e6)

        # Standardize
        means = np.mean(Phi, axis=0)
        stds = np.std(Phi, axis=0)
        stds[stds < 1e-10] = 1.0
        Phi_std = (Phi - means) / stds

        return Phi_std, means, stds

    feature_types = ['poly', 'fourier', 'exp', 'rational', 'tanh', 
                     'sqrt_abs', 'mixed1', 'mixed2', 'log_positive', 'gaussian_rbf']

    # For each feature type, try several regularization strengths
    alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    # Cross-validation to select best (feature_type, alpha)
    def cv_evaluate(Phi, y_vec, k=5):
        """Return best alpha and CV error for a given feature matrix."""
        n_samples = len(y_vec)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)

        best_alpha = alphas[0]
        best_error = np.inf

        for alpha in alphas:
            mse_sum = 0.0
            valid = True
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

                X_tr = Phi[tr_idx]
                y_tr = y_vec[tr_idx]
                X_val = Phi[val_idx]
                y_val = y_vec[val_idx]

                try:
                    XtX = X_tr.T @ X_tr
                    Xty = X_tr.T @ y_tr
                    reg = alpha * np.eye(XtX.shape[0])
                    A = XtX + reg
                    w = np.linalg.solve(A, Xty)

                    pred_val = X_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    if not np.isfinite(mse):
                        valid = False
                        break
                    mse_sum += mse
                except Exception:
                    valid = False
                    break

            if valid:
                mean_mse = mse_sum / k
                if mean_mse < best_error:
                    best_error = mean_mse
                    best_alpha = alpha

        return best_alpha, best_error

    # Evaluate all feature types
    best_feat_type = 'poly'
    best_alpha = 1.0
    best_cv_error = np.inf
    best_Phi_train = None
    best_means = None
    best_stds = None

    for ft in feature_types:
        if remaining() < 5:
            break

        try:
            Phi_train, means, stds = make_features(x, ft)
            alpha, cv_err = cv_evaluate(Phi_train, y)

            if cv_err < best_cv_error:
                best_cv_error = cv_err
                best_alpha = alpha
                best_feat_type = ft
                best_Phi_train = Phi_train
                best_means = means
                best_stds = stds
        except Exception:
            continue

    # Fallback if nothing worked
    if best_Phi_train is None:
        best_Phi_train, best_means, best_stds = make_features(x, 'poly')
        best_alpha = 1.0

    # Fit final model on all training data with best alpha
    try:
        XtX = best_Phi_train.T @ best_Phi_train
        Xty = best_Phi_train.T @ y
        reg = best_alpha * np.eye(XtX.shape[0])
        A = XtX + reg
        w_final = np.linalg.solve(A, Xty)
    except Exception:
        try:
            w_final, *_ = np.linalg.lstsq(best_Phi_train, y, rcond=None)
        except Exception:
            w_final = np.zeros(best_Phi_train.shape[1])
            w_final[0] = np.mean(y) / best_stds[0]  # Approximate constant

    # Create predict function
    def predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        try:
            x_in = X_in[:, 0]
            Phi_in, _, _ = make_features(x_in, best_feat_type)

            # Ensure shape matches
            if Phi_in.shape[1] != len(best_means):
                return np.full(len(X_in), np.mean(y))

            pred = Phi_in @ w_final

            if not np.all(np.isfinite(pred)):
                pred = np.where(np.isfinite(pred), pred, np.mean(y))

            return pred
        except Exception:
            return np.full(len(X_in), np.mean(y))

    return predict
