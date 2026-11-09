import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a Kernel Ridge Regression model with a Sum-of-Sines kernel.
    The kernel is a sum of sinusoidal components with fixed frequencies,
    which naturally captures periodic and smooth nonlinear structures.
    Hyperparameters (amplitude, decay, regularization) are selected via LOOCV.
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize features
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X_std = (X - mu) / sd

    # Define frequency grid for the sum-of-sines kernel.
    # We use a dense grid of frequencies in the standardized space.
    # Frequencies are chosen to capture variations up to a certain scale.
    # max_freq determines the highest frequency component.
    max_freq = 10.0
    n_freqs = 20
    freqs = np.linspace(0.5, max_freq, n_freqs)

    # Hyperparameter grid
    # alpha: amplitude of the sum-of-sines part
    # beta: decay factor for frequencies (optional, but we'll keep it simple first)
    # lambda: regularization
    alphas = np.array([0.5, 1.0, 2.0, 5.0])
    lambdas = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])

    best_score = np.inf
    best_alpha = 1.0
    best_lambda = 1.0

    def compute_kernel(X_a, X_b, alpha):
        """
        Compute the Sum-of-Sines kernel matrix.
        K(x, y) = 1 + alpha * sum_i sin(f_i * ||x - y||)
        Note: Using Euclidean distance inside sine is one option.
        Alternatively, we can use a dot-product based sine kernel:
        K(x, y) = 1 + alpha * sum_i sin(f_i * (x . y))
        The dot-product version is more common for RKHS.
        Let's use: K(x, y) = 1 + alpha * sum_i cos(f_i * ||x - y||)
        Cosine is even and smooth at 0.
        """
        # Compute squared distances
        sq_norms_a = np.sum(X_a**2, axis=1)[:, np.newaxis]
        sq_norms_b = np.sum(X_b**2, axis=1)[np.newaxis, :]
        sq_dists = sq_norms_a + sq_norms_b - 2 * (X_a @ X_b.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)

        # Compute sum of cosines: sum_i cos(f_i * dist)
        # dists shape: (n_a, n_b), freqs shape: (n_freqs,)
        # We need to compute cos(freqs[:, None, None] * dists[None, :, :])
        # This creates a (n_freqs, n_a, n_b) array.
        # To save memory, we can do it in a loop or use broadcasting carefully.
        # n=200, n_freqs=20 -> 20 * 200 * 200 = 800,000 floats, which is fine.
        
        kernel = np.ones((X_a.shape[0], X_b.shape[0]))
        for f in freqs:
            kernel += np.cos(f * dists)
        
        kernel = 1.0 + alpha * kernel
        
        return kernel

    # LOOCV Loop
    for alpha in alphas:
        if time.time() > DEADLINE:
            break
        K_train = compute_kernel(X_std, X_std, alpha)
        for lam in lambdas:
            if time.time() > DEADLINE:
                break
            
            A = K_train + lam * np.eye(n)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            
            diag_inv = np.diag(A_inv)
            diag_inv[diag_inv < 1e-12] = 1e-12
            
            y_hat = y - 1.0 / diag_inv
            mse = np.mean((y_hat - y)**2)
            
            if mse < best_score:
                best_score = mse
                best_alpha = alpha
                best_lambda = lam

    # Final Fit
    K_final = compute_kernel(X_std, X_std, best_alpha)
    A_final = K_final + best_lambda * np.eye(n)
    
    try:
        alpha_vec = np.linalg.solve(A_final, y)
    except np.linalg.LinAlgError:
        alpha_vec, *_ = np.linalg.lstsq(A_final, y, rcond=None)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_std = (X_test - mu) / sd
        K_test = compute_kernel(X_test_std, X_std, best_alpha)
        return K_test @ alpha_vec

    return predict
