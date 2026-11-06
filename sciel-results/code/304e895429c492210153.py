"""
Strategy: Physical Prior + Orthogonal Basis Residual Fit
1. Estimate the linear coupling coefficient $c_{lin}$ using the known structure of L96.
   In the fast-slow limit, the unresolved forcing $\Pi$ is dominated by the term 
   $-h \cdot c \cdot \frac{1}{J} \sum Y$. 
   Due to the fast dynamics, $\langle Y \rangle \approx 0$, but the correlation with $X$ 
   yields a linear response. 
   However, a more robust "physical prior" for the linear term comes from the 
   advection term balance or simply assuming the dominant term is linear.
   
   Actually, let's look at the fast equation:
   dY_j/dt = c(-b Y_{j+1}(Y_{j+2}-Y_{j-1}) - Y_j + h X / J)
   
   The unresolved tendency in the slow equation is:
   dX_i/dt = ... - h c \langle Y \rangle_i + \text{noise}
   Wait, the task says "unresolved fast-scale forcing". 
   The slow equation is dX = adv - X + F - h*c*mean(Y).
   The "unresolved tendency" usually refers to the term $-h*c*mean(Y)$ which depends on the fast variables.
   We are fitting this term as a function of the local slow state $x$ (which is $X_i$).
   
   Since $Y$ is driven by $X$, there is a linear response. 
   Let's fit the full polynomial but using an orthogonal basis to avoid ill-conditioning,
   and use a composite loss that penalizes the norm of the coefficients in this orthogonal basis.
   
   Key Structural Change: 
   - Use shifted Legendre polynomials mapped to the range of x observed in training.
   - This orthogonal basis is numerically superior to monomials for regression, 
     allowing us to use higher effective degrees or better regularization without 
     the ill-conditioning issues of monomials.
   - We fit all 6 coefficients (degree 0 to 5) but regularize them.
   - We do NOT restrict to odd powers initially, as the antisymmetry might be broken 
     by boundary effects or specific F regimes, though it is likely strong. 
     However, the prompt's inspirations suggest odd powers work well. 
     Let's stick to the structural change: Orthogonal Basis + Ridge.
"""

import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 140.0
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    if len(x) < 6:
        return np.zeros(6)
        
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        n = len(x)
        # Determine the range of x to map to [-1, 1] for Legendre polynomials
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-12:
            x_max = x_min + 1.0
            
        # Map x to t in [-1, 1]
        # t = 2*(x - x_mid) / (x_max - x_min) - 1
        x_mid = (x_min + x_max) / 2.0
        x_range = (x_max - x_min) / 2.0
        t = (x - x_mid) / x_range
        
        # Generate shifted Legendre polynomials of degree 0 to 5
        # P_0(t) = 1
        # P_1(t) = t
        # P_2(t) = (3t^2 - 1)/2
        # P_3(t) = (5t^3 - 3t)/2
        # P_4(t) = (35t^4 - 30t^2 + 3)/8
        # P_5(t) = (63t^5 - 70t^3 + 15t)/8
        
        # We construct the design matrix B where B[:, k] = P_k(t)
        B = np.zeros((n, 6))
        B[:, 0] = 1.0
        B[:, 1] = t
        B[:, 2] = 0.5 * (3.0 * t**2 - 1.0)
        B[:, 3] = 0.5 * (5.0 * t**3 - 3.0 * t)
        B[:, 4] = 0.125 * (35.0 * t**4 - 30.0 * t**2 + 3.0)
        B[:, 5] = 0.125 * (63.0 * t**5 - 70.0 * t**3 + 15.0)
        
        # Check for finite values
        if not np.all(np.isfinite(B)) or not np.all(np.isfinite(y)):
            return np.zeros(6)
            
        # K-fold Cross-Validation for Ridge Regularization on Orthogonal Basis
        # Since the basis is orthogonal (in the continuous limit, and approximately 
        # discrete orthogonal if uniform), the coefficients are less correlated.
        # However, discrete samples are not perfectly orthogonal.
        
        k_folds = 5
        if n < k_folds * 10:
            k_folds = max(2, n // 10)
            
        np.random.seed(42)
        indices = np.random.permutation(n)
        fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
        fold_sizes[:n % k_folds] += 1
        folds = [indices[i*fold_sizes[i]:(i+1)*fold_sizes[i]] for i in range(k_folds)]
        
        # Regularization strengths
        # Scale lambda relative to the variance of y to be robust
        y_var = np.var(y)
        if y_var < 1e-12:
            y_var = 1.0
            
        # Lambda range: from 0 (LS) to large
        # For orthogonal basis, the optimal lambda might be smaller than for monomials
        lambdas = y_var * 10**np.linspace(-6, 2, 20)
        
        best_lambda = lambdas[0]
        best_cv_error = np.inf
        
        for lam in lambdas:
            if time.time() - start_time > deadline:
                break
                
            total_error = 0.0
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                B_train = B[train_idx]
                y_train = y[train_idx]
                B_val = B[val_idx]
                y_val = y[val_idx]
                
                # Ridge solution: c = (B^T B + lambda * I)^-1 B^T y
                # Note: We regularize all coefficients including the intercept (index 0).
                # Ideally, we shouldn't regularize the mean (index 0) as strongly, 
                # but for simplicity and structural consistency, we use standard ridge.
                # To be more precise, we could use a diagonal matrix for lambda,
                # but identity is the standard "Tikhonov" requested.
                
                BtB = B_train.T @ B_train
                Bty = B_train.T @ y_train
                I = np.eye(6)
                
                try:
                    c_val = np.linalg.solve(BtB + lam * I, Bty)
                except np.linalg.LinAlgError:
                    c_val = np.linalg.lstsq(BtB + lam * I, Bty, rcond=None)[0]
                
                y_pred = B_val @ c_val
                error = np.sum((y_val - y_pred)**2)
                total_error += error
                
            mean_error = total_error / k_folds
            if mean_error < best_cv_error:
                best_cv_error = mean_error
                best_lambda = lam
                
        # Final fit with best lambda
        BtB_full = B.T @ B
        Bty_full = B.T @ y
        I = np.eye(6)
        
        try:
            coeff_legendre = np.linalg.solve(BtB_full + best_lambda * I, Bty_full)
        except np.linalg.LinAlgError:
            coeff_legendre = np.linalg.lstsq(BtB_full + best_lambda * I, Bty_full, rcond=None)[0]
            
        # Convert Legendre coefficients back to Monomial coefficients
        # We have y = sum_{k=0}^5 a_k P_k(t)
        # And t = (x - x_mid) / x_range
        # We need c_i such that y = sum_{i=0}^5 c_i x^i
        
        # P_k(t) are polynomials in t.
        # Let's expand P_k(t) in terms of t^j.
        # P_0 = 1
        # P_1 = t
        # P_2 = 1.5 t^2 - 0.5
        # P_3 = 2.5 t^3 - 1.5 t
        # P_4 = 4.375 t^4 - 3.75 t^2 + 0.375
        # P_5 = 7.875 t^5 - 8.75 t^3 + 1.875 t
        
        # Matrix M where P_k(t) = sum_j M[k,j] t^j
        M = np.zeros((6, 6))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        M[2, 0] = -0.5
        M[2, 2] = 1.5
        M[3, 1] = -1.5
        M[3, 3] = 2.5
        M[4, 0] = 0.375
        M[4, 2] = -3.75
        M[4, 4] = 4.375
        M[5, 1] = 1.875
        M[5, 3] = -8.75
        M[5, 5] = 7.875
        
        # coeff_t = M.T @ coeff_legendre  (coefficients in powers of t)
        coeff_t = M.T @ coeff_legendre
        
        # Now substitute t = (x - x_mid) / x_range = x/x_range - x_mid/x_range
        # Let A = 1/x_range, B_const = -x_mid/x_range
        # t = A*x + B_const
        # t^j = (A*x + B_const)^j = sum_{k=0}^j binom(j,k) A^k B_const^{j-k} x^k
        
        A_inv = 1.0 / x_range
        B_const = -x_mid / x_range
        
        c_monomial = np.zeros(6)
        for j in range(6):
            if coeff_t[j] == 0:
                continue
            # Expand (A*x + B)^j
            for k in range(j + 1):
                # Term: binom(j,k) * A^k * B^(j-k) * x^k
                binom_val = 1.0
                for m in range(k):
                    binom_val *= (j - m) / (m + 1.0)
                
                term_coeff = binom_val * (A_inv ** k) * (B_const ** (j - k))
                c_monomial[k] += coeff_t[j] * term_coeff
                
        # Ensure finite
        if not np.all(np.isfinite(c_monomial)):
            # Fallback to simple LS on monomials
            A_mono = np.vander(x, 6, increasing=True)
            c_mono, _, _, _ = np.linalg.lstsq(A_mono, y, rcond=None)
            if np.all(np.isfinite(c_mono)):
                return c_mono
            return np.zeros(6)
            
        return c_monomial
        
    except Exception:
        # Robust fallback
        try:
            A_mono = np.vander(x, 6, increasing=True)
            c_mono, _, _, _ = np.linalg.lstsq(A_mono, y, rcond=None)
            if np.all(np.isfinite(c_mono)):
                return c_mono
        except:
            pass
        return np.zeros(6)
