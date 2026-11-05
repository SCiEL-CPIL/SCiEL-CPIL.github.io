"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-Parametric Kernel Recursive Least Squares (KRLS) with RBF kernel.
    
    This approach replaces the linear weight vector with a set of adaptive basis 
    function centers. It captures local, non-linear reward structures and abrupt 
    regime shifts without explicit changepoint detection.
    
    Key features:
    - RBF kernel mapping to infinite-dimensional feature space.
    - Recursive Least Squares (RLS) update for fast adaptation.
    - Forgetting factor (lambda) to handle non-stationarity.
    - Adaptive number of centers (bounded) to manage compute.
    - Deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Kernel parameters
        # Sigma controls the width of the RBF. Smaller sigma = more local.
        # We use a heuristic initial sigma based on dim.
        self.sigma = 1.0 
        self.gamma_kernel = 1.0 / (2.0 * self.sigma * self.sigma)
        
        # RLS parameters
        self.lam = 0.95  # Forgetting factor
        self.mu = 0.01   # Regularization (ridge)
        
        # Per-arm state
        # Centers: list of context vectors
        # Coefficients: weights for the centers
        self.centers = [[] for _ in range(n_arms)]
        self.coeffs = [np.array([]) for _ in range(n_arms)]
        
        # Max centers per arm to prevent memory/compute explosion
        self.max_centers = 50
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 115.0  # Safety margin before 120s limit
        
    def _kernel(self, x, y):
        """Compute RBF kernel between x and y."""
        # Efficient squared distance
        diff = x - y
        return np.exp(-self.gamma_kernel * np.dot(diff, diff))
    
    def _kernel_matrix(self, x, centers):
        """Compute kernel vector between x and all centers."""
        if len(centers) == 0:
            return np.array([])
        # centers is a list of arrays. Stack for vectorized op if possible, 
        # but loop is safer for small N and variable lengths.
        k_vec = np.zeros(len(centers))
        for i, c in enumerate(centers):
            diff = x - c
            k_vec[i] = np.exp(-self.gamma_kernel * np.dot(diff, diff))
        return k_vec

    def _add_center(self, arm, context):
        """Add a new center to the arm's dictionary."""
        self.centers[arm].append(context.copy())
        # Initialize coefficient to 0? Or some small value?
        # In RLS, we usually append a new dimension to the coefficient vector.
        # If we have K centers, coeff is size K.
        # When adding a center, we need to expand the inverse matrix too?
        # Standard KRLS:
        # K_{t+1} = [[K_t, k_t], [k_t^T, k_t(x_t,x_t)]]
        # We maintain K_inv.
        # This is getting complex to implement from scratch correctly in a few lines.
        
        # Alternative simpler approach: Fixed dictionary or greedy dictionary.
        # Given the constraint, let's stick to a simpler "Kernelized LinUCB" style 
        # or a simplified RLS that doesn't require full matrix inversion updates 
        # if the dictionary grows too much.
        
        # Actually, let's implement a proper RLS update for the coefficient vector.
        # State:
        # P: Inverse of the weighted kernel matrix (size M x M)
        # alpha: Coefficients (size M)
        
        # If we just add a center, we need to update P.
        # This is computationally expensive if done naively.
        
        # Let's simplify: Use a fixed set of centers? No, "adaptive basis function centers".
        # Let's use a "Sparse" approximation: Only add a center if the prediction error is large.
        
        pass # Placeholder, logic moved to update

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check
        if time.time() - self._start_time > self._deadline:
            # Fallback: Greedy based on current coeffs
            scores = []
            for a in range(self.n):
                if len(self.centers[a]) == 0:
                    scores.append(0.0)
                else:
                    k_vec = self._kernel_matrix(context, self.centers[a])
                    pred = np.dot(k_vec, self.coeffs[a])
                    scores.append(pred)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            if len(self.centers[a]) == 0:
                # No data yet, explore
                scores.append(np.random.randn())
                continue
            
            k_vec = self._kernel_matrix(context, self.centers[a])
            pred = np.dot(k_vec, self.coeffs[a])
            
            # Uncertainty estimation for UCB-like behavior
            # In KRLS, variance is approximated by k(x,x) * (1 - k^T (K + mu I)^-1 k)
            # We don't maintain K_inv explicitly in the simple loop above.
            # Let's add a simple exploration bonus based on novelty.
            # If the context is far from all centers, uncertainty is high.
            min_dist = min([np.dot(context - c, context - c) for c in self.centers[a]])
            # Heuristic: higher novelty -> higher bonus
            bonus = 1.0 * np.exp(-0.5 * min_dist) # Wait, if dist is small, exp is 1. If dist is large, exp is 0.
            # We want high bonus for high dist.
            bonus = 1.0 * min_dist / (1.0 + min_dist)
            
            # Simple UCB: pred + beta * uncertainty
            # Let's use a fixed beta for now, or adapt it.
            beta = 1.0
            score = pred + beta * bonus
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = float(reward)
        
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            return

        # 1. Check if we should add a new center
        # Strategy: Add a center if the current dictionary doesn't explain the context well.
        # Or, just add a center every N steps if dictionary is small.
        # For robustness, let's add a center if the number of centers is less than max_centers.
        # To keep it "adaptive", we can use a greedy criterion: 
        # If the residual is large, add a center.
        
        # First, predict with current centers
        if len(self.centers[a]) == 0:
            # First observation: Add as first center
            self.centers[a].append(x.copy())
            # Initialize P (inverse kernel matrix) and coeffs
            # K = [k(x,x)] = [1.0] (since RBF(x,x)=1)
            # P = 1 / (lam * 1 + mu) ? No, initial P is usually 1/mu or similar.
            # Let's initialize P as a scalar 1.0 (inverse of kernel value)
            # And coeff as 0.
            self.P_inv = [1.0]  # List of matrices? No, per arm.
            # We need to store P_inv per arm.
            # Let's restructure state to include P_inv.
            
            # RESTRUCTURE:
            # self.P_inv[arm] is a matrix of size M x M.
            # self.coeffs[arm] is a vector of size M.
            
            # Since I didn't initialize P_inv in __init__, I must fix that.
            # I will assume the code below handles the initialization properly.
            
            # Let's restart the state definition in my head to ensure the code below is correct.
            # I will rewrite the __init__ and update to be consistent.
            
            pass

        # NOTE: The code above in select/update was a draft. 
        # I will provide the complete, corrected implementation below.
# EVOLVE-BLOCK-END

# Re-implementing fully to ensure correctness and consistency.

import numpy as np
import time

class Policy:
    """
    Non-Parametric Kernel Recursive Least Squares (KRLS) policy.
    Uses RBF kernel and an adaptive dictionary of centers.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Kernel parameters
        self.sigma = 1.5
        self.gamma_kernel = 1.0 / (2.0 * self.sigma * self.sigma)
        
        # RLS parameters
        self.lam = 0.98  # Forgetting factor
        self.mu = 0.01   # Regularization
        
        # Per-arm state
        self.centers = [[] for _ in range(n_arms)]
        self.coeffs = [np.array([]) for _ in range(n_arms)]
        self.P_inv = [None for _ in range(n_arms)] # Inverse of weighted kernel matrix
        
        # Max centers per arm
        self.max_centers = 40
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 115.0
        
    def _rbf(self, x, y):
        diff = x - y
        return np.exp(-self.gamma_kernel * np.dot(diff, diff))

    def _k_vec(self, x, centers):
        if not centers:
            return np.array([])
        k = np.zeros(len(centers))
        for i, c in enumerate(centers):
            k[i] = self._rbf(x, c)
        return k

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        if time.time() - self._start_time > self._deadline:
            scores = []
            for a in range(self.n):
                if len(self.centers[a]) == 0:
                    scores.append(0.0)
                else:
                    k = self._k_vec(context, self.centers[a])
                    scores.append(np.dot(k, self.coeffs[a]))
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            if len(self.centers[a]) == 0:
                scores.append(np.random.randn())
                continue
            
            k = self._k_vec(context, self.centers[a])
            pred = np.dot(k, self.coeffs[a])
            
            # Uncertainty: Approximate using distance to nearest center
            # High distance -> high uncertainty
            if self.centers[a]:
                min_dist = min(np.dot(context - c, context - c) for c in self.centers[a])
                # Map distance to a bonus
                bonus = 0.5 * np.tanh(min_dist) 
            else:
                bonus = 1.0
                
            score = pred + bonus
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = float(reward)
        
        if time.time() - self._start_time > self._deadline:
            return

        # 1. Determine if to add a new center
        # Heuristic: Add center if error is large or dictionary is small
        add_center = False
        if len(self.centers[a]) < 3:
            add_center = True
        else:
            # Predict
            k = self._k_vec(x, self.centers[a])
            pred = np.dot(k, self.coeffs[a])
            error = abs(r - pred)
            # If error is significant, add a center to capture the new regime
            if error > 1.0:
                add_center = True
            # Also add if dictionary is not full and we haven't added in a while?
            # For simplicity, only on high error or initial phase.
            # Also, if dictionary is small, add more often.
            if len(self.centers[a]) < self.max_centers and self.t % 10 == 0:
                add_center = True

        if add_center and len(self.centers[a]) < self.max_centers:
            # Add new center
            self.centers[a].append(x.copy())
            M_new = len(self.centers[a])
            
            # Update P_inv
            # K_new = [[K_old, k_col], [k_row, k_xx]]
            # We use Sherman-Morrison-Woodbury or block inversion.
            # Simpler: Re-invert the full matrix if M is small? 
            # M <= 40, so 40x40 inversion is fast.
            # But we want to be efficient.
            
            # Let's just build the full K matrix and invert it for simplicity and robustness,
            # given M is small (<=40).
            K = np.zeros((M_new, M_new))
            for i in range(M_new):
                for j in range(M_new):
                    K[i, j] = self._rbf(self.centers[a][i], self.centers[a][j])
            
            # Apply forgetting factor to the kernel matrix?
            # In RLS with forgetting, we usually update P recursively.
            # If we rebuild, we should apply weights to the data.
            # This is complex. Let's stick to the recursive P update.
            
            # Recursive Update for P when adding a new row/col:
            # Let P_old be M x M.
            # k_col is M x 1.
            # k_xx is scalar (1.0).
            
            if self.P_inv[a] is None:
                # First center
                P_inv_new = np.array([[1.0 / (self.lam * 1.0 + self.mu)]])
            else:
                P_old = self.P_inv[a]
                M_old = P_old.shape[0]
                
                # k_col: kernel between new center and all old centers
                k_col = np.array([self._rbf(x, c) for c in self.centers[a][:-1]])
                k_xx = 1.0
                
                # Block inversion formula:
                # P_new = [[P_old + P_old k_col (k_xx - k_col^T P_old k_col)^-1 k_col^T P_old, -P_old k_col (k_xx - k_col^T P_old k_col)^-1],
                #          [- (k_xx - k_col^T P_old k_col)^-1 k_col^T P_old, (k_xx - k_col^T P_old k_col)^-1]]
                
                # Note: This assumes the new kernel matrix is K_old + lambda * ...?
                # Actually, the forgetting factor complicates the "add center" step.
                # Standard RLS with forgetting factor updates P as:
                # P_t = (1/lambda) * (P_{t-1} - (P_{t-1} k k^T P_{t-1}) / (lambda + k^T P_{t-1} k))
                
                # If we add a center, we are effectively adding a new dimension.
                # This is tricky with forgetting.
                
                # Alternative: Don't add centers dynamically in a complex way.
                # Use a fixed set of centers? No, "adaptive".
                
                # Let's simplify: If we add a center, we reset the P_inv for that arm?
                # Or just use a simple heuristic: 
                # If we add a center, we treat it as a new "epoch" for that arm?
                # No, that loses history.
                
                # Let's try a different strategy: 
                # Maintain a fixed set of "basis" contexts that are updated over time?
                # Or, use a simpler kernel method: 
                # "Kernelized UCB" with a fixed dictionary that is grown greedily.
                
                # Given the complexity and time limit, let's implement a simpler version:
                # 1. Start with empty dictionary.
                # 2. Add a center if prediction error is high.
                # 3. Update coefficients using RLS formula on the FIXED dictionary.
                # 4. If we add a center, we append to the dictionary and re-initialize P_inv 
                #    by inverting the new kernel matrix (weighted by recency?).
                #    Since M is small (<=40), inverting a 40x40 matrix is ~1ms.
                #    We can afford to do this occasionally.
                
                # So, if add_center is True:
                # 1. Append x to centers.
                # 2. Build full K matrix (M x M).
                # 3. Apply forgetting weights? 
                #    For simplicity, let's just use the raw kernel matrix + mu I.
                #    This effectively "forgets" nothing about the geometry, but the 
                #    RLS update below will handle the time decay of coefficients.
                #    However, if we don't weight K by recency, older centers have 
                #    the same influence as new ones.
                #    To handle non-stationarity, the RLS update with lambda < 1 
                #    will naturally down-weight older data in the coefficient vector.
                #    So, just inverting K + mu I is acceptable.
                
                K_full = np.zeros((M_new, M_new))
                for i in range(M_new):
                    for j in range(M_new):
                        K_full[i, j] = self._rbf(self.centers[a][i], self.centers[a][j])
                
                # Add regularization
                K_full += self.mu * np.eye(M_new)
                
                # Invert
                try:
                    self.P_inv[a] = np.linalg.inv(K_full)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse or identity
                    self.P_inv[a] = np.eye(M_new)
                
                # Expand coeffs with 0 for the new center
                old_coeffs = self.coeffs[a]
                new_coeffs = np.zeros(M_new)
                if len(old_coeffs) > 0:
                    new_coeffs[:-1] = old_coeffs
                self.coeffs[a] = new_coeffs

        # 2. RLS Update (for existing dictionary, whether or not we just added a center)
        if len(self.centers[a]) > 0:
            # Get kernel vector
            k = self._k_vec(x, self.centers[a])
            
            # Update P_inv
            # P_new = (1/lambda) * (P_old - (P_old k k^T P_old) / (lambda + k^T P_old k))
            P_old = self.P_inv[a]
            Pk = P_old @ k
            denom = self.lam + np.dot(k, Pk)
            
            if denom < 1e-10:
                denom = 1e-10
                
            # Sherman-Morrison
            P_new = (1.0 / self.lam) * (P_old - np.outer(Pk, Pk) / denom)
            
            # Symmetrize
            P_new = 0.5 * (P_new + P_new.T)
            self.P_inv[a] = P_new
            
            # Update coefficients
            # alpha_new = alpha_old + P_new * k * (y - k^T alpha_old) / lambda ?
            # Formula: alpha_t = alpha_{t-1} + K_t (y_t - k_t^T alpha_{t-1})
            # Where K_t is the gain vector: K_t = P_new * k / lambda?
            # Let's check the standard RLS derivation.
            # P_t = (1/lambda) (P_{t-1} - (P_{t-1} k k^T P_{t-1}) / (lambda + k^T P_{t-1} k))
            # alpha_t = alpha_{t-1} + (P_t k / lambda) * (y_t - k^T alpha_{t-1})?
            # No, the gain vector is usually defined as:
            # K_t = P_{t-1} k / (lambda + k^T P_{t-1} k)
            # alpha_t = alpha_{t-1} + K_t * (y_t - k^T alpha_{t-1})
            
            gain = Pk / denom
            pred = np.dot(k, self.coeffs[a])
            error = r - pred
            self.coeffs[a] = self.coeffs[a] + gain * error

# EVOLVE-BLOCK-END
