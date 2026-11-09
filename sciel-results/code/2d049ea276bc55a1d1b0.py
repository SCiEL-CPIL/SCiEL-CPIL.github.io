"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 10s buffer before 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # For d=5, lambda=9 is a good balance
        lam = 9
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Initial state
        # Start at center of the box
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        Cinv = np.eye(d)
        Csqrt = np.eye(d)
        Csqrinv = np.eye(d)
        
        # Parameters
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        c1 = (2 + (mueff - 2) / (d + 1)) / ((d + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        # Precompute Cholesky for C (initially identity)
        # We will update C and its Cholesky factor
        
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            Y = np.random.randn(lam, d)
            # Use Cholesky of C to generate samples
            # C = Csqrt @ Csqrt.T
            # We maintain Csqrt
            P = Y @ Csqrt.T
            X = x + sigma * P
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls += lam
            
            # Update best
            if np.min(F) < best_f:
                best_f = np.min(F)
                best_x = X[np.argmin(F)]
            
            # Selection
            I = np.argsort(F)
            Xs = X[I[:mu]]
            Fs = F[I[:mu]]
            
            # Update mean
            x = x + sigma * np.sum(w[:, None] * Xs, axis=0)
            
            # Update step size
            ys = (Xs - x) / sigma  # This is incorrect, should be based on old x
            # Correct update:
            # The mean update should be: x_new = x_old + sigma * sum(w_i * (x_i - x_old))
            # Let's restart the loop logic to be precise with CMA-ES formulas
            
            # Actually, standard CMA-ES implementation:
            # 1. Sample Y
            # 2. X = m + sigma * Csqrt * Y
            # 3. Evaluate
            # 4. Sort
            # 5. m_new = m + sigma * sum(w_i * Y_i)  (using Y, not X, for stability)
            # 6. Update sigma
            # 7. Update C
            
            # Let's re-implement the loop body correctly.
            # The previous block was pseudocode. Here is the actual execution path.
            
            # We need to store Y for the weight update
            # Since I already called func, I have X and F.
            # But I need Y to update m correctly in the standard formulation.
            # m_new = m + sigma * sum(w * Y_selected)
            
            # Let's refine the variables.
            # I will restart the main loop logic in the code below.
            # For now, just break and let the code below handle it properly.
            # This draft was just to think.
            
            break # Break out of this draft loop to write the real one below

        # Real Implementation Start
        # Re-initialize state for the actual run
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        Cinv = np.eye(d)
        Csqrt = np.eye(d)
        
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        c1 = (2 + (mueff - 2) / (d + 1)) / ((d + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # 1. Sample
            Y = np.random.randn(lam, d)
            X = x + sigma * (Csqrt @ Y.T).T
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(xi) for xi in X])
            calls += lam
            
            # Update global best
            if np.min(F) < best_f:
                best_f = np.min(F)
                best_x = X[np.argmin(F)]
                
            # 3. Selection
            I = np.argsort(F)
            Ys = Y[I[:mu]]
            
            # 4. Update mean
            x = x + sigma * np.sum(w[:, None] * Ys, axis=0)
            
            # 5. Update step size
            ys = (x - x_prev) / sigma if hasattr(self, 'x_prev') else None # This is messy, let's use standard formula
            # Standard formula:
            # y_sigma = (m_new - m_old) / sigma
            # We need m_old.
            
            # Let's use a cleaner structure.
            break # Stop drafting, write clean code.

        # Clean Implementation
        x = np.zeros(d)
        m = x.copy()
        sigma = 2.0
        C = np.eye(d)
        Cinv = np.eye(d)
        Csqrt = np.eye(d)
        
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        c1 = (2 + (mueff - 2) / (d + 1)) / ((d + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # 1. Sample
            Y = np.random.randn(lam, d)
            # X = m + sigma * Csqrt @ Y.T
            X = m + sigma * (Csqrt @ Y.T).T
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(xi) for xi in X])
            calls += lam
            
            # Update global best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx]
                
            # 3. Selection
            I = np.argsort(F)
            Ys = Y[I[:mu]]
            
            # 4. Update mean
            m_new = m + sigma * np.sum(w[:, None] * Ys, axis=0)
            
            # 5. Update step size
            # y_sigma = (m_new - m) / sigma
            y_sigma = (m_new - m) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            
            # 6. Update covariance
            # pc = (1-cs)*pc + sqrt(cs*(2-cs)*mueff) * y_sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            
            # Delta term
            delta = (1 + (2 * (mueff / d - 1) - cs) * (1 - np.linalg.norm(pc) / chiN)) if False else 1.0
            # Standard delta:
            # delta_h = 1 if ||pc||/chiN < (1.4 + 2/(d+1)) * (1-cs) else 0
            if np.linalg.norm(pc) / chiN < (1.4 + 2.0 / (d + 1.0)) * (1.0 - cs):
                delta_h = 1.0
            else:
                delta_h = 0.0
                
            # C update
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - delta_h) * cs * (2 - cs)) + cmu * sum(w_i * (y_i - y_bar) @ (y_i - y_bar).T)
            # Note: The formula uses Ys (selected Y)
            # y_bar = sum(w * Ys)
            y_bar = np.sum(w[:, None] * Ys, axis=0)
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - delta_h) * cs * (2 - cs)) + cmu * (w * (Ys - y_bar) ** 2).T @ (Ys - y_bar)
            
            # Wait, the standard formula is:
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - delta_h) * cs * (2 - cs)) + cmu * sum_i w_i (y_i - y_bar)(y_i - y_bar)^T
            
            # Let's compute the rank-mu update
            # Ys is (mu, d)
            # w is (mu,)
            # Ys_w = Ys * w[:, None]
            # Rank-mu part = Ys_w.T @ Ys - w_sum * y_bar @ y_bar.T ? No.
            # It is sum_i w_i (y_i - y_bar)(y_i - y_bar)^T
            # = sum_i w_i (y_i y_i^T - y_i y_bar^T - y_bar y_i^T + y_bar y_bar^T)
            # = sum_i w_i y_i y_i^T - y_bar (sum_i w_i y_i)^T - (sum_i w_i y_i) y_bar^T + y_bar y_bar^T sum_i w_i
            # sum_i w_i y_i = y_bar
            # = sum_i w_i y_i y_i^T - 2 y_bar y_bar^T + y_bar y_bar^T
            # = sum_i w_i y_i y_i^T - y_bar y_bar^T
            
            # So:
            Ys_w = Ys * w[:, None]
            rank_mu = Ys_w.T @ Ys - np.outer(y_bar, y_bar)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - delta_h) * cs * (2 - cs)) + cmu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            # C = Csqrt @ Csqrt.T
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-9 * np.eye(d)
                Csqrt = np.linalg.cholesky(C)
                
            m = m_new

        return best_x, best_f

# EVOLVE-BLOCK-END
