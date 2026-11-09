"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware dynamic budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        # 1. Initialization
        # Use a deterministic seed for reproducibility within the instance, 
        # but the problem has hidden shifts, so we start at 0 which is often 
        # central for BBOB-like shifted/rotated functions.
        # Actually, random start is safer against specific shifts if we don't know them.
        # However, CMA-ES is robust to start. Let's start at 0.0.
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial step size
        sigma = 2.0
        
        # Population size
        # Standard CMA-ES uses lambda = 4 + floor(3 ln n)
        # For n=5, lambda = 4 + floor(3*1.609) = 4 + 4 = 8
        # Let's use 10 for slightly better exploration.
        lambda_ = 10
        
        # CMA-ES parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:int(1.0 / np.sum(weights**2))], axis=0))
        
        c1 = (2 / ((self.dim + 1)**2 + 10)) * (1 - mueff / lambda_)
        cmu = (2 * (mueff - 2 + 2 / mueff) / ((self.dim + 2)**2 + 10)) * (
            mueff / lambda_ - 1 + 2 / mueff
        )
        cs = (mueff + 2 + 2 / mueff) / (self.dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        cc = (4 + self.dim / mueff) / (self.dim + 4 + 2 / mueff)
        
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Precompute sqrt(C) using Cholesky decomposition
        # We will update this periodically or every step? 
        # For 5D, Cholesky is fast. Let's do it every step for accuracy.
        
        # Evaluation loop
        # We need to track the number of evaluations to ensure we don't exceed 300.
        # But the task says "More than 300 calls... makes the whole result invalid".
        # So we must strictly limit to 300 calls.
        # The dynamic budget idea: we can't exceed 300 total.
        # So the "dynamic" part is how we use the 300.
        # CMA-ES is generally better than (1+1) for multimodal functions like Rastrigin/Gallagher.
        
        # Let's run CMA-ES until we hit 300 calls or time out.
        # Note: func(x) counts as 1 call.
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # y = x + sigma * z, where z ~ N(0, C)
            # We use Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(self.dim)
                ps = np.zeros(self.dim)
                pc = np.zeros(self.dim)
                sigma = 2.0
                L = np.eye(self.dim)
            
            # Sample lambda_ points
            # To save time, we can generate all at once
            Z = rng.standard_normal((lambda_, self.dim))
            # Transform to normal with covariance C
            # z = L @ Z.T
            Y = x + sigma * (L @ Z.T).T
            
            # Evaluate
            Fs = []
            for i in range(lambda_):
                # Clip to box
                y_i = np.clip(Y[i], -5, 5)
                f_i = func(y_i)
                Fs.append(f_i)
                calls_made += 1
                if calls_made >= self.budget:
                    break
            
            if calls_made >= self.budget:
                break
            
            # Sort by fitness
            order = np.argsort(Fs)
            Y_sorted = Y[order]
            Fs_sorted = Fs[order]
            
            # Update mean
            w = weights
            x_new = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update sigma (CSA)
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (x_new - x)/sigma
            # C^{-1/2} is inv(L) @ inv(L).T ? No, L @ L.T = C. So C^{-1/2} = inv(L.T) @ inv(L)?
            # Actually, if C = L L.T, then C^{-1/2} = (L^{-1})^T L^{-1} is not quite right for the transformation.
            # The standard formula uses C^{-1/2} (x_new - x).
            # Let's compute inv_L = np.linalg.inv(L)
            inv_L = np.linalg.inv(L)
            # C^{-1/2} is not simply inv_L. 
            # Wait, the standard CMA-ES update for ps is:
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (x_new - x)/sigma
            # Where C^{-1/2} is the matrix square root inverse.
            # Since C = L L.T, C^{-1/2} is not trivially inv(L).
            # However, often an approximation or exact computation is used.
            # For 5D, we can compute the matrix square root inverse exactly.
            # Or, we can use the fact that the transformation was Y = x + sigma L Z.
            # The step in the standardized space is (x_new - x)/sigma.
            # We need to project this back to the isotropic space using C^{-1/2}.
            
            # Let's use scipy.linalg.sqrtm for accuracy if available, or just approximate.
            # Actually, for 5D, we can just compute the eigen-decomposition or use the Cholesky property.
            # A simpler approach for small D: 
            # C^{-1/2} = V diag(1/sqrt(lambda_i)) V.T
            # But we already have L. 
            # Let's stick to the standard implementation which often computes C^{-1/2} via Cholesky of C^{-1}? No.
            # Let's use the property: if C = L L.T, then the "whitened" step is inv_L @ (x_new - x)/sigma.
            # This is an approximation if C is not exactly L L.T from the same step, but it's standard.
            
            step = (x_new - x) / sigma
            # Project step to isotropic space
            # The vector (x_new - x) is in the "current" coordinate system.
            # We want to update ps which is in the "initial" isotropic space? 
            # No, ps is in the space where C is the covariance.
            # The formula is: ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (x_new - x)/sigma
            # Let's compute C_inv_sqrt.
            # For 5D, we can use:
            # eigs, V = np.linalg.eigh(C)
            # eigs = np.clip(eigs, 1e-12, None)
            # C_inv_sqrt = V @ np.diag(1/np.sqrt(eigs)) @ V.T
            
            # This is robust.
            eigs, V = np.linalg.eigh(C)
            eigs = np.clip(eigs, 1e-12, None)
            C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(eigs)) @ V.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ step)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made - 1)))) < (1 + 2 / (self.dim + 1)) * np.sqrt(1 - (1 - cs)**(2 * (calls_made - 1)))
            # Simplified hsig check:
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - cs)**(2 * calls_made))
            if denom < 1e-10:
                hsig = 0
            else:
                hsig = 1 if (norm_ps / denom) < (1 + 2 / (self.dim + 1)) * np.sqrt(1 - (1 - cs)**(2 * calls_made)) else 0
            
            pc = (1 - cc) * pc + hsig * cc * step
            
            # Update C
            # C = (1 - mu1 - c1 + cmu)C + c1 (ps ps.T + (1-hsig) cc (1-cc) pc pc.T) + cmu sum w_i (y_i - x)(y_i - x).T
            # Note: y_i - x should be in the isotropic space? No, the outer product is in the current space.
            # The formula uses the transformed points.
            # Actually, the standard update is:
            # C = (1 - mu1 - c1 + cmu)C + c1 (ps ps.T + (1-hsig) cc (1-cc) pc pc.T) + cmu sum w_i (y_i - x_new)(y_i - x_new).T ?
            # No, it's (y_i - x_old) or (y_i - x_new)?
            # The standard CMA-ES update uses the deviation from the OLD mean x.
            # Let's use (Y_sorted[i] - x)
            
            # Calculate the rank-1 update term
            # Term 1: ps ps.T
            # Term 2: (1-hsig) cc (1-cc) pc pc.T
            # Term 3: sum w_i (y_i - x)(y_i - x).T
            
            # Wait, the standard formula is:
            # C = (1 - mu1 - c1 + cmu) C + c1 (ps ps.T + (1-hsig) cc (1-cc) pc pc.T) + cmu sum_{i=1}^mu w_i (y_i - x)(y_i - x).T
            # Where mu1 = c1 + cmu? No.
            # Let's use the simplified update often found in implementations:
            # C = (1 - mu1 - c1 + cmu) C + c1 (ps ps.T + (1-hsig) cc (1-cc) pc pc.T) + cmu sum w_i (y_i - x)(y_i - x).T
            # mu1 = c1 + cmu? 
            # Actually, mu1 = c1 + cmu is not standard.
            # The coefficient of C is (1 - mu1 - c1 + cmu) where mu1 = c1 + cmu?
            # Let's use the values from Hansen's paper:
            # mu1 = c1 + cmu
            # Then C = (1 - mu1 - c1 + cmu) C + ... -> (1 - 2*c1) C ...? No.
            
            # Let's use the standard implementation from pyCMA or similar:
            # c1 = (2 / ((n + 1)**2 + 10)) * (1 - mueff / lam)
            # cmu = (2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + 10)) * (mueff / lam - 1 + 2 / mueff)
            # mu1 = c1 + cmu
            # C = (1 - mu1 - c1 + cmu) C + c1 (ps ps.T + (1-hsig) cc (1-cc) pc pc.T) + cmu sum w_i (y_i - x)(y_i - x).T
            # Wait, (1 - mu1 - c1 + cmu) = 1 - (c1 + cmu) - c1 + cmu = 1 - 2*c1.
            # This seems correct for the "rank-mu" update.
            
            mu1 = c1 + cmu
            
            # Rank-1 update
            rank1 = ps @ ps.T + (1 - hsig) * cc * (1 - cc) * (pc @ pc.T)
            
            # Rank-mu update
            # Sum w_i (y_i - x)(y_i - x).T
            # We use the top mu points
            Y_diff = Y_sorted[:mu] - x
            rank_mu = np.sum(w[:mu, np.newaxis, np.newaxis] * (Y_diff[:, :, np.newaxis] * Y_diff[:, np.newaxis, :]), axis=0)
            
            C = (1 - mu1 - c1 + cmu) * C + c1 * rank1 + cmu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(cs/2sigma * (||ps||/chiN - 1))
            chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
            sigma = sigma * np.exp(cs / (2 * sigma) * (norm_ps / chiN - 1))
            
            # Update x
            x = x_new
            fx = Fs_sorted[0]
            
            # Stop criterion: if sigma is very small or no improvement
            # We continue until budget or time.
            
            # Optional: Restart if sigma is too small
            if sigma < 1e-10:
                break

        return x, fx
# EVOLVE-BLOCK-END
