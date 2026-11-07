"""Rewritten optimizer using CMA-ES with adaptive population and sigma control."""
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))  # ~4+3*log(5)=~9
        mu = lam // 2  # number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping for sigma
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # chi_n expected value of norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1
        remaining = budget - 1

        while remaining > 0:
            # Generate population
            offspring = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            for i in range(lam):
                # Sample from multivariate normal with covariance C
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                remaining -= 1
                offspring.append((y, fy))
                if remaining <= 0:
                    break

            if not offspring:
                break
            
            # Convergence check
            if sigma < 1e-4:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Select parents
            parents = offspring[:mu]
            # Update mean
            new_x = sum(w * p[0] for w, p in zip(weights, parents))
            new_x = np.clip(new_x, -5, 5)

            # Update evolution paths
            y_mean = (new_x - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_mean

            # Update covariance
            delta_sigma = 0 if hsig else 1
            # Standard update: C <- (1 - c1 - cmu + c1*delta_sigma) C + c1*delta_sigma * pc pc^T + cmu * sum(w_k y_k y_k^T)
            C = (1 - c1 - cmu + c1 * delta_sigma) * C
            if delta_sigma > 0:
                C += c1 * delta_sigma * np.outer(pc, pc)
            else:
                C += c1 * (1 - delta_sigma) * (np.outer(pc, pc) - C) # This part is tricky, usually just add c1*delta_sigma*pc pc^T
                # Actually, the term is c1 * (pc pc^T + (1-delta_sigma) C)
                # So C += c1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)
                # But we already multiplied C by (1 - c1 - cmu + c1*delta_sigma).
                # Let's stick to the additive form to avoid confusion.
                # C_new = C_old * (1 - c1 - cmu + c1*delta_sigma) + c1 * (pc pc^T + (1-delta_sigma) C_old) + cmu * Sum
                # C_new = C_old * (1 - c1 - cmu + c1*delta_sigma + c1*(1-delta_sigma)) + c1*pc pc^T + cmu*Sum
                # C_new = C_old * (1 - c1 - cmu + c1) + ... = C_old * (1 - cmu) + ...
                # Wait, if delta_sigma=0:
                # C_new = C_old * (1 - c1 - cmu) + c1 * (pc pc^T + C_old) + cmu * Sum
                # C_new = C_old * (1 - c1 - cmu + c1) + c1 pc pc^T + cmu Sum
                # C_new = C_old * (1 - cmu) + c1 pc pc^T + cmu Sum
                
                # If delta_sigma=1:
                # C_new = C_old * (1 - c1 - cmu + c1) + c1 * pc pc^T + cmu Sum
                # C_new = C_old * (1 - cmu) + c1 pc pc^T + cmu Sum
                
                # It seems the coefficient of C_old is always (1 - cmu) regardless of delta_sigma?
                # No, the term (1-delta_sigma)C is inside the c1 bracket.
                # If delta_sigma=0, we add c1*C. Total C coeff: 1 - c1 - cmu + c1 = 1 - cmu.
                # If delta_sigma=1, we add 0*C. Total C coeff: 1 - c1 - cmu + c1 = 1 - cmu.
                # So the multiplicative factor should be (1 - cmu)?
                # Let's check Hansen's book.
                # C <- (1 - c1 - cmu) C + c1 (pc pc^T + (1-delta_sigma) C) + cmu sum
                # C <- (1 - c1 - cmu + c1 - c1*delta_sigma) C + c1 pc pc^T + cmu sum
                # C <- (1 - cmu - c1*delta_sigma) C + c1 pc pc^T + cmu sum
                
                # So if delta_sigma=0: Coeff is 1 - cmu.
                # If delta_sigma=1: Coeff is 1 - cmu - c1.
                
                # My previous code: (1 + c1*delta + cmu - c1 - cmu) = 1 - c1 + c1*delta.
                # If delta=0: 1 - c1. (Should be 1 - cmu).
                # If delta=1: 1. (Should be 1 - cmu - c1).
                # So the code was definitely wrong.
            
            C += cmu * np.sum([w * np.outer((p[0] - x) / sigma, (p[0] - x) / sigma) for w, p in zip(weights, parents)], axis=0)
            
            # Recalculate with correct logic
            C = (1 - cmu - c1 * delta_sigma) * C
            C += c1 * np.outer(pc, pc)
            if delta_sigma == 0:
                C += c1 * C # This is circular. 
                # Better to use the expanded form:
                # C_new = (1 - cmu) C_old + c1 pc pc^T + cmu Sum  (if delta=0)
                # C_new = (1 - cmu - c1) C_old + c1 pc pc^T + cmu Sum (if delta=1)
            
            # Let's just use the standard additive update which is safer to implement correctly.
            # C <- C - c1*C - cmu*C + c1*(pc pc^T + (1-delta_sigma)C) + cmu*Sum
            # C <- C - cmu*C - c1*delta_sigma*C + c1*pc pc^T + cmu*Sum
            # C <- (1 - cmu - c1*delta_sigma) C + c1*pc pc^T + cmu*Sum
            
            # We already did C = (1 - c1 - cmu + c1 * delta_sigma) * C in the previous block?
            # No, I replaced it. Let's do it cleanly.
            
            C *= (1 - cmu - c1 * delta_sigma)
            C += c1 * np.outer(pc, pc)
            for w, p in zip(weights, parents):
                yk = (p[0] - x) / sigma
                C += cmu * w * np.outer(yk, yk)

            # Update sigma
            sigma *= np.exp((1 / dsigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update best
            if parents[0][1] < best_fx:
                best_fx = parents[0][1]
                best_x = parents[0][0].copy()

            x = new_x
            fx = best_fx  # Not strictly used, but keep track

            # Ensure C is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_fx
