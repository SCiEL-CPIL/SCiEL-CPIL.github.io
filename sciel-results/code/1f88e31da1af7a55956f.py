"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        t0 = time.time()
        deadline = t0 + 290.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Selection size mu
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sqrt_C = np.eye(dim)
        
        # Initial evaluation
        calls = 1
        best_x = x.copy()
        best_f = func(x)
        
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Sample population
            Y = rng.standard_normal((lam, dim))
            # Update inverse sqrt C periodically (approx)
            # For dim=5, we can just compute it every step or use eigendecomp
            # For simplicity and speed in dim 5, we'll use the standard update
            # but computing inv_sqrt_C from C every step is O(dim^3) = 125 ops, negligible.
            # However, the standard CMA-ES algorithm uses a recurrence. 
            # Let's use a simplified robust CMA-ES that recomputes sqrt C via eigendecomp 
            # or just uses the standard algorithm. 
            # Actually, for dim=5, let's just use the standard CMA-ES algorithm 
            # but ensure we don't blow up.
            
            # Generate points
            # We need the square root of C.
            # eigvals, eigvecs = np.linalg.eigh(C)
            # sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            # This is O(dim^3). For dim=5, it's fast.
            # But we can also just use the standard update rule.
            
            # Let's use the standard CMA-ES update.
            # 1. Sample
            # We need sqrt(C). Let's compute it.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Generate Y such that Y = sqrt(C) @ N
            # Or X = x + sigma * Y
            # Y ~ N(0, C)
            # Y = sqrt_C @ Z where Z ~ N(0, I)
            Z = rng.standard_normal((lam, dim))
            Y = Z @ sqrt_C.T  # (lam, dim)
            # Wait, if Z is (lam, dim) and sqrt_C is (dim, dim), then Z @ sqrt_C.T is correct.
            
            Pop = x + sigma * Y
            Pop = np.clip(Pop, -5, 5)
            
            # Evaluate
            FPop = np.array([func(xi) for xi in Pop])
            calls += lam
            
            # Check time again
            if time.time() > deadline:
                break
                
            # Update best
            if np.min(FPop) < best_f:
                best_f = np.min(FPop)
                best_x = Pop[np.argmin(FPop)]
                
            # Sort
            order = np.argsort(FPop)
            FSorted = FPop[order]
            XSorted = Pop[order]
            
            # Mean update
            m = np.sum(w[:, np.newaxis] * XSorted[:mu], axis=0)
            
            # pc update
            pc = (1 - c_c) * pc + c_c * (m - x) / sigma
            cs = np.linalg.norm(pc) / chi_n / np.sqrt(1 - (1 - c_c)**(2 * (calls - dim)))
            
            # sigma update
            sigma *= np.exp(c_sigma / d_sigma * (cs - 1))
            
            # C update
            delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) * c_c
            C = (1 + c_1 + c_mu - c_c - delta_d) * C
            C += c_1 * (np.outer(pc, pc) - C)
            C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(XSorted[:mu] - x, XSorted[:mu] - x) / sigma**2, axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update x
            x = m
            
        return best_x, best_f
# EVOLVE-BLOCK-END
