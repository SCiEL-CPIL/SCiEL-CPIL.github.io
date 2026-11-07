"""Optimizer for the BBOB-style Büche-Rastrigin task (f4).

Bayesian Optimization with Gaussian Process surrogate and local refinement.
Uses EI acquisition to balance exploration of rotated Büche-Rastrigin peaks
and exploitation of the global minimum within the 500-call budget.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimizer with GP surrogate for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _rbf_kernel(self, X, Y, length_scale, sigma_f):
        """Compute RBF kernel matrix between X (n,d) and Y (m,d)."""
        # X: (n, d), Y: (m, d)
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        return sigma_f ** 2 * np.exp(-0.5 * sq_dists / (length_scale ** 2))

    def _gp_predict(self, X_new, X_train, y_train, length_scale, sigma_f, sigma_n):
        """Predict mean and variance at X_new using GP with RBF kernel."""
        K = self._rbf_kernel(X_train, X_train, length_scale, sigma_f)
        K += sigma_n ** 2 * np.eye(len(X_train))
        K_new = self._rbf_kernel(X_train, X_new, length_scale, sigma_f)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(K))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-4 * np.eye(len(K))
                L = np.linalg.cholesky(K)
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v = np.linalg.solve(L, K_new.T)
        
        mu = K_new @ alpha
        var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-12)
        
        return mu, np.sqrt(var)

    def _expected_improvement(self, mu, sigma, best_f, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-8)
        z = (best_f - mu - xi) / sigma
        ei = (best_f - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _local_refine(self, func, x0, budget, deadline):
        """Local refinement using Nelder-Mead around x0."""
        n = self.dim
        if budget < 10 or time.time() > deadline:
            return x0, func(x0) if False else None
        
        # Initialize simplex
        step = 0.3
        simplex = np.empty((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += step
        
        fx = np.empty(n + 1)
        calls = 0
        for i in range(n + 1):
            fx[i] = func(simplex[i])
            calls += 1
            if calls >= budget or time.time() > deadline:
                break
        
        if calls < n + 1:
            best_idx = np.argmin(fx[:calls])
            return simplex[best_idx], fx[best_idx]
        
        best_idx = np.argmin(fx)
        best_x = simplex[best_idx].copy()
        best_fx = fx[best_idx]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        while calls < budget and time.time() < deadline:
            order = np.argsort(fx)
            simplex = simplex[order]
            fx = fx[order]
            
            if fx[-1] - fx[0] < 1e-12:
                break
            
            centroid = simplex[:n].mean(axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls += 1
            if calls >= budget or time.time() > deadline:
                break
            
            if f_r < fx[0]:
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
                if f_e < f_r:
                    simplex[-1], fx[-1] = x_e, f_e
                else:
                    simplex[-1], fx[-1] = x_r, f_r
            elif f_r < fx[-2]:
                simplex[-1], fx[-1] = x_r, f_r
            else:
                if f_r < fx[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
                if f_c < min(f_r, fx[-1]):
                    simplex[-1], fx[-1] = x_c, f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fx[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget or time.time() > deadline:
                            break
                    if calls >= budget or time.time() > deadline:
                        break
            
            cur_best = np.argmin(fx)
            if fx[cur_best] < best_fx:
                best_fx = fx[cur_best]
                best_x = simplex[cur_best].copy()
        
        return best_x, best_fx

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        
        # Phase 1: Structured initial sampling (20% of budget)
        init_budget = max(20, min(100, budget // 5))
        rng = np.random.default_rng(42)
        
        X_samples = []
        y_samples = []
        calls = 0
        
        # Add structured points: center, corners, axis points
        structured = [np.zeros(n)]
        for i in range(n):
            for s in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = s * 4.0
                structured.append(pt)
        # Random points
        for _ in range(init_budget - len(structured)):
            structured.append(rng.uniform(-5, 5, n))
        
        for x in structured:
            if calls >= init_budget or time.time() > deadline:
                break
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            X_samples.append(x)
            y_samples.append(fx)
            calls += 1
        
        X_train = np.array(X_samples)
        y_train = np.array(y_samples)
        
        best_x = X_train[np.argmin(y_train)]
        best_fx = y_train.min()
        
        remaining = budget - calls
        
        # Phase 2: Bayesian Optimization loop
        # Adaptively tune kernel hyperparameters
        length_scale = 1.5
        sigma_f = 1.0
        sigma_n = 1e-4
        
        # Normalize y for GP stability
        y_mean = y_train.mean()
        y_std = max(y_train.std(), 1e-8)
        y_norm = (y_train - y_mean) / y_std
        
        bo_iterations = min(remaining, max(10, remaining // 2))
        local_budget_per_iter = max(5, (remaining // 2) // max(bo_iterations, 1))
        
        for it in range(bo_iterations):
            if calls >= budget or time.time() > deadline:
                break
            
            # Update hyperparameters adaptively based on data
            # Estimate length scale from data spread
            if len(X_train) > 5:
                dists = []
                for i in range(min(20, len(X_train))):
                    for j in range(i+1, min(20, len(X_train))):
                        dists.append(np.linalg.norm(X_train[i] - X_train[j]))
                if dists:
                    length_scale = max(0.5, min(3.0, np.median(dists) * 0.8))
            
            # Predict on a candidate set
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, n))
            
            mu, sigma = self._gp_predict(X_cand, X_train, y_norm, length_scale, sigma_f, sigma_n)
            
            # Denormalize for EI
            best_norm = y_norm.min()
            ei = self._expected_improvement(mu, sigma, best_norm, xi=0.01)
            
            # Select top candidates by EI
            top_k = min(5, len(ei))
            top_indices = np.argsort(ei)[::-1][:top_k]
            
            best_new_fx = best_fx
            best_new_x = best_x
            
            for idx in top_indices:
                if calls >= budget or time.time() > deadline:
                    break
                
                x_next = X_cand[idx]
                f_next = func(x_next)
                calls += 1
                
                if f_next < best_fx:
                    best_fx = f_next
                    best_x = x_next.copy()
                
                X_train = np.vstack([X_train, x_next])
                y_train = np.vstack([y_train, f_next])
                y_norm = (y_train - y_mean) / y_std
            
            # Local refinement around best point every few iterations
            if (it + 1) % 3 == 0 and remaining - (calls - init_budget) > 10:
                local_budget = min(local_budget_per_iter, budget - calls)
                if local_budget >= 10 and time.time() < deadline:
                    lx, lf = self._local_refine(func, best_x, local_budget, deadline)
                    if lf is not None and lf < best_fx:
                        best_fx = lf
                        best_x = lx
                    calls += local_budget
        
        # Phase 3: Final local polish if budget remains
        if calls < budget and time.time() < deadline:
            polish_budget = min(50, budget - calls)
            if polish_budget >= 10:
                lx, lf = self._local_refine(func, best_x, polish_budget, deadline)
                if lf is not None and lf < best_fx:
                    best_fx = lf
                    best_x = lx
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
