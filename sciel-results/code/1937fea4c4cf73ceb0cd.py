"""
Multi-objective optimizer for WFG7 using LHS + Nelder-Mead refinement.
"""
import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of lists of indices, where each inner list is a non-dominated front.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # Dominance sets
        n_dominating = np.zeros(n, dtype=int)
        fronts = []
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dominating[p] += 1
            if n_dominating[p] == 0:
                if not fronts:
                    fronts.append([])
                fronts[0].append(p)
        
        i = 0
        while fronts[i] is not None and len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dominating[q] -= 1
                    if n_dominating[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for points in a front.
        """
        if len(indices) <= 2:
            return np.full(len(indices), float('inf'))
        
        distances = np.zeros(len(indices))
        m = self.n_obj
        for k in range(m):
            Fk = F[indices, k]
            min_k = np.min(Fk)
            max_k = np.max(Fk)
            if max_k == min_k:
                continue
            # Sort indices by objective k
            sorted_indices = indices[np.argsort(Fk)]
            distances[np.where(indices == sorted_indices[0])[0][0]] = float('inf')
            distances[np.where(indices == sorted_indices[-1])[0][0]] = float('inf')
            
            for i in range(1, len(sorted_indices) - 1):
                # Find the position of this point in the original indices array
                idx_in_front = np.where(indices == sorted_indices[i])[0][0]
                prev_val = F[sorted_indices[i-1], k]
                next_val = F[sorted_indices[i+1], k]
                distances[idx_in_front] += (next_val - prev_k) / (max_k - min_k)
        return distances

    def _sbx_crossover(self, x1, x2, lo, hi):
        """Simulated Binary Crossover"""
        if np.random.random() < 0.9: # Crossover probability
            xc = np.zeros_like(x1)
            for i in range(len(x1)):
                if np.random.random() < 0.5:
                    beta = 1 + 2 * (x1[i] - lo[i]) / (hi[i] - lo[i]) if x1[i] > x2[i] else 1 + 2 * (hi[i] - x2[i]) / (hi[i] - lo[i])
                    alpha = 2 - beta ** (-1)
                    if np.random.random() <= 1 / (alpha + 1):
                        betaq = beta ** (-1 / (eta_c + 1))
                    else:
                        betaq = (1 / (2 - beta * (2 * np.random.random() - 1))) ** (1 / (eta_c + 1))
                    xc[i] = 0.5 * ((x1[i] + x2[i]) - betaq * (x1[i] - x2[i]))
                else:
                    beta = 1 + 2 * (x2[i] - lo[i]) / (hi[i] - lo[i]) if x2[i] > x1[i] else 1 + 2 * (hi[i] - x1[i]) / (hi[i] - lo[i])
                    alpha = 2 - beta ** (-1)
                    if np.random.random() <= 1 / (alpha + 1):
                        betaq = beta ** (-1 / (eta_c + 1))
                    else:
                        betaq = (1 / (2 - beta * (2 * np.random.random() - 1))) ** (1 / (eta_c + 1))
                    xc[i] = 0.5 * ((x1[i] + x2[i]) + betaq * (x1[i] - x2[i]))
            return np.clip(xc, lo, hi)
        return x1

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation"""
        xm = x.copy()
        for i in range(len(x)):
            if np.random.random() < self.p_m:
                delta1 = (xm[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - xm[i]) / (hi[i] - lo[i])
                rand = np.random.random()
                if rand < 0.5:
                    xq = 2 * rand + (1 - 2 * rand) * (1 + delta1) ** (eta_m + 1)
                    deltaq = (xq ** (1 / (eta_m + 1)) - 1)
                else:
                    xq = 2 * (1 - rand) + (2 * rand - 1) * (1 + delta2) ** (eta_m + 1)
                    deltaq = 1 - (xq ** (1 / (eta_m + 1)))
                xm[i] = xm[i] + deltaq * (hi[i] - lo[i])
        return np.clip(xm, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Latin Hypercube Sampling for diverse initial coverage
        # Use about 40% of budget for LHS sampling
        n_lhs = max(20, budget // 2)
        
        # Generate LHS samples
        X_lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Random permutation for LHS
            perm = np.random.permutation(n_lhs)
            X_lhs[:, d] = (perm + np.random.rand(n_lhs)) / n_lhs
            X_lhs[:, d] = lo[d] + X_lhs[:, d] * (hi[d] - lo[d])
        
        # Evaluate LHS samples
        F_lhs = np.array([problem(x) for x in X_lhs])
        evals = n_lhs
        
        # Phase 2: Select non-dominated points and refine with Nelder-Mead
        # Get non-dominated points from LHS samples
        fronts = self._fast_non_dominated_sort(F_lhs)
        
        # Take first front (non-dominated points)
        nd_indices = fronts[0] if fronts else [0]
        
        # Limit number of refinements to avoid timeout
        # Each Nelder-Mead refinement uses about dim+1 evaluations
        max_refinements = min(len(nd_indices), (budget - evals) // (dim + 2))
        
        # Select diverse non-dominated points to refine
        # Use simple diversity selection: spread across objective space
        if len(nd_indices) > max_refinements:
            # Select evenly spaced points from the non-dominated set
            step = len(nd_indices) / max_refinements
            nd_indices = [nd_indices[int(i * step)] for i in range(max_refinements)]
        
        # Refine each selected point with Nelder-Mead
        for idx in nd_indices:
            if time.time() > deadline:
                break
            if evals >= budget:
                break
            
            x0 = X_lhs[idx]
            refined = self._nelder_mead(x0, problem, lo, hi, max_iter=30)
            
            # Evaluate refined point
            f_refined = problem(refined)
            evals += 1
            
            # Add to population
            X_lhs = np.vstack([X_lhs, [refined]])
            F_lhs = np.vstack([F_lhs, [f_refined]])
        
        # Return all evaluated points
        return X_lhs


