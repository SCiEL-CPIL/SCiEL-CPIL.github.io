"""Rewritten optimizer for WFG9 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size / Number of subproblems
        # N=21 corresponds to a simplex lattice of resolution 5 for 3 objectives
        N = 21
        if N > self.budget:
            N = self.budget
            
        # Generate weight vectors
        # Uniform distribution on the simplex
        weights = np.random.random((N, self.n_obj))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Initialize population
        # Use LHS for better initial coverage
        def lhs(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.arange(n) / n
                offsets = np.random.random(n)
                points[:, d] = intervals + offsets
                np.random.shuffle(points[:, d])
            return points

        X = lo + (hi - lo) * lhs(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point z*
        z_star = np.min(F, axis=0)
        
        # Define neighbors
        # For small N, use all other subproblems as neighbors
        neighbors = [list(set(range(N)) - {i}) for i in range(N)]
        
        # Helper: Tchebycheff aggregation
        def tchebycheff(f, weight, z):
            return np.max(weight * np.abs(f - z))
            
        # Calculate initial Tchebycheff values
        g_vals = np.array([tchebycheff(F[i], weights[i], z_star) for i in range(N)])
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[i] += delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Shuffle order to avoid bias
            order = np.random.permutation(N)
            
            for i in order:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select parents from neighborhood of i
                if len(neighbors[i]) < 2:
                    # Fallback for very small N
                    p1_idx = i
                    p2_idx = i
                else:
                    p1_idx, p2_idx = np.random.choice(neighbors[i], size=2, replace=False)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = polynomial_mutate(c1)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f1)
                
                # Calculate Tchebycheff for child with respect to subproblem i
                g_child_i = tchebycheff(f1, weights[i], z_star)
                
                # Update subproblem i
                if g_child_i < g_vals[i]:
                    X[i] = c1
                    F[i] = f1
                    g_vals[i] = g_child_i
                    
                    # Update neighbors
                    for j in neighbors[i]:
                        g_child_j = tchebycheff(f1, weights[j], z_star)
                        if g_child_j < g_vals[j]:
                            X[j] = c1
                            F[j] = f1
                            g_vals[j] = g_child_j
                            
        # Return the non-dominated front from the final population
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0: return []
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            front = [set() for _ in range(n)]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
                if n_doms[p] == 0:
                    ranks[p] = 0
                    front[0].add(p)
                    
            i = 0
            while front[i]:
                next_front = set()
                for p in front[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.add(q)
                i += 1
                if next_front:
                    front[i] = next_front
                else:
                    break
            return [list(f) for f in front if f]

        fronts = fast_non_dominated_sort(F)
        if fronts:
            return F[fronts[0]]
        else:
            return F
# EVOLVE-BLOCK-END
