"""MOEA/D with DE mutation, Nelder-Mead polish, and deadline guard for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with Tchebycheff decomposition for 3-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weight_vectors(self, n_obj, N):
        """Generate weight vectors on a simplex (Holland's method)."""
        if n_obj == 3:
            H = int(np.floor(np.sqrt(2.0 * N) - 1.5))
            H = max(H, 1)
            vectors = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    w = np.array([i, j, k], dtype=float) / H
                    vectors.append(w)
            vectors = np.array(vectors)
            if len(vectors) < N:
                centroid = np.ones(n_obj) / n_obj
                while len(vectors) < N:
                    vectors = np.vstack([vectors, centroid])
            elif len(vectors) > N:
                idx = np.linspace(0, len(vectors) - 1, N, dtype=int)
                vectors = vectors[idx]
            return vectors
        else:
            rng = np.random.default_rng(42)
            raw = rng.random((N, n_obj))
            return raw / raw.sum(axis=1, keepdims=True)

    def _nelder_mead(self, x0, problem, bounds, max_iter=50, tol=1e-8):
        """Nelder-Mead optimization for a single point (minimizes sum of objectives)."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = len(x0)
        
        # Initial simplex
        simplex = [x0.copy()]
        f0 = sum(problem(x0))
        for i in range(d):
            xi = x0.copy()
            step = 0.05 * (hi[i] - lo[i]) if hi[i] != lo[i] else 0.05
            xi[i] += step
            xi = np.clip(xi, lo, hi)
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = [f0] + [sum(problem(x)) for x in simplex[1:]]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        for _ in range(max_iter):
            # Sort by objective value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[-1] - fvals[0] < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lo, hi)
            fr = sum(problem(xr))
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lo, hi)
                fe = sum(problem(xe))
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = sum(problem(xc))
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = sum(problem(simplex[i]))
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = sum(problem(xc))
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = sum(problem(simplex[i]))
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = min(budget, 30)
        N = max(N, 10)

        # Generate weight vectors
        W = self._generate_weight_vectors(n_obj, N)

        # Initialize population: random points
        rng = np.random.default_rng(123)
        X = lo + (hi - lo) * rng.random((N, d))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Ideal and nadir points
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0

        # Neighborhood
        W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        cos_sim = W_norm @ W_norm.T
        dist = 1.0 - cos_sim
        T = N
        neighbors = []
        for i in range(N):
            d_i = dist[i].copy()
            d_i[i] = 1e9
            idx = np.argsort(d_i)[:T]
            neighbors.append(idx)

        # Tchebycheff scalarization
        def tchebycheff(f, w, ideal):
            f_norm = (f - ideal) / F_range
            return np.max(w * f_norm)

        # DE parameters
        F_de = 0.8  # Mutation factor
        CR = 0.9    # Crossover rate

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring for each subproblem
            new_X = []
            new_F = []
            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break

                # Select parents from neighborhood of i
                neigh = neighbors[i]
                # DE-style selection: pick 3 distinct parents from neighborhood
                p_indices = rng.choice(len(neigh), size=3, replace=False)
                p1_idx = neigh[p_indices[0]]
                p2_idx = neigh[p_indices[1]]
                p3_idx = neigh[p_indices[2]]

                # DE mutation: v = x_p1 + F * (x_p2 - x_p3)
                v = X[p1_idx] + F_de * (X[p2_idx] - X[p3_idx])

                # Binomial crossover
                child = v.copy()
                jrand = rng.integers(d)
                for j in range(d):
                    if rng.random() < CR or j == jrand:
                        child[j] = v[j]
                    else:
                        child[j] = X[i][j]

                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                new_X.append(child)
                new_F.append(fc)

            if not new_X:
                break

            new_X = np.array(new_X)
            new_F = np.array(new_F)

            # Update ideal and nadir points
            F_min = np.minimum(F_min, new_F.min(axis=0))
            F_max = np.maximum(F_max, new_F.max(axis=0))
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0

            # Replacement: for each subproblem i, compare parent and offspring
            for i in range(len(new_X)):
                f_parent = tchebycheff(F[i], W[i], F_min)
                f_child = tchebycheff(new_F[i], W[i], F_min)
                if f_child <= f_parent:
                    X[i] = new_X[i]
                    F[i] = new_F[i]

        # Nelder-Mead polish on non-dominated points
        # Find non-dominated points
        non_dom_mask = np.ones(N, dtype=bool)
        for i in range(N):
            for j in range(N):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dom_mask[i] = False
                    break

        non_dom_indices = np.where(non_dom_mask)[0]
        
        # Polish a subset of non-dominated points (limit to avoid timeout)
        max_polish = min(len(non_dom_indices), 8)
        if max_polish > 0:
            # Pick diverse non-dominated points to polish
            if len(non_dom_indices) > max_polish:
                # Select evenly spaced indices
                polish_indices = np.linspace(0, len(non_dom_indices) - 1, max_polish, dtype=int)
                polish_indices = non_dom_indices[polish_indices]
            else:
                polish_indices = non_dom_indices
            
            for idx in polish_indices:
                if time.time() > deadline - 30:  # Leave 30s buffer
                    break
                polished = self._nelder_mead(X[idx], problem, bounds, max_iter=30, tol=1e-10)
                f_polished = problem(polished)
                # Update if improved
                if np.any(f_polished < F[idx]):
                    X[idx] = polished
                    F[idx] = f_polished

        return X
# EVOLVE-BLOCK-END
