import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size
        N = 20
        
        # Generate weight vectors (uniform on simplex)
        weights = self._generate_weights(N, m)
        
        # Find neighbors
        neighbors = self._find_neighbors(weights, K=5)
        
        # Initialize population
        X = np.random.rand(N, n) * (hi - lo) + lo
        F = np.array([problem(x) for x in X])
        self.eval_count += N
        
        # Ideal point
        ideal_point = np.min(F, axis=0)
        
        # Calculate Tchebycheff values
        g_values = np.zeros(N)
        for i in range(N):
            g_values[i] = self._tchebycheff(F[i], weights[i], ideal_point)
            
        # Main Loop
        while self.eval_count < self.budget:
            for i in range(N):
                if self.eval_count >= self.budget:
                    break
                
                # Select parents from neighbors
                neigh = neighbors[i]
                p1_idx = neigh[np.random.randint(len(neigh))]
                p2_idx = neigh[np.random.randint(len(neigh))]
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover (SBX)
                child = self._sbx(p1, p2, lo, hi)
                
                # Mutation (Polynomial)
                child = self._polynomial_mutation(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                self.eval_count += 1
                
                # Update ideal point
                ideal_point = np.minimum(ideal_point, f_child)
                
                # Calculate Tchebycheff value for child with weight i
                g_child = self._tchebycheff(f_child, weights[i], ideal_point)
                
                # Update if better
                if g_child < g_values[i]:
                    X[i] = child
                    F[i] = f_child
                    g_values[i] = g_child
        
        return X

    def _generate_weights(self, N, m):
        """
        Generate N uniform weight vectors on the m-1 dimensional simplex.
        """
        # Use Dirichlet distribution with alpha=1 for uniformity
        # But Dirichlet might be tricky to import or use correctly without scipy.stats if we want to be safe?
        # scipy is available.
        from scipy.stats import dirichlet
        weights = dirichlet.rvs(np.ones(m), size=N, random_state=42)
        return weights

    def _find_neighbors(self, weights, K):
        """
        Find K nearest neighbors for each weight vector.
        """
        N = len(weights)
        neighbors = []
        # Compute pairwise distances
        # dist[i][j] = ||w_i - w_j||
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Exclude self
            dists[i] = np.inf
            # Get K smallest
            idx = np.argsort(dists)[:K]
            neighbors.append(idx)
        return neighbors

    def _tchebycheff(self, f, w, ideal):
        """
        Calculate Tchebycheff distance.
        g(x, w) = max_i (w_i * |f_i(x) - ideal_i|)
        """
        # Handle case where ideal is infinity or if f is not better?
        # Tchebycheff is defined for minimization.
        # If f_i > ideal_i, the term is w_i * (f_i - ideal_i).
        # We want to minimize this.
        vals = w * (f - ideal)
        # If any val is negative (f < ideal), it's 0? No, ideal is the min seen.
        # So f >= ideal always.
        return np.max(vals)

    def _sbx(self, p1, p2, lo, hi, eta=20):
        """
        Simulated Binary Crossover. Returns one child.
        """
        c1 = p1.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta1 = (2 * u) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
                    else:
                        beta1 = (0.5 / (1 - u)) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1 + beta1) * x2 + (1 - beta1) * x1)
                        
        c1 = np.clip(c1, lo, hi)
        return c1

    def _polynomial_mutation(self, x, lo, hi, eta=20):
        """
        Polynomial Mutation.
        """
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    val = 2 * u
                    delta = (val) ** (1.0 / (eta + 1.0))
                    mut = (x[i] - lo[i]) / (hi[i] - lo[i])
                    mut += delta * (1 - mut) ** eta
                    mut = np.clip(mut, 0, 1)
                    x[i] = lo[i] + (hi[i] - lo[i]) * mut
                else:
                    val = 2 * (1 - u)
                    delta = (val) ** (1.0 / (eta + 1.0))
                    mut = (x[i] - lo[i]) / (hi[i] - lo[i])
                    mut -= delta * (1 - mut) ** eta
                    mut = np.clip(mut, 0, 1)
                    x[i] = lo[i] + (hi[i] - lo[i]) * mut
        return x
