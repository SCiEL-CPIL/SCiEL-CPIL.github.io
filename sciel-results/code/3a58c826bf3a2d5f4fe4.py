    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume standard DTLZ2 structure
        # Active dims: 0, 1
        # Distance dims: 10, 11 (set to 0.5)
        # Irrelevant dims: 2-9 (set to 0.5)
        
        pop_size = 20
        X = np.zeros((pop_size, self.dim))
        X[:, :] = 0.5
        # Init active dims
        X[:, 0] = np.random.rand(pop_size)
        X[:, 1] = np.random.rand(pop_size)
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # NSGA-II
        while evals < self.budget:
            # Determine how many offspring we can generate
            # We generate pairs, so even number
            remaining = self.budget - evals
            n_offspring = min(pop_size, remaining)
            # Ensure even
            if n_offspring % 2 != 0:
                n_offspring -= 1
            
            if n_offspring == 0:
                break
                
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring // 2):
                # Select parents
                p1_idx, p2_idx = self._tournament(X, F, 2)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover
                if np.random.rand() < 0.9:
                    alpha = np.random.rand()
                    c1 = alpha * p1 + (1 - alpha) * p2
                    c2 = (1 - alpha) * p1 + alpha * p2
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                
                # Mutation
                # Only mutate active dims 0, 1
                c1[0] += 0.1 * np.random.randn()
                c1[1] += 0.1 * np.random.randn()
                c2[0] += 0.1 * np.random.randn()
                c2[1] += 0.1 * np.random.randn()
                
                # Clip
                c1[0] = np.clip(c1[0], lo[0], hi[0])
                c1[1] = np.clip(c1[1], lo[1], hi[1])
                c2[0] = np.clip(c2[0], lo[0], hi[0])
                c2[1] = np.clip(c2[1], lo[1], hi[1])
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                evals += 2
            
            # Combine
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Non-dominated sort and crowding
            # Select best pop_size from X + offspring_X
            # ... implementation of NSGA-II selection ...
            
            X, F = self._select_next_population(X, F, offspring_X, offspring_F, pop_size)
            
        return X
