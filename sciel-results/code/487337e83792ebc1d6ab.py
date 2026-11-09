    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        rng = np.random.default_rng(42)
        
        # 1. Initialization
        # Use 30% of budget for init
        init_pop_size = int(self.budget * 0.3)
        
        X_init = []
        # Add corners
        for i in range(2**self.dim):
            x = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(self.dim)])
            X_init.append(x)
            
        # Add random
        while len(X_init) < init_pop_size:
            X_init.append(rng.random(self.dim) * span + lo)
            
        X_init = np.array(X_init)
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init])
        used_evals = len(X_init)
        
        # Build Front
        pareto_front = []
        
        def dominates(f_a, f_b):
            return np.all(f_a <= f_b) and np.any(f_a < f_b)
            
        for x, f in zip(X_init, F_init):
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, f):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((x, f))
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not dominates(f, f_p)
                ]
                
        def calc_crowding(front):
            if len(front) <= 2:
                return np.ones(len(front))
            F = np.array([f for _, f in front])
            N = len(F)
            cd = np.zeros(N)
            for m in range(F.shape[1]):
                idx = np.argsort(F[:, m])
                cd[idx[0]] = np.inf
                cd[idx[-1]] = np.inf
                denom = F[idx[-1], m] - F[idx[0], m]
                if denom < 1e-10:
                    denom = 1e-10
                for i in range(1, N-1):
                    cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / denom
            return cd

        sigma = 0.2
        min_sigma = 0.01
        decay = 0.99
        
        stagnation = 0
        
        while used_evals < self.budget:
            if time.time() - start_time > timeout:
                break
                
            if not pareto_front:
                x = rng.random(self.dim) * span + lo
                f = problem(x)
                pareto_front.append((x, f))
                used_evals += 1
                continue
                
            # Selection
            if len(pareto_front) > 1:
                cd = calc_crowding(pareto_front)
                # Tournament: pick 2, choose higher CD
                i1 = rng.integers(len(pareto_front))
                i2 = rng.integers(len(pareto_front))
                parent_idx = i1 if cd[i1] > cd[i2] else i2
            else:
                parent_idx = 0
                
            parent_x, parent_f = pareto_front[parent_idx]
            
            # Mutation
            noise = rng.standard_normal(self.dim) * sigma * span
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            used_evals += 1
            
            # Update
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, child_f):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append((child_x, child_f))
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not dominates(child_f, f_p)
                ]
                stagnation = 0
                sigma *= decay
                if sigma < min_sigma:
                    sigma = min_sigma
            else:
                stagnation += 1
                if stagnation > 50:
                    sigma = min(0.3, sigma * 1.2)
                    stagnation = 0
                    
        return pareto_front
