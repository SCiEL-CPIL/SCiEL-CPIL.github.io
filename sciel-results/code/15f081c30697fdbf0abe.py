    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290
        
        dim = self.dim
        total_budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Global Exploration
        n_explore = 50
        candidates = []
        for _ in range(n_explore):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)
            candidates.append((fx, x))
        
        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Nelder-Mead
        # Use top 3 distinct points
        starts = []
        min_dist = 1.0
        for fx, x in candidates:
            if len(starts) >= 3:
                break
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((fx, x))
        
        for f_start, x_start in starts:
            if n_calls >= total_budget or time.time() > deadline:
                break
            
            # Initialize simplex
            x0 = np.array(x_start, dtype=float)
            simplex = [x0.copy()]
            simplex_f = [f_start] # Reuse f_start? No, we need to be sure.
            # Actually, x0 was evaluated in Phase 1. We can reuse f_start.
            # But we need dim more points.
            
            # Build simplex
            for j in range(dim):
                point = x0.copy()
                # Step size 0.05 is good for 5D
                point[j] += 0.05
                if point[j] > 5: point[j] = 5
                if point[j] < -5: point[j] = -5
                simplex.append(point)
            
            # Evaluate new points
            for j in range(1, dim + 1):
                if n_calls >= total_budget or time.time() > deadline:
                    break
                fx = func(simplex[j])
                record(simplex[j], fx)
                simplex_f.append(fx)
            
            if len(simplex_f) < dim + 1:
                continue

            # Nelder-Mead Loop
            max_iter = 100
            for iter in range(max_iter):
                if n_calls >= total_budget or time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Convergence check
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_ref = centroid + (centroid - worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                record(x_ref, f_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + 2 * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    record(x_exp, f_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < worst_f:
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        x_con = centroid - 0.5 * (centroid - worst)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    record(x_con, f_con)
                    
                    if f_con < min(f_ref, worst_f):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            if n_calls >= total_budget or time.time() > deadline:
                                break
                            f_shr = func(simplex[j])
                            record(simplex[j], f_shr)
                            simplex_f[j] = f_shr
                        if n_calls >= total_budget or time.time() > deadline:
                            break

        return best_x, best_f
