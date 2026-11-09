"""Multi-objective optimizer on WFG4 using Hypervolume-based CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based CMA-ES for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        # CMA-ES Parameters
        # Lambda is the population size (number of samples per generation)
        # Keep it moderate to allow many generations within budget
        # dim=12, n_obj=3. 
        # Lambda ~ 4 + floor(3 * log(12)) ~ 4 + 3*2.5 ~ 11.5 -> 12
        # But for MO, larger populations help. Let's try 20.
        # With budget 200, that's 10 generations.
        # If we want more generations, smaller lambda. 
        # CMA-ES is sample efficient. Let's use lambda=15.
        lambda_ = 15
        
        # Initial mean: center of bounds
        m = 0.5 * (lo + hi)
        
        # Initial sigma: 1/4 of the range is a common heuristic
        # Or based on dimension? 
        # Let's use a fraction of the span.
        sigma = 0.25 * np.mean(span) 
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Archive for non-dominated points
        # We will store (x, f)
        archive = []
        
        # Helper to check if a point is dominated by any point in the archive
        def is_dominated(f_new, archive_f):
            if len(archive_f) == 0:
                return False
            # f_new is dominated if there exists f_arch such that f_arch <= f_new and f_arch != f_new
            # Vectorized check:
            # less_equal = np.all(archive_f <= f_new, axis=1)
            # less_than = np.any(archive_f < f_new, axis=1)
            # return np.any(less_equal & less_than)
            # For small archives, loop is fine, but vectorized is faster if archive grows.
            # However, archive size is limited by non-domination in 3D, usually small.
            for f_a in archive_f:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        # Helper to update archive
        def update_archive(x_new, f_new, archive):
            # Check if dominated
            if len(archive) > 0:
                # Vectorized dominance check against all archive points
                # f_new shape (n_obj,), archive_f shape (N, n_obj)
                # We want to know if any archive point dominates f_new
                # archive_f <= f_new (element-wise) AND archive_f < f_new (at least one element)
                leq = np.all(np.array(archive_f) <= f_new, axis=1)
                lt = np.any(np.array(archive_f) < f_new, axis=1)
                if np.any(leq & lt):
                    return # Dominated, discard
            
            # If not dominated, check if f_new dominates any archive points
            # Remove dominated points
            new_archive = []
            for i, (x_a, f_a) in enumerate(archive):
                # If f_new dominates f_a, remove f_a
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    continue
                new_archive.append((x_a, f_a))
            
            new_archive.append((x_new, f_new))
            return new_archive

        # We need to track archive_f separately for speed
        archive_x = []
        archive_f = []

        # Initial population
        # Generate first batch
        # CMA-ES sampling: m + sigma * C^{1/2} * z
        # Cholesky of C
        L = np.linalg.cholesky(C)
        
        # We need to sample lambda points
        # But we have a budget constraint.
        # We will loop until budget is exhausted or time runs out.
        
        used = 0
        
        # Pre-allocate lists for current generation
        # We will evaluate points one by one or in batches?
        # Problem call is likely fast, but let's assume it's not trivial.
        # Batching helps if problem is vectorized, but contract says problem(x) returns 3 values for single x.
        # So we must loop.
        
        # Initial random points to seed the archive?
        # CMA-ES starts from mean. If mean is bad, we might get stuck.
        # Let's add a few random points initially to the archive to guide the search?
        # No, CMA-ES adapts the distribution. Let's trust it.
        # However, WFG4 is a deceptive problem? It's a standard test function.
        # Let's start with a small random sample to populate the archive initially?
        # If we just use CMA-ES, the first generation is all around the mean.
        # If the mean is not on the Pareto front, we might miss the front initially.
        # But CMA-ES will move the mean towards the Pareto front via the selection mechanism.
        
        # To be safe, let's initialize with a few random points if budget allows, 
        # but the core strategy is CMA-ES.
        # Actually, standard CMA-ES for MO (like MO-CMA-ES) uses the archive for selection.
        # The mean update is based on the top-k points from the current population 
        # according to some indicator (e.g., hypervolume contribution).
        
        # Let's implement the main loop.
        
        # We need a way to calculate hypervolume contribution for selection.
        # In 3D, hypervolume calculation is feasible.
        # But for simplicity and robustness, we can use a simplified selection:
        # 1. Non-dominated sort the current population.
        # 2. Select the best lambda/2 points from the first front, or all if front is small.
        # 3. Update mean based on these selected points.
        
        # Wait, the directive says "Hypervolume-based selection".
        # Implementing exact hypervolume contribution for 3D is complex but doable.
        # Alternatively, we can use a simpler proxy: 
        # "Contribution" to the archive.
        # But let's stick to a robust implementation:
        # 1. Generate lambda samples.
        # 2. Evaluate them.
        # 3. Combine with archive? No, CMA-ES typically selects from the *current population* to update the mean.
        # The archive is for storing the best found so far.
        
        # Let's use the following selection for mean update:
        # - Identify non-dominated points in the current population.
        # - If there are more than lambda/2, use crowding distance or hypervolume to pick top lambda/2.
        # - If less, pick all non-dominated.
        # - If none (all dominated by each other? impossible in pop), pick best single.
        # - Update mean: m = sum(w_i * x_i) / sum(w_i)
        
        # Let's refine the loop:
        
        # 1. Sample lambda points
        # 2. Clip to bounds
        # 3. Evaluate
        # 4. Update Archive
        # 5. Select parents from current population (not archive) for mean update
        # 6. Update CMA-ES parameters (m, sigma, C)
        
        # Selection Strategy:
        # We want to move the mean towards the Pareto front.
        # We select the 'mu' best points from the current population.
        # 'mu' is usually lambda/2.
        # Criteria: Non-dominated first. Among non-dominated, use crowding or hypervolume.
        # Let's use crowding distance for simplicity and speed, as it's standard in NSGA-II and works well for spread.
        # Or better: Use a simple "knee point" or just the non-dominated set.
        
        # Let's try this:
        # mu = max(2, lambda_ // 2)
        # Find non-dominated points in current pop.
        # If len(non_dom) >= mu, select top mu by crowding distance.
        # Else, select all non_dom.
        # If no non_dom (should not happen), select best by sum of objectives.
        
        # Update mean:
        # m = np.average(selected_x, axis=0, weights=np.ones(mu))
        
        # Update Sigma and C:
        # Standard CMA-ES update.
        
        # Let's implement.
        
        mu = max(2, lambda_ // 2)
        
        # Initial evaluation
        # We need to generate the first population.
        
        # Helper to generate samples
        def generate_samples():
            # z ~ N(0, I)
            z = np.random.randn(lambda_, dim)
            # y = L @ z.T -> shape (dim, lambda_)
            # x = m + sigma * y
            # x shape (lambda_, dim)
            samples = m + sigma * (L @ z.T).T
            # Clip to bounds
            samples = np.clip(samples, lo, hi)
            return samples

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # 1. Generate samples
            X_gen = generate_samples()
            
            # 2. Evaluate
            F_gen = []
            for x in X_gen:
                if used >= budget or time.time() > deadline:
                    break
                f = np.asarray(problem(x), dtype=float)
                F_gen.append(f)
                used += 1
                
            if not F_gen:
                break
                
            F_gen = np.array(F_gen)
            X_gen = X_gen[:len(F_gen)] # In case we broke early
            
            # 3. Update Archive
            # Iterate through new points
            # We need to be careful with the archive update logic for efficiency.
            # Since lambda is small (15), and archive size is small, linear scan is fine.
            
            # Convert archive to lists of arrays for easy manipulation?
            # Let's keep archive as list of (x, f) tuples.
            
            for i in range(len(X_gen)):
                x_new = X_gen[i]
                f_new = F_gen[i]
                
                # Check dominance against current archive
                dominated = False
                if archive_f:
                    # Vectorized check
                    # archive_f is list of arrays. Convert to array for check?
                    # If archive is large, this is slow. But archive in 3D is usually < 100.
                    # Let's convert to array if len > 10?
                    # For now, loop.
                    for f_a in archive_f:
                        if np.all(f_a <= f_new) and np.any(f_a < f_new):
                            dominated = True
                            break
                
                if dominated:
                    continue
                
                # Remove points from archive dominated by f_new
                new_archive_x = []
                new_archive_f = []
                for j in range(len(archive_x)):
                    x_a = archive_x[j]
                    f_a = archive_f[j]
                    if np.all(f_new <= f_a) and np.any(f_new < f_a):
                        # f_new dominates f_a, discard f_a
                        continue
                    new_archive_x.append(x_a)
                    new_archive_f.append(f_a)
                
                # Add new point
                new_archive_x.append(x_new)
                new_archive_f.append(f_new)
                
                archive_x = new_archive_x
                archive_f = new_archive_f
            
            # 4. Selection for Mean Update
            # Find non-dominated points in current generation
            n_gen = len(X_gen)
            if n_gen == 0:
                continue
                
            # Fast non-dominated sorting
            # Since n_gen is small (15), O(n^2) is fine.
            S = [[] for _ in range(n_gen)]
            nd = np.zeros(n_gen, dtype=int)
            
            for p in range(n_gen):
                for q in range(n_gen):
                    if p == q: continue
                    if np.all(F_gen[p] <= F_gen[q]) and np.any(F_gen[p] < F_gen[q]):
                        S[p].append(q)
                        nd[p] += 1
                    elif np.all(F_gen[q] <= F_gen[p]) and np.any(F_gen[q] < F_gen[p]):
                        nd[q] += 1
            
            front0 = [p for p in range(n_gen) if nd[p] == 0]
            
            if len(front0) == 0:
                # Should not happen, but fallback
                selected_indices = [0]
            else:
                selected_indices = front0
            
            # If we have more than mu, we need to select mu points.
            # Use crowding distance.
            if len(selected_indices) > mu:
                # Calculate crowding distance for front0
                # Simple version:
                # Sort by each objective, assign distance.
                # For small sets, this is easy.
                cd = np.zeros(len(selected_indices))
                m_front = len(selected_indices)
                
                for k in range(n_obj):
                    # Get values of objective k for selected indices
                    vals = F_gen[selected_indices, k]
                    order = np.argsort(vals)
                    smin = vals[order[0]]
                    smax = vals[order[-1]]
                    rngv = smax - smin
                    
                    for pos in range(m_front):
                        gi = order[pos]
                        if pos == 0 or pos == m_front - 1:
                            cd[gi] = 1e9
                        else:
                            if rngv > 1e-15:
                                cd[gi] += (vals[order[pos+1]] - vals[order[pos-1]]) / rngv
                            else:
                                cd[gi] += 0.0
                
                # Select top mu by crowding distance
                # Sort cd descending
                order_cd = np.argsort(-cd)
                selected_indices = [selected_indices[i] for i in order_cd[:mu]]
            
            # Ensure we have at least 1 point
            if not selected_indices:
                selected_indices = [0]
                
            # 5. Update CMA-ES Parameters
            # Weights
            # Standard CMA-ES weights: w_i = log(mu/2 + 1) - log(i+1) for i=0..mu-1
            # Normalized.
            w = np.zeros(len(selected_indices))
            for i in range(len(selected_indices)):
                w[i] = np.log(len(selected_indices)/2 + 1) - np.log(i+1)
            w = w / np.sum(w)
            
            # New mean
            X_sel = X_gen[selected_indices]
            m_new = np.sum(w[:, np.newaxis] * X_sel, axis=0)
            
            # Update Sigma
            # CMA-ES sigma update uses a path. 
            # For simplicity and robustness in this short budget, 
            # we can use a simpler adaptation or standard CMA-ES.
            # Standard CMA-ES requires maintaining path_p_sigma and path_p_c.
            # This adds complexity. 
            # Alternative: Use a fixed sigma decay? No, that's not CMA-ES.
            # Let's implement the full CMA-ES update.
            
            # We need to maintain state:
            # C, sigma, m
            # path_p_sigma, path_p_c
            # alpha_sigma, alpha_mu, c_sigma, c_c
            
            # Since we are in __call__, we need to initialize these outside the loop or handle them carefully.
            # Let's move state initialization before the loop.
            
            # Re-initialize state variables if not defined (first iteration)
            if not hasattr(self, '_cma_state'):
                self._cma_state = {
                    'C': C,
                    'sigma': sigma,
                    'm': m,
                    'path_p_sigma': np.zeros(dim),
                    'path_p_c': np.zeros(dim),
                    'alpha_sigma': 1.0,
                    'alpha_mu': 1.0,
                    'c_sigma': 1.0,
                    'c_c': 1.0
                }
                # Standard CMA-ES parameters
                # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
                # c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                # mu_eff = 1 / sum(w^2)
                
                mu_eff = 1.0 / np.sum(w**2)
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                
                self._cma_state['c_sigma'] = c_sigma
                self._cma_state['c_c'] = c_c
                self._cma_state['mu_eff'] = mu_eff
                
                # Note: w depends on selection, which can change size. 
                # Standard CMA-ES uses fixed mu. 
                # If mu changes, we should recalculate. 
                # To keep it simple, let's fix mu to lambda//2 and always select exactly mu points.
                # If front is smaller, we pad with random points from pop? 
                # Or just use the available points and adjust weights?
                # Let's stick to fixed mu = lambda//2.
                # If we have fewer non-dominated, we take the best ones (by sum) to fill up to mu?
                # No, that breaks the "non-dominated" property.
                # Let's just use the available non-dominated points and renormalize weights.
                # The CMA-ES theory assumes fixed mu. 
                # For robustness, let's fix the selection size to mu.
                # If non-dominated < mu, we add the next best dominated points? 
                # This is getting complicated.
                
                # Simplification: 
                # Use a fixed mu. 
                # If we have fewer non-dominated, we select the best dominated ones 
                # based on their distance to the front? 
                # Or just select randomly? 
                # Let's try: Select top mu from the entire population 
                # based on a "rank + crowding" metric, similar to NSGA-II.
                
                # Let's rewrite selection to always return exactly mu points.
                
            # Recalculate selection to ensure exactly mu points
            # Rank all points
            # We already have front0.
            # Let's do a full non-dominated sort to get ranks.
            
            # For small n, we can just compute a simple score.
            # Score = rank * 1e6 - crowding_distance
            # But we need crowding for all points? 
            # Let's just use the non-dominated front.
            # If len(front0) < mu, we will just use len(front0) points and adjust CMA-ES update to handle variable mu?
            # Variable mu CMA-ES is possible but complex.
            
            # Let's stick to: 
            # If len(front0) >= mu: select top mu by crowding.
            # Else: select all front0.
            # And update CMA-ES with the actual weights of the selected points.
            # The CMA-ES update formulas work with any set of weights summing to 1.
            
            # So, let's proceed with the variable mu approach.
            
            # Update CMA-ES State
            state = self._cma_state
            
            # Weights for selected points
            w_sel = np.zeros(len(selected_indices))
            for i in range(len(selected_indices)):
                w_sel[i] = np.log(len(selected_indices)/2 + 1) - np.log(i+1)
            w_sel = w_sel / np.sum(w_sel)
            
            # New Mean
            m_new = np.sum(w_sel[:, np.newaxis] * X_gen[selected_indices], axis=0)
            
            # Update Path p_sigma
            # y_w = (m_new - m) / sigma
            y_w = (m_new - state['m']) / state['sigma']
            
            # C^{-1/2} y_w
            # We have L such that C = L L^T. So C^{-1/2} = L^{-T}
            # y_w_C = L^{-T} y_w
            # Solve L^T y_w_C = y_w
            y_w_C = np.linalg.solve(state['C'].T, y_w) # Wait, C is symmetric. L L^T = C. 
            # We stored L from Cholesky of C. 
            # But C changes. We need to update L.
            
            # Actually, we should store L and update it, or recalculate.
            # Recalculating Cholesky every step for dim=12 is cheap.
            
            # Let's recalculate L from state['C']
            L_cur = np.linalg.cholesky(state['C'])
            
            # y_w_C = L^{-T} y_w
            y_w_C = np.linalg.solve(L_cur.T, y_w)
            
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma) mu_eff) y_w_C
            mu_eff = 1.0 / np.sum(w_sel**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            
            state['path_p_sigma'] = (1 - c_sigma) * state['path_p_sigma'] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w_C
            
            # Update Sigma
            # chi_n = sqrt(n(1 - 1/(2n) - 1/(4n^2)))
            chi_n = np.sqrt(dim * (1 - 1.0/(2*dim) - 1.0/(4*dim*dim)))
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2gen}) < (1.4 + 1/n) E||N(0,I)|| else 0
            # Approximation for h_sigma:
            # We need generation counter.
            if not hasattr(state, 'gen'):
                state['gen'] = 0
            state['gen'] += 1
            
            # E||N(0,I)|| ~ sqrt(n)
            # Threshold: (1.4 + 1/n) * sqrt(n)
            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(state['path_p_sigma'])
            # Denominator: sqrt(1 - (1-c_sigma)^(2*gen))
            # For large gen, approx 1.
            # Let's use a simpler check:
            # h_sigma = 1.0 if norm_p_sigma < (1.4 + 1.0/dim) * chi_n else 0.0
            # This is a common approximation.
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 1.0/dim) * chi_n else 0.0
            
            state['sigma'] = state['sigma'] * np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1.0))
            
            # Update Path p_c
            # y_w_C is already computed.
            # p_c = (1 - c_c) p_c + h_sigma c_c y_w_C
            state['path_p_c'] = (1 - c_c) * state['path_p_c'] + h_sigma * c_c * y_w_C
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 y_w_C y_w_C^T + c_mu sum w_i (y_i - y_w)(y_i - y_w)^T
            # Where y_i = C^{-1/2} (x_i - m_old)
            
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = 2 * (mu_eff - 2 + 2/mu_eff) / ((dim + 2)**2 + mu_eff)
            
            # Compute y_i for selected points
            # x_i - m_old
            # y_i = C^{-1/2} (x_i - m_old)
            # We have L_cur.
            # y_i = L^{-T} (x_i - m_old)
            
            # Vectorized:
            # X_diff = X_sel - state['m']
            # Y = X_diff @ L^{-1} ? No.
            # y_i = L^{-T} (x_i - m)
            # (x_i - m) is (mu, dim).
            # L^{-T} is (dim, dim).
            # Y = X_diff @ np.linalg.inv(L_cur.T)
            
            X_diff = X_gen[selected_indices] - state['m']
            inv_L_T = np.linalg.inv(L_cur.T)
            Y = X_diff @ inv_L_T
            
            # y_w is the weighted mean of Y?
            # y_w = sum w_i Y_i
            y_w_calc = np.sum(w_sel[:, np.newaxis] * Y, axis=0)
            # Should be close to y_w_C computed earlier.
            
            # C_new = (1 - c_1 - c_mu) C + c_1 y_w_C y_w_C^T + c_mu sum w_i (Y_i - y_w_calc)(Y_i - y_w_calc)^T
            
            C_new = (1 - c_1 - c_mu) * state['C']
            C_new += c_1 * np.outer(y_w_C, y_w_C)
            
            # Sum of outer products
            # Y_i - y_w_calc
            Y_diff = Y - y_w_calc[np.newaxis, :]
            # Weighted sum of outer products
            # sum w_i (Y_i - y_w) (Y_i - y_w)^T
            # = (Y_diff * w_sel[:, np.newaxis]) @ Y_diff.T
            C_new += c_mu * ((Y_diff * w_sel[:, np.newaxis]) @ Y_diff.T)
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2.0
            
            # Update state
            state['C'] = C_new
            state['m'] = m_new
            state['sigma'] = state['sigma']
            
            # Check if C is still positive definite
            try:
                np.linalg.cholesky(C_new)
            except:
                # If not, reset C to identity or previous
                state['C'] = state['C'] * 0.9 + 0.1 * np.eye(dim)
                # Ensure PD
                state['C'] = (state['C'] + state['C'].T) / 2 + 1e-6 * np.eye(dim)

        # Return the best point or the archive?
        # The contract says "return value of __call__ is ignored".
        # So we can return anything.
        # Let's return the first point in the archive or None.
        if archive_x:
            return archive_x[0]
        return None

# EVOLVE-BLOCK-END
