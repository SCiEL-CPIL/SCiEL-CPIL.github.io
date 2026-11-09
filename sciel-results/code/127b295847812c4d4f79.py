import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave 40s safety margin
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def cma_es(x_start, sigma_init, max_evals):
            """CMA-ES for the main search phase."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if evals_used >= budget or time.time() > deadline:
                return
            
            # CMA-ES parameters for dim=5
            # Larger population for better exploration in short budget
            lam = 12
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            w_full = np.concatenate([w, np.zeros(lam - mu)])
            
            mueff = 1.0 / np.sum(w**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
            
            # Initialize state
            m = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = (mueff + 2) / (dim + 2)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Initial population
            pops = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                pops.append(m + sigma * z)
            
            # Evaluate initial population
            fs = []
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    return
                f_val = try_func(pops[i])
                fs.append(f_val)
            
            # Sort
            order = np.argsort(fs)
            pops = [pops[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            m_new = sum(w[i] * pops[i] for i in range(mu))
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (m_new - m)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim + 1)) * chiN:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update covariance
            diff = (m_new - m) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff
            delta_sig = (1 - (1 - c_c)**2) * np.dot(pc, pc) / dim - 1
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sig * np.eye(dim))
            C += c_mu * sum(w[i] * np.outer(pops[i] - m, pops[i] - m) / sigma**2 for i in range(mu))
            
            m = m_new
            
            # Main loop
            while evals_used < budget and time.time() < deadline and evals_used < max_evals:
                # Generate new population
                pops = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    # Use Cholesky decomposition for sampling from C
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(dim)
                    pops.append(m + sigma * L @ z)
                
                # Evaluate
                fs = []
                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        return
                    f_val = try_func(pops[i])
                    fs.append(f_val)
                
                # Sort
                order = np.argsort(fs)
                pops = [pops[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update mean
                m_new = sum(w[i] * pops[i] for i in range(mu))
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (m_new - m)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim + 1)) * chiN:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
                
                # Update covariance
                diff = (m_new - m) / sigma
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff
                delta_sig = (1 - (1 - c_c)**2) * np.dot(pc, pc) / dim - 1
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sig * np.eye(dim))
                C += c_mu * sum(w[i] * np.outer(pops[i] - m, pops[i] - m) / sigma**2 for i in range(mu))
                
                m = m_new

        def nelder_mead(x_start, initial_step=0.5):
            """Nelder-Mead for local refinement."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if time.time() > deadline or evals_used >= budget:
                return
            
            # Initialize simplex
            step = initial_step
            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            fs = []
            for i in range(dim + 1):
                if evals_used >= budget or time.time() > deadline:
                    return
                f_val = try_func(simplex[i])
                fs.append(f_val)
                
                # Update global best immediately
                if f_val < best_f:
                    best_f = f_val
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha_nm = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            # Run until budget/time exhausted or converged
            while evals_used < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence
                if fs[-1] - fs[0] < 1e-10:
                    break
                
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                f_ref = try_func(x_ref)
                
                if np.isinf(f_ref):
                    break
                    
                if f_ref < fs[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_func(x_exp)
                    if np.isinf(f_exp):
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                    elif f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = try_func(x_con)
                    if np.isinf(f_con):
                        break
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fs[i] = try_func(simplex[i])
                            if np.isinf(fs[i]):
                                break
                        if any(np.isinf(f) for f in fs):
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()

        # Phase 1: Exploration with CMA-ES (70% of budget)
        cma_budget = int(budget * 0.7)
        cma_es(np.zeros(dim), sigma_init=2.0, max_evals=cma_budget)
        
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Local refinement with Nelder-Mead (30% of budget)
        remaining_budget = budget - evals_used
        nm_budget = int(remaining_budget * 0.3)
        
        # Start NM from current best
        nelder_mead(best_x, initial_step=0.5)
        
        return best_x, best_f
