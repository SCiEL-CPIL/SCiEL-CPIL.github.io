"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar


class Optimizer:
    """Hybrid CMA-ES + Powell + Adaptive 1D Polish Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= N:
                raise StopIteration("Budget exceeded")
            if time.time() > deadline:
                raise StopIteration("Time limit exceeded")
            
            calls_made += 1
            fx = func(x)
            
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Helper for 1D minimization along a direction
        def line_search_1d(center, direction, bracket_range=5.0):
            """
            Performs a 1D minimization along 'direction' starting from 'center'.
            Uses scipy's minimize_scalar with bounded method.
            """
            nonlocal best_x, best_f
            
            def f1d(t):
                x_new = center + t * direction
                return eval_func(x_new)
            
            best_t = 0.0
            best_f_local = float('inf')
            
            try:
                # Adaptive bracket: start with a moderate range.
                res = minimize_scalar(f1d, bounds=(-bracket_range, bracket_range), method='bounded', options={'maxiter': 30, 'xatol': 1e-8})
                if res.fun < best_f_local:
                    best_f_local = res.fun
                    best_t = res.x
                    x_candidate = center + best_t * direction
                    if best_f_local < best_f:
                        best_f = best_f_local
                        best_x = x_candidate.copy()
            except StopIteration:
                pass
            except Exception:
                pass

            return best_t, best_f_local

        # 1. Coarse Global Sampling to identify the best basin
        rng = np.random.default_rng(42)
        
        # Define candidate points
        candidates = [np.zeros(d)]
        # Add random points
        for _ in range(8):
            candidates.append(rng.uniform(-4.5, 4.5, size=d))
        # Add points near boundaries
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        # Evaluate all candidates
        for c in candidates:
            if calls_made >= N or time.time() > deadline:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. CMA-ES Phase: Robustly navigate the anisotropic landscape
        # CMA-ES is good at adapting to the narrow valley of Bent Cigar
        # We run it for a portion of the budget to get close to the minimum
        cma_budget = int(N * 0.6)  # Use 60% of budget for CMA-ES
        cma_limit = calls_made + cma_budget
        
        if calls_made < cma_limit and time.time() < deadline:
            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(d))  # 4 + 3*ln(5) ≈ 4 + 4.83 = 8
            mu = lambda_ // 2
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Learning rates
            cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = 2 / ((d + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))

            # Initial state
            x = best_x.copy()
            sigma = 2.0
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)

            # Track fitness history for stagnation detection
            no_improve = 0
            stagnation_threshold = 20  # Lower threshold to allow quicker exit to Powell

            # Main CMA-ES loop
            while calls_made < cma_limit:
                # Check time
                if time.time() > deadline:
                    break

                # Check if we have enough budget for a full generation
                if calls_made + lambda_ > cma_limit:
                    break

                # Generate population
                population = []
                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    try:
                        y = x + sigma * rng.multivariate_normal(np.zeros(d), C)
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        population.append((fy, y))
                    except StopIteration:
                        break
                    except Exception:
                        continue

                if len(population) < lambda_:
                    break

                # Sort by fitness
                population.sort(key=lambda t: t[0])

                # Check for improvement
                if population[0][0] < best_f - 1e-10:
                    no_improve = 0
                else:
                    no_improve += 1

                # Stagnation handling: if stuck for too long, break to Powell
                if no_improve > stagnation_threshold:
                    break

                # Update mean (weighted average of best mu solutions)
                w_mean = np.zeros(d)
                for i in range(mu):
                    w_mean += weights[i] * population[i][1]
                x_new = w_mean

                # Update sigma (CSA)
                c_sigma = np.sqrt((1 + 2 * max(0, cs - 1) * cs / (cs + 1)) * mueff / (d + 2))
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff / lambda_) * (x_new - x) / sigma
                chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                c_c = np.sqrt((1 + 2 * max(0, cc - 1) * cc / (cc + 1)) * mueff / (d + 2))
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mueff / lambda_) * (x_new - x) / sigma

                # Rank-one update
                rank_one = np.outer(p_c, p_c)
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n < (1.4 + 2 / (d + 1)) * (1 - c_sigma) else 0
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    z = (population[i][1] - x) / sigma
                    rank_mu += weights[i] * np.outer(z, z)

                C = (1 + c1 * h_sigma * (cc - 1 + cc) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                C = (C + C.T) / 2  # Symmetrize

                # Update mean
                x = x_new

                # Ensure sigma doesn't go too small or too large
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

        # 3. Powell Phase: Exploit the directional information
        # Powell is very efficient for smooth, convex-like functions once in the basin
        if calls_made < N and time.time() < deadline:
            try:
                minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100, # Limit iterations
                        'ftol': 1e-12,
                        'xatol': 1e-8,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 4. Adaptive 1D Coordinate Descent Polish
        # This phase cleans up the residual error, especially in the narrow valley direction
        if calls_made < N and time.time() < deadline:
            # Determine how many rounds we can afford
            # Each round is d 1D searches. Each search takes ~10-30 calls.
            # Let's aim for 3-5 rounds if budget allows.
            max_rounds = 5
            for r in range(max_rounds):
                if calls_made >= N or time.time() > deadline:
                    break
                
                for i in range(d):
                    if calls_made >= N or time.time() > deadline:
                        break
                    
                    direction = np.zeros(d)
                    direction[i] = 1.0
                    
                    # Use a smaller bracket for polishing
                    # As we get closer, the step size needed is smaller
                    bracket = 2.0 / (r + 1)
                    _, _ = line_search_1d(best_x, direction, bracket_range=bracket)

        return best_x, best_f
# EVOLVE-BLOCK-END
