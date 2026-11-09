"""Hybrid k-NN Surrogate Optimization with Adaptive ES Refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid k-NN Surrogate Optimization with Adaptive ES Refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Storage for surrogate model
        X_eval = []
        Y_eval = []

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Update surrogate data
            X_eval.append(x)
            Y_eval.append(f)
            return f

        # --- Phase 1: Initialization ---
        # Use Halton sequence for space-filling initialization
        n_init = 30
        init_results = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.05, 0.05, dim)
            point = np.clip(point, -5, 5)

            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        if best_x is None:
            return None, np.inf

        # Sort init results to identify good starting points
        init_results.sort(key=lambda x: x[1])

        # --- Phase 2: Hybrid BO + ES Loop ---
        # We alternate between surrogate-guided exploration and local ES exploitation
        # k-NN parameters
        k_nn = 5
        
        # Epsilon for exploration
        epsilon = 0.5
        epsilon_decay = 0.95
        
        # Local refinement trigger
        steps_since_last_global_best = 0
        last_global_best_f = best_f
        
        # Main loop
        while calls_made < budget and time.time() < deadline:
            
            # Check if we should perform local refinement
            # If we found a new global best recently, or if we are stuck
            if best_f < last_global_best_f - 1e-6:
                steps_since_last_global_best = 0
                last_global_best_f = best_f
            else:
                steps_since_last_global_best += 1

            # Trigger local refinement if we have a new best or are stuck for a while
            should_refine = False
            if steps_since_last_global_best == 0:
                # Just found a new best, refine it
                should_refine = True
            elif steps_since_last_global_best > 20 and calls_made < budget - 50:
                # Stuck for a while, try to refine current best more aggressively
                should_refine = True

            if should_refine:
                # --- Local Refinement with Adaptive (1+1)-ES ---
                x_es = best_x.copy()
                f_es = best_f
                sigma = 0.5
                max_es_steps = 10
                es_steps = 0
                
                while es_steps < max_es_steps and calls_made < budget and time.time() < deadline:
                    y = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    es_steps += 1
                    
                    if fy < f_es:
                        x_es, f_es = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 0.01, 5.0)
                    
                # Update best if ES found something better
                if f_es < best_f:
                    best_f = f_es
                    best_x = x_es.copy()
                    steps_since_last_global_best = 0
                    last_global_best_f = best_f

            else:
                # --- Surrogate-Guided Exploration ---
                if len(X_eval) < k_nn:
                    # Need more data for k-NN
                    x_new = rng.uniform(-5, 5, dim)
                    evaluate(x_new)
                    continue

                # Generate candidate points
                n_candidates = 50
                candidates = []
                
                # 50% random, 50% around best
                for i in range(n_candidates):
                    if i < n_candidates // 2:
                        x_cand = rng.uniform(-5, 5, dim)
                    else:
                        x_cand = np.clip(best_x + rng.standard_normal(dim) * 1.0, -5, 5)
                    candidates.append(x_cand)
                
                candidates = np.array(candidates)

                # Compute k-NN predictions and distances
                X_eval_arr = np.array(X_eval)
                Y_eval_arr = np.array(Y_eval)
                
                # Vectorized distance calculation
                # Shape: (n_candidates, n_eval)
                dists = np.linalg.norm(candidates[:, np.newaxis, :] - X_eval_arr[np.newaxis, :, :], axis=2)
                
                # k-NN Prediction
                # Get indices of k nearest neighbors
                k_nn_idx = np.argpartition(dists, k_nn, axis=1)[:, :k_nn]
                # Get distances for these neighbors
                k_nn_dists = np.take_along_axis(dists, k_nn_idx, axis=1)
                # Weights: Inverse distance (add small epsilon to avoid div by 0)
                weights = 1.0 / (k_nn_dists + 1e-8)
                weights /= np.sum(weights, axis=1, keepdims=True)
                
                # Get Y values for neighbors
                k_nn_Y = np.take_along_axis(Y_eval_arr[np.newaxis, :], k_nn_idx, axis=1)
                
                # Predicted values
                pred_Y = np.sum(weights * k_nn_Y, axis=1)
                
                # Exploration Metric: Min distance to any evaluated point
                min_dists = np.min(dists, axis=1)
                
                # Epsilon-Greedy Selection
                if rng.random() < epsilon:
                    # Explore: Maximize distance to existing points
                    # To avoid picking points that are far but in bad regions, 
                    # we can penalize high predicted values slightly or just pick max dist.
                    # Let's pick the candidate with max min_dist.
                    best_idx = np.argmax(min_dists)
                else:
                    # Exploit: Minimize predicted value
                    best_idx = np.argmin(pred_Y)
                
                x_new = candidates[best_idx]
                f_new = evaluate(x_new)
                
                # Decay epsilon
                epsilon *= epsilon_decay
                epsilon = max(epsilon, 0.1)

        # --- Phase 3: Final Polish with Nelder-Mead ---
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_nm = best_x.copy()
            f_nm = best_f

            delta = 0.2
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [f_nm]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i + 1])
                simplex_f.append(f_new)

            if len(simplex_f) < dim + 1:
                return best_x, best_f

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = min(20, budget - calls_made)
            no_improve_count = 0

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break

                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                max_dist = 0.0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-6:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    no_improve_count = 0
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    no_improve_count = 0
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        no_improve_count = 0
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve_count += 1

                if no_improve_count > 5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
