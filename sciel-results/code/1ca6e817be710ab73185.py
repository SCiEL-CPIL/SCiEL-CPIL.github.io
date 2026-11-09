"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Hyperband-style Successive Halving with CMA-ES for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        def cma_es_step(x, sigma, C, pc, ps, pop, pop_f, weights, lambda_, mu, mueff, cs, ds, cc, c1, cmu):
            """
            Performs one iteration of CMA-ES.
            Returns updated state and list of new points to evaluate.
            """
            nonlocal best_x, best_f, total_calls

            # Sort population by fitness
            order = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in order]
            pop_f_sorted = [pop_f[i] for i in order]

            # Update mean
            x_old = x
            x_new = sum(weights[i] * pop_sorted[i] for i in range(mu))

            # Update step size control (CSA)
            # Note: total_calls is global, but we need the number of generations for this specific run?
            # Actually, the standard CMA-ES uses the generation number t for the hsig check denominator.
            # Since we are doing "steps" in a loop, we can track a local generation counter or approximate.
            # For simplicity in this "step" function, we assume the caller manages the generation count or we use a fixed heuristic.
            # However, standard CMA-ES hsig check uses (1 - (1-cs)^(2t)).
            # Let's pass a generation counter 't' to this function to be accurate.
            # Wait, the signature above didn't include 't'. Let's add it or handle it internally?
            # To keep the signature simple for the caller, let's assume we can calculate it or pass it.
            # I will modify the function to accept 't' (generation number).
            
            # Re-define to accept t
            # This is a draft, I will implement the full class below properly.
            pass

        # To avoid complex state passing, I will implement a lightweight CMA-ES runner class inside __call__
        
        class CmaEsRunner:
            def __init__(self, x0, sigma0, rng_local):
                self.x = x0.copy()
                self.sigma = sigma0
                self.lambda_ = 4 + int(3 * np.log(n))
                self.mu = self.lambda_ // 2
                self.weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
                self.weights /= self.weights.sum()
                self.mueff = 1.0 / self.weights.sum()
                
                self.C = np.eye(n)
                self.pc = np.zeros(n)
                self.ps = np.zeros(n)
                
                self.cs = (self.mueff + 2) / (n + self.mueff + 5)
                self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
                self.cc = (4 + self.mueff / (n + 4)) / ((n + 2) + self.mueff / 2)
                self.c1 = (self.mueff - 2 + 2 / (n + self.mueff)) / ((n + 2) ** 2 + self.mueff - 2)
                self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 2 / (self.mueff)) / ((n + 2) ** 2 + self.mueff))
                
                self.rng_local = rng_local
                self.pop = []
                self.pop_f = []
                self.t = 0
                self.best_f_local = float('inf')
                self.best_x_local = x0.copy()
                self.initialized = False

            def initialize(self):
                for i in range(self.lambda_):
                    z = self.rng_local.standard_normal(n)
                    xi = self.x + self.sigma * z
                    self.pop.append(xi)
                self.initialized = True

            def evaluate_population(self):
                """Evaluates current population. Returns list of (x, f)."""
                results = []
                for xi in self.pop:
                    if total_calls >= budget or time.time() > deadline:
                        return results, False
                    yi = func(xi)
                    total_calls += 1
                    results.append((xi, yi))
                    if yi < self.best_f_local:
                        self.best_f_local = yi
                        self.best_x_local = xi.copy()
                    if yi < best_f:
                        best_f = yi
                        best_x = xi.copy()
                self.pop_f = [y for _, y in results]
                return results, True

            def step(self):
                """
                Performs one CMA-ES iteration.
                Returns: (new_population, success_flag)
                """
                if not self.initialized:
                    self.initialize()
                    return self.pop, True

                self.t += 1
                x_old = self.x
                # Update mean
                order = np.argsort(self.pop_f)
                pop_sorted = [self.pop[i] for i in order]
                f_sorted = [self.pop_f[i] for i in order]
                
                self.x = sum(self.weights[i] * pop_sorted[i] for i in range(self.mu))
                
                # Update paths
                self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma * self.C @ (self.x - x_old)
                hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.t))) < (1 + 2 / (n + 1)) * math.sqrt(n + 1)
                
                self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.x - x_old) / self.sigma
                
                # Update covariance
                delta_d = (1 - self.c1) * (1 + self.cmu * self.mueff)
                rank_one = np.outer(self.pc, self.pc)
                rank_mu = sum(self.weights[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old) for i in range(self.mu))
                self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu
                
                # Update sigma
                self.sigma *= math.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / math.sqrt(n) - 1))
                
                # Check termination
                if self.sigma < 1e-12 or np.max(np.abs(self.C)) < 1e-12:
                    return None, False

                # Generate new population
                new_pop = []
                for i in range(self.lambda_):
                    z = self.rng_local.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(self.C)
                        xi = self.x + self.sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        diag = np.sqrt(np.diag(self.C))
                        xi = self.x + self.sigma * (diag * z)
                    new_pop.append(xi)
                    
                self.pop = new_pop
                return self.pop, True

        # Phase 1: Initial Sampling (Latin Hypercube)
        # Use 20% of budget for global exploration to seed Hyperband
        n_init = max(20, budget // 5)
        
        X_init = np.zeros((n_init, n))
        for j in range(n):
            X_init[:, j] = (np.arange(n_init) + rng.uniform(size=n_init)) / n_init
            rng.shuffle(X_init[:, j])
        X_init = X_init * 10 - 5 # Scale to [-5, 5]
        
        y_init = np.zeros(n_init)
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline:
                break
            x = X_init[i]
            y = func(x)
            total_calls += 1
            y_init[i] = y
            if y < best_f:
                best_f = y
                best_x = x.copy()

        # Phase 2: Hyperband Successive Halving
        # We will run multiple rounds. In each round, we start K new CMA-ES instances.
        # We let them run for 'r' evaluations. Then we keep the top K/2 based on best_f_local.
        # We repeat until budget is exhausted.
        
        # Parameters for Hyperband
        # s_max: max number of successive halving steps
        # K0: initial number of trials
        # r: initial budget per trial
        # Total budget N = sum over s of (K_s * r_s)
        
        # Let's use a simplified Hyperband:
        # We have remaining budget.
        # We will run "rounds".
        # In round 1, we start 10 trials with 10 evals each. Keep best 5.
        # In round 2, we start 10 new trials? No, Hyperband usually restarts or continues.
        # Standard Hyperband:
        # For s in 0..s_max:
        #   K = ceil(N / (s_max + 1) * (s_max - s + 1))
        #   r = floor(N / K)
        #   ...
        
        # Given the small budget (500), let's use a fixed structure that fits well.
        # s_max = 2
        # s=0: K=250, r=2 (Too many small trials, CMA-ES needs ~10-20 to start)
        # Let's adjust for CMA-ES specific needs. CMA-ES needs at least lambda (9) evaluations to start.
        # So r must be >= 9.
        
        # Let's use a manual schedule:
        # Round 1: Start 10 instances. Budget per instance: 10. Total: 100. Keep best 5.
        # Round 2: Start 10 NEW instances? Or continue the 5?
        # Hyperband typically starts NEW trials in each "s" level, but here we have a single budget pool.
        # A better approach for this specific "multi-start" context:
        # We maintain a list of active CMA-ES runners.
        # Initially, we have 0 active.
        # We will use the remaining budget (400) to run a Successive Halving loop.
        
        active_runners = []
        # We need to initialize them.
        # Let's pick top 10 points from initial sampling as seeds.
        top_k = 10
        best_indices = np.argsort(y_init)[:top_k]
        
        # Create runners for these seeds
        for i in range(top_k):
            x_start = X_init[best_indices[i]].copy()
            sigma_start = 0.5
            runner = CmaEsRunner(x_start, sigma_start, np.random.default_rng(100 + i))
            runner.initialize()
            active_runners.append(runner)
            
        # Now we have 10 runners, each needing 9 evals to complete their first population.
        # Total needed: 90.
        
        # Successive Halving Loop
        # We will run in "cycles".
        # In each cycle, we give each active runner 'step_budget' evaluations.
        # After the cycle, we rank them by best_f_local and keep the top half.
        
        # Let's define the schedule:
        # Cycle 1: 10 runners * 10 evals = 100. Keep 5.
        # Cycle 2: 5 runners * 20 evals = 100. Keep 2.
        # Cycle 3: 2 runners * 40 evals = 80. Keep 1.
        # Cycle 4: 1 runner * remaining (120) evals.
        # Total: 100 + 100 + 80 + 120 = 400. Perfect fit for remaining budget.
        
        schedule = [
            (10, 10),  # (num_runners, evals_per_runner)
            (5, 20),
            (2, 40),
            (1, 100)   # Use up to remaining
        ]
        
        current_runners = active_runners
        
        for num_to_keep, evals_per_runner in schedule:
            if not current_runners:
                break
            if total_calls >= budget or time.time() > deadline:
                break
                
            # Evaluate current population for all active runners
            # Note: Some runners might have just been initialized and not evaluated yet.
            # We need to ensure we evaluate them.
            
            # We will evaluate in batches to respect the budget.
            # For each runner, we need to complete its current population evaluation if not done,
            # then perform steps.
            
            # Actually, the CmaEsRunner state is:
            # 1. Initialize (pop created, not evaluated)
            # 2. Evaluate pop (pop_f filled)
            # 3. Step (new pop created)
            
            # We need to be careful.
            # Let's assume all active runners are in "Step" mode (pop ready to evaluate).
            # If a runner was just initialized, it needs evaluation first.
            
            # To simplify, we will just run evals_per_runner evaluations for each runner.
            # If a runner is in "needs eval" state, it will eval.
            # If in "needs step" state, it will step (which generates new pop, but doesn't eval).
            # This is tricky.
            
            # Better approach:
            # 1. Ensure all active runners have their current population evaluated.
            # 2. Perform (evals_per_runner // lambda_) steps? No, lambda_ is 9.
            # 3. Just do a loop of evaluations.
            
            # Let's refine the runner to be "evaluation-driven".
            # We will just call a method `run_for_budget(k)` which performs k evaluations.
            
            # I will implement `run_for_budget` in the class or simulate it here.
            
            # Let's simulate it here to avoid rewriting the class structure too much.
            
            evals_done_this_cycle = 0
            max_evals_this_cycle = num_to_keep * evals_per_runner
            
            # We need to cycle through the runners.
            # Round-robin: give 1 eval to runner 1, then runner 2, etc.
            # This ensures fair treatment and allows early termination if budget runs out.
            
            runner_idx = 0
            while evals_done_this_cycle < max_evals_this_cycle and total_calls < budget and time.time() < deadline:
                runner = current_runners[runner_idx % len(current_runners)]
                
                # Check if runner is dead (terminated internally)
                if runner is None:
                    current_runners.remove(runner)
                    if not current_runners:
                        break
                    continue
                
                # If runner has a population to evaluate
                if runner.pop and runner.pop_f is not None and len(runner.pop_f) < len(runner.pop):
                    # This logic is flawed in my previous class design.
                    # Let's stick to the previous pattern:
                    # 1. If not initialized: Initialize (no cost)
                    # 2. If pop_f is empty/None: Evaluate pop (cost lambda_)
                    # 3. If pop_f is filled: Step (cost 0, generates new pop)
                    
                    if not runner.initialized:
                        runner.initialize()
                        # After initialize, pop is ready to evaluate.
                        # We don't consume budget here.
                    elif runner.pop_f is None or len(runner.pop_f) == 0:
                        # Evaluate population
                        # We need to evaluate one by one to respect global budget
                        # But CMA-ES step needs the WHOLE population.
                        # So we must evaluate the whole population before stepping.
                        
                        # Check if we have enough budget for the whole population
                        if total_calls + len(runner.pop) > budget:
                            # Can't afford full pop. Just evaluate as many as we can?
                            # No, CMA-ES breaks if incomplete.
                            # Better to stop this runner or skip.
                            # For simplicity, if we can't afford a full pop, we stop this cycle.
                            break
                        
                        results, success = runner.evaluate_population()
                        if not success:
                            break
                        evals_done_this_cycle += len(runner.pop)
                        
                    else:
                        # pop_f is filled. Perform step.
                        new_pop, success = runner.step()
                        if not success:
                            # Runner terminated
                            current_runners.remove(runner)
                            if not current_runners:
                                break
                        else:
                            # New pop is ready. pop_f is now stale/empty effectively.
                            # We need to clear pop_f so next time we evaluate.
                            runner.pop_f = None
                            # No budget consumed for step itself.
                            # To avoid infinite loop of steps without eval, we should
                            # ensure we evaluate soon.
                            # The round-robin will naturally come back to it.
                            # But we might step multiple times? No, step requires pop_f.
                            # So next time we visit this runner, pop_f is None, so we evaluate.
                            pass
                            
                runner_idx += 1
                
            # End of cycle evaluation.
            # Now rank runners by best_f_local and keep top num_to_keep
            if current_runners:
                current_runners.sort(key=lambda r: r.best_f_local)
                if len(current_runners) > num_to_keep:
                    current_runners = current_runners[:num_to_keep]

        # Final polish with the best runner
        if current_runners and total_calls < budget and time.time() < deadline:
            best_runner = current_runners[0]
            # Run remaining budget
            while total_calls < budget and time.time() < deadline:
                if not best_runner.initialized:
                    best_runner.initialize()
                elif best_runner.pop_f is None or len(best_runner.pop_f) == 0:
                    if total_calls + len(best_runner.pop) > budget:
                        break
                    results, success = best_runner.evaluate_population()
                    if not success:
                        break
                else:
                    new_pop, success = best_runner.step()
                    if not success:
                        break
                    best_runner.pop_f = None

        return best_x, best_f
# EVOLVE-BLOCK-END
