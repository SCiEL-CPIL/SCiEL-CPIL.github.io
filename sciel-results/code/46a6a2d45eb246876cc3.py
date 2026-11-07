"""
Optimization strategy:
1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as the primary
   optimizer. CMA-ES is a state-of-the-art derivative-free optimizer that
   adapts its sampling distribution based on the history of successful
   solutions, making it particularly effective for multimodal functions
   like Rastrigin.
2. Multi-restart: If CMA-ES stagnates (no improvement for a while), we
   restart from a different random point or from the best found so far.
3. Final polish: A fine-grained (1+1)-ES around the global best to squeeze
   out the last bit of precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s, we stop at 290s to be safe
        deadline = time.time() + 290.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # --- CMA-ES Implementation ---
        class CMAES:
            def __init__(self, x0, sigma0, budget_left, dim):
                self.x = x0.copy()
                self.sigma = sigma0
                self.dim = dim
                self.budget = budget_left
                # Population size
                self.lam = max(4, int(4 + 3 * np.log(dim)))
                # Weights
                weights = np.log(self.lam / 2 + 0.5) - np.log(1 + np.arange(self.lam))
                self.weights = weights / weights.sum()
                self.mu = int(np.sum(self.weights > 0))
                self.mu_eff = 1.0 / np.sum(self.weights[:self.mu] ** 2)
                # Step sizes for adaptation
                self.c_sigma = (self.mu_eff + 2) / (dim + 4 + self.mu_eff)
                self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
                self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
                self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
                self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1. / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))
                # State variables
                self.pc = np.zeros(dim)
                self.ps = np.zeros(dim)
                self.C = np.eye(dim)
                self.inv_C = np.eye(dim)
                self.chi_N = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
                self.fitness_history = []
                self.best_x = x0.copy()
                self.best_f = np.inf
                self.stagnation_counter = 0
                self.no_improve_threshold = 150  # Restart if no improvement for this many iterations

            def _inv_sqrt(self, A):
                # Compute A^{-1/2}
                w, V = np.linalg.eigh(A)
                # Ensure positive definiteness
                w = np.maximum(w, 1e-12)
                return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

            def step(self, func):
                if self.budget <= 0:
                    return False

                # Generate population
                B = self._inv_sqrt(self.C)
                ys = np.zeros((self.lam, self.dim))
                xs = np.zeros((self.lam, self.dim))
                fs = np.zeros(self.lam)

                for i in range(self.lam):
                    if self.budget <= 0:
                        break
                    z = rng.standard_normal(self.dim)
                    y = B @ z
                    x = self.x + self.sigma * y
                    # Clip to box
                    x = np.clip(x, -5, 5)
                    ys[i] = y
                    xs[i] = x
                    fs[i] = func(x)
                    self.budget -= 1

                    if fs[i] < self.best_f:
                        self.best_f = fs[i]
                        self.best_x = x.copy()

                # Sort
                order = np.argsort(fs)
                ys = ys[order]
                xs = xs[order]
                fs = fs[order]

                # Mean update
                y_w = np.dot(self.weights[:self.mu], ys[:self.mu])
                x_w = np.dot(self.weights[:self.mu], xs[:self.mu])

                # Adaptation of step size
                self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * B @ y_w
                hsig = int(np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff / dim) * np.linalg.norm(self.ps) < (1 + self.c_sigma) * self.chi_N)
                self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.ps) / self.chi_N - 1))

                # Adaptation of covariance matrix
                self.pc = (1 - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w
                delta = hsig * (2 - self.c_c) * self.c_c
                self.C = (1 + self.c_1 * delta - self.c_1 - self.c_mu) * self.C
                self.C += self.c_1 * np.outer(self.pc, self.pc)
                self.C += self.c_mu * np.dot(self.weights[:self.mu], np.outer(ys[:self.mu], ys[:self.mu]))

                # Ensure symmetry
                self.C = (self.C + self.C.T) / 2
                # Regularize
                self.C += 1e-12 * np.eye(self.dim)

                # Update mean
                self.x = x_w

                # Update best
                if fs[0] < self.best_f:
                    self.best_f = fs[0]
                    self.best_x = xs[0].copy()

                # Stagnation check
                if fs[0] < self.fitness_history[-1] if self.fitness_history else True:
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                self.fitness_history.append(fs[0])
                if len(self.fitness_history) > 100:
                    self.fitness_history.pop(0)

                # Check for restart
                if self.stagnation_counter > self.no_improve_threshold:
                    return False

                return True

        # Phase 1: Initial CMA-ES run
        # Start from a random point or the origin
        x0 = rng.uniform(-5, 5, dim)
        sigma0 = 2.0
        cma = CMAES(x0, sigma0, budget, dim)

        while cma.budget > 0 and not check_deadline():
            improved = cma.step(func)
            if cma.best_f < best_f:
                best_f = cma.best_f
                best_x = cma.best_x.copy()
            if not improved:
                break

        # Phase 2: If we still have budget, try a second CMA-ES restart from best_x
        if cma.budget > 0 and not check_deadline():
            x0 = best_x.copy()
            sigma0 = 0.5 * np.linalg.norm(best_x) + 0.1
            cma2 = CMAES(x0, sigma0, cma.budget, dim)
            while cma2.budget > 0 and not check_deadline():
                improved = cma2.step(func)
                if cma2.best_f < best_f:
                    best_f = cma2.best_f
                    best_x = cma2.best_x.copy()
                if not improved:
                    break

        # Phase 3: Final Polish with (1+1)-ES
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_count = 0
            total_steps = 0

            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                total_steps += 1

                # Adapt sigma
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    success_count = 0

                sigma = np.clip(sigma, 1e-8, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
