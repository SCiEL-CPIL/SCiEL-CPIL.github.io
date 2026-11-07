"""BBOB f14 (Different powers) optimizer.

Uses a Best-First Multi-Start CMA-ES (Covariance Matrix Adaptation Evolution Strategy) ensemble.
CMA-ES is superior to Nelder-Mead for ill-conditioned, non-separable landscapes like BBOB f14.
The ensemble allows for global exploration while CMA-ES handles local exploitation efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class CmaES:
    """
    A lightweight, efficient implementation of CMA-ES for small dimensions.
    """
    def __init__(self, x0, sigma, func, dim, budget_remaining, rng, bounds=(-5, 5)):
        self.dim = dim
        self.n = dim
        self.lambda_ = int(4 + 3 * np.log(self.n))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_w = np.sum(self.weights * self.weights)
        
        # Strategy parameters
        self.c_sigma = 1.0 / (self.n + np.sqrt(2) - 2)
        self.c1 = 2.0 / ((self.n + 1.3)**2 + self.lambda_)
        self.c_mu = min(1.0 - self.c1, 2.0 * (self.mu_w - 2 + 1.0/self.mu_w) / ((self.n + 2)**2 + self.lambda_))
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_w - 2 + 1.0/self.mu_w) / (self.lambda_ + 2) - 1)) + self.c_sigma
        self.chi_n = np.sqrt(self.n) * (1 - 1.0/(4*self.n) + 1.0/(21*self.n**2))
        
        self.x0 = np.array(x0, dtype=float)
        self.mean = self.x0.copy()
        self.sigma = sigma
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.func = func
        self.budget_remaining = budget_remaining
        self.rng = rng
        self.bounds = bounds
        self.best_x = self.x0.copy()
        self.best_f = np.inf
        self.generation = 0
        self.last_improve_gen = 0
        self.calls_used = 0
        
        # Initial evaluation
        self._evaluate_initial()

    def _evaluate_initial(self):
        # Evaluate initial mean and some neighbors to get a baseline
        # We will just rely on the first population generation
        pass

    def ask(self):
        """Generate next population."""
        if self.calls_used >= self.budget_remaining:
            return None
            
        # Eigen decomposition of C for sampling
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(D)
        
        pop = []
        for i in range(self.lambda_):
            z = self.rng.standard_normal(self.n)
            x = self.mean + self.sigma * (B @ (D * z))
            x = np.clip(x, self.bounds[0], self.bounds[1])
            pop.append(x)
            
        return pop

    def tell(self, pop, fitnesses):
        """Update internal state based on evaluated population."""
        if pop is None:
            return
            
        # Sort by fitness
        indices = np.argsort(fitnesses)
        pop_sorted = [pop[i] for i in indices]
        fit_sorted = [fitnesses[i] for i in indices]
        
        # Update best
        if fit_sorted[0] < self.best_f:
            self.best_f = fit_sorted[0]
            self.best_x = pop_sorted[0].copy()
            self.last_improve_gen = self.generation
            
        # Selection
        x_w = np.zeros((self.mu, self.n))
        for i in range(self.mu):
            x_w[i] = pop_sorted[i]
            
        # Weighted recombination
        m_new = np.sum(self.weights[:, np.newaxis] * x_w, axis=0)
        
        # Update step size
        y = (m_new - self.mean) / self.sigma
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_w) * y
        h_sig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * (self.generation + 1)))) / self.chi_n < (1.4 + 2 / (self.n + 1))
        
        if h_sig:
            self.pc = (1 - self.c1) * self.pc + np.sqrt(self.c1 * (2 - self.c1) * self.mu_w) * y
            
        self.mean = m_new
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        
        # Update Covariance
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.sum(self.weights[:, np.newaxis, np.newaxis] * np.outer(x_w, x_w), axis=0)
        self.C = (1 + self.c1 * (2 - self.c1) - self.c_mu) * self.C + self.c1 * rank_one + self.c_mu * rank_mu
        
        self.generation += 1
        self.calls_used += len(pop)

    def is_stagnated(self, threshold=20):
        return (self.generation - self.last_improve_gen) > threshold


class Optimizer:
    """Best-First Multi-Start CMA-ES for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            if not ok():
                return None
            calls += 1
            try:
                return func(np.asarray(x, dtype=float))
            except Exception:
                return np.inf

        # Number of concurrent CMA-ES agents
        n_agents = 3
        
        # Latin Hypercube Sampling for diverse initial points
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_agents + 1)
            for i in range(n_agents):
                lhs_points.append(rng.uniform(strata[i], strata[i+1]))
        
        lhs_points = np.array(lhs_points).reshape(n_agents, dim)

        # Initialize agents
        agents = []
        for i in range(n_agents):
            x0 = lhs_points[i]
            # Initial step size: cover a reasonable portion of the space
            sigma0 = 1.5 
            agent = CmaES(x0, sigma0, func, dim, budget, rng)
            agents.append(agent)
            
            # Initial evaluation is done in CmaES constructor? 
            # No, we need to explicitly evaluate the first population to get feedback.
            # Let's do the first generation for all agents to seed them.
            if ok():
                pop = agent.ask()
                if pop:
                    fits = []
                    for x in pop:
                        f = call(x)
                        if f is None:
                            f = np.inf
                        fits.append(f)
                        record(x, f)
                    agent.tell(pop, fits)
        
        # Main loop: Best-First selection
        global_iter = 0
        
        while ok():
            global_iter += 1
            
            # Select the agent with the best current minimum
            # We prioritize agents that are close to the global best or have made recent progress
            # Simple strategy: pick the agent with the lowest best_f
            best_agent_idx = np.argmin([a.best_f for a in agents])
            agent = agents[best_agent_idx]
            
            # Check for stagnation
            if agent.is_stagnated(threshold=30):
                # Restart this agent
                # Generate a new random point, possibly near the global best
                if rng.random() < 0.7 and best_x is not None:
                    x_new = best_x + rng.normal(0, 0.5, dim)
                    sigma_new = 0.5
                else:
                    x_new = rng.uniform(-5, 5, dim)
                    sigma_new = 1.5
                
                x_new = np.clip(x_new, -5, 5)
                
                # Re-initialize the agent
                agents[best_agent_idx] = CmaES(x_new, sigma_new, func, dim, budget - calls, rng)
                new_agent = agents[best_agent_idx]
                
                # Evaluate first population of new agent
                if ok():
                    pop = new_agent.ask()
                    if pop:
                        fits = []
                        for x in pop:
                            f = call(x)
                            if f is None:
                                f = np.inf
                            fits.append(f)
                            record(x, f)
                        new_agent.tell(pop, fits)
                continue
            
            # Ask for next population
            pop = agent.ask()
            if pop is None:
                # Budget exhausted for this agent? Should not happen if we check ok()
                continue
                
            # Evaluate population
            fits = []
            for x in pop:
                if not ok():
                    # If we run out of budget in the middle of a population, 
                    # we must still update the agent with what we have to maintain consistency?
                    # Or just break. Since we are done, it doesn't matter much.
                    # But we should record the calls made.
                    break
                f = call(x)
                if f is None:
                    f = np.inf
                fits.append(f)
                record(x, f)
            
            if len(fits) == len(pop):
                agent.tell(pop, fits)
            elif len(fits) > 0:
                # Partial evaluation
                # This complicates CMA-ES state. 
                # For simplicity and robustness, if we can't finish a population, 
                # we might just restart this agent or skip it.
                # Given the budget is 500 and pop size is ~10, we rarely hit this exactly.
                # If we do, we just continue the loop, and the agent will be stale.
                # Better: if we run out of budget, the loop condition `ok()` will fail next iteration.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
