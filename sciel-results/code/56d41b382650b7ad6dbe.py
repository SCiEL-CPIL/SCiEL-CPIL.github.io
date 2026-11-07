import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = None
        self.start_time = time.time()
        
        # Safety margin: 20 seconds
        deadline = self.start_time + 280.0
        
        # CMA-ES Parameters
        # For dim=5, default lambda is 4 + floor(3 * ln(5)) = 4 + 4 = 8.
        # We will use lambda=10 for a bit more diversity, or stick to standard.
        # Standard CMA-ES is very efficient.
        # Initial sigma: The box is [-5, 5]. The minimum is in [-4, 4].
        # Starting sigma=1.0 is reasonable.
        
        # We implement a simplified CMA-ES loop to have full control over budget and time.
        # Using scipy.cmaes is an option, but a manual loop allows tighter integration 
        # with the strict call counting and time checks.
        
        # However, writing a full CMA-ES from scratch is error-prone.
        # Let's use scipy.optimize.minimize with method='CMA' if available? 
        # No, scipy.optimize.minimize doesn't have CMA built-in in standard versions.
        # We need to implement the core CMA-ES logic or use a simpler robust method.
        
        # Given the constraints and the success of inspiration 2 (score 0.89) which used CMA-ES,
        # and the failure of the current Powell approach, I will implement a robust 
        # (1+lambda)-ES with adaptive step size and potentially a simple covariance 
        # update, or just a very well-tuned (1+lambda)-ES which is often sufficient 
        # for low-dimensional problems like dim=5 Rosenbrock.
        
        # Actually, for dim=5, a well-tuned (1+lambda)-ES with adaptive sigma 
        # and a diverse population (lambda=10) is extremely competitive and robust.
        # Let's call it "Adaptive Population ES".
        
        rng = np.random.default_rng(42)
        
        # Initial point: Center of the box is a good bet for Rosenbrock if shifted, 
        # but the shift is hidden. Let's start at 0.
        x = np.zeros(self.dim)
        fx = self._safe_eval(func, x)
        
        sigma = 1.0
        lam = 10  # Population size
        # CMA-ES like parameters for step size adaptation
        # We will use a simple success/failure based adaptation for sigma
        # and a simple covariance matrix update if we want to be fancy, 
        # but for dim=5, a diagonal covariance (or identity) with adaptive sigma 
        # is often enough if the landscape isn't too pathological.
        # Rosenbrock is valley-shaped. A fixed identity might struggle.
        # Let's implement a basic CMA-ES logic.
        
        # CMA-ES Implementation
        # Reference: Hansen's CMA-ES
        # Parameters:
        # mu = lam // 2 (number of parents)
        # weights = log(mu + 0.5) - log(1..mu) normalized
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_amps = 1 + 2 * max(0, np.sqrt(mu_eff / self.dim) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        
        # Covariance matrix update parameters
        c_covpos = (2 + 2 * mu_eff / self.dim) / (self.dim + 5 + 2 * mu_eff / self.dim)
        c_covneg = (2 + 2 * (mu_eff - 1) / self.dim) / (self.dim + 5 + 2 * (mu_eff - 1) / self.dim)
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Initial sigma
        sigma = 1.0
        
        # Main Loop
        while self.call_count < self.budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                # Check time and budget
                if time.time() > deadline or self.call_count >= self.budget:
                    break
                    
                z = rng.standard_normal(self.dim)
                # Mahalanobis distance calculation is expensive, 
                # but for dim=5 it's fine.
                # y = sigma * z (simplified, ignoring C for speed in first pass? 
                # No, C is important for Rosenbrock valley alignment)
                # We need to sample from N(0, C).
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(self.dim)
                    L = np.eye(self.dim)
                    
                y = sigma * (L @ z)
                x_new = x + y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = self._safe_eval(func, x_new)
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
            
            if len(fs) == 0:
                break
                
            # Sort by fitness
            sorted_indices = np.argsort(fs)
            xs_sorted = [xs[i] for i in sorted_indices]
            ys_sorted = [ys[i] for i in sorted_indices]
            fs_sorted = [fs[i] for i in sorted_indices]
            
            # Update best
            if fs_sorted[0] < self.best_f:
                self.best_f = fs_sorted[0]
                self.best_x = xs_sorted[0].copy()
                
            # Update mean
            x_old = x
            x = np.sum([weights[i] * xs_sorted[i] for i in range(mu)], axis=0)
            
            # Update step size control
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^(-1/2) * (x - x_old))
            # We need C^(-1/2). Since C = B B^T, C^(-1/2) = B^(-T) B^(-1)
            # This is getting complex to implement robustly from scratch.
            
            # Alternative Strategy: 
            # Given the risk of implementing CMA-ES incorrectly and the high score of 
            # a "pure high-performance CMA-ES" in inspiration 2, maybe I should use 
            # a library if available? No, only standard/NumPy/SciPy.
            # SciPy does NOT have a built-in CMA-ES in `scipy.optimize.minimize`.
            #
            # Let's pivot to a more robust, simpler method that is known to work 
            # well for low-dim black-box: **Nelder-Mead** with multiple restarts 
            # and a **global exploration** phase.
            #
            # Wait, Inspiration 3 tried Nelder-Mead and got 0.059. 
            # Inspiration 2 (CMA-ES) got 0.89. 
            # The current program (Powell + 1+1 ES) got 0.116.
            #
            # The key difference in Inspiration 2 was "strict evaluation budgeting" 
            # and "pure CMA-ES".
            #
            # I will implement a simplified CMA-ES that avoids the complex 
            # C^(-1/2) calculation by using a diagonal covariance matrix adaptation 
            # (like SA-ES or a simplified CMA). For Rosenbrock, aligning the 
            # step sizes with the axes is less critical than in high dimensions, 
            # but the valley structure makes it tricky.
            #
            # Actually, let's look at the "Inspiration 1" (score 1.0). 
            # It's the "Initial program for the suite task". 
            # This suggests the baseline was very good. 
            # The current program is a regression from that baseline.
            #
            # What was the baseline? Usually, for these tasks, a simple 
            # **Differential Evolution** or **Particle Swarm Optimization** 
            # or a well-tuned **CMA-ES** is the standard.
            #
            # Let's try **Differential Evolution** (DE). It is robust, 
            # derivative-free, and works well on low-dimensional problems.
            # It maintains a population and uses vector differences to generate 
            # new candidates.
            
            # Break out of the CMA-ES draft and implement DE.
            break

        # Implementing Differential Evolution (DE)
        # Parameters:
        # NP = 10 (Population size)
        # F = 0.5 (Scale factor)
        # CR = 0.5 (Crossover rate)
        
        NP = 10
        F = 0.5
        CR = 0.5
        
        # Initialize population
        pop = []
        for i in range(NP):
            if time.time() > deadline or self.call_count >= self.budget:
                break
            x_i = rng.uniform(-5, 5, self.dim)
            f_i = self._safe_eval(func, x_i)
            pop.append([x_i, f_i])
            
        # Sort initial population
        pop.sort(key=lambda item: item[1])
        
        # Update best
        if pop[0][1] < self.best_f:
            self.best_f = pop[0][1]
            self.best_x = pop[0][0].copy()
            
        # Main DE Loop
        while self.call_count < self.budget and time.time() < deadline:
            # If we have used up budget for a full generation, break
            if self.call_count + NP > self.budget:
                # Use remaining budget for local refinement of best point
                break
                
            # Generate new population
            new_pop = []
            for i in range(NP):
                if time.time() > deadline or self.call_count >= self.budget:
                    break
                    
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                    
                x_r1 = pop[r1][0]
                x_r2 = pop[r2][0]
                x_r3 = pop[r3][0]
                
                # Differential variation
                v = x_r1 + F * (x_r2 - x_r3)
                
                # Binomial crossover
                x_trial = x_i = pop[i][0]
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        x_trial[j] = v[j]
                    else:
                        x_trial[j] = x_i[j]
                        
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = self._safe_eval(func, x_trial)
                
                # Selection
                if f_trial <= pop[i][1]:
                    new_pop.append([x_trial, f_trial])
                else:
                    new_pop.append(pop[i])
                    
            # Update population
            pop = new_pop
            pop.sort(key=lambda item: item[1])
            
            # Update best
            if pop[0][1] < self.best_f:
                self.best_f = pop[0][1]
                self.best_x = pop[0][0].copy()
                
        # If we have budget left, do a local refinement with (1+1)-ES on the best point
        if self.call_count < self.budget - 5 and self.best_x is not None:
            x = self.best_x.copy()
            fx = self.best_f
            sigma = 0.1
            # Use remaining budget
            while self.call_count < self.budget - 1 and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = self._safe_eval(func, y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 5.0)
                
            if fx < self.best_f:
                self.best_x = x
                self.best_f = fx

        return self.best_x, self.best_f

    def _safe_eval(self, func, x):
        if self.call_count >= self.budget:
            raise ValueError("Budget Exceeded")
        
        x = np.clip(x, -5, 5)
        f = func(x)
        self.call_count += 1
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f
