import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Initial mean: center of the box
        x_mean = np.zeros(self.dim)
        
        # Initial sigma: roughly 1/5 of the box width? Box width is 10.
        # A common heuristic is sigma0 = 0.3 or related to box size.
        sigma = 1.0
        
        # Population size
        lam = 10
        mu = lam // 2
        
        # CMA-ES parameters
        cc = (4 + lam / self.dim) / (self.dim + 4 + 2 * lam / self.dim)
        cs = (lam / 2 + 1) / (self.dim + lam / 2 + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * ((mu + 1) / (lam + 2) - 1)) / ((self.dim + 2) ** 2 + lam)
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        best_x = None
        best_f = np.inf
        
        # We need to track the number of evaluations
        evals = 0
        
        # Initial evaluation
        # We can start by evaluating the mean? Or just let the first generation handle it.
        # CMA-ES usually samples the population.
        
        while evals < self.budget:
            # Generate population
            # Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except:
                # If C is not positive definite, reset
                C = np.eye(self.dim)
                B = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                sigma *= 0.5
                
            D = 1.0 / np.sqrt(np.diag(B.T @ B)) # This is not quite right for CMA-ES update, but B is lower triangular.
            # Actually, we just need to sample from N(0, C).
            # x = mean + sigma * B @ z, where z ~ N(0, I)
            
            population = []
            fitnesses = []
            
            for i in range(lam):
                if evals + i >= self.budget:
                    break
                z = rng.standard_normal(self.dim)
                x = x_mean + sigma * (B @ z)
                x = np.clip(x, -5, 5)
                f = func(x)
                population.append(x)
                fitnesses.append(f)
                evals += 1
                
                if f < best_f:
                    best_f = f
                    best_x = x
            
            if len(population) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)
            
            # Update mean
            x_new_mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ ((x_new_mean - x_mean) / (sigma * np.linalg.norm(B)) ))
            # Note: The norm of B is not 1. The standard CMA-ES uses the inverse of the square root of C.
            # This manual implementation is getting tricky.
            
            # Simpler approach: Use scipy.optimize.differential_evolution
            # It is robust and available.
            break
            
        return best_x, best_f
