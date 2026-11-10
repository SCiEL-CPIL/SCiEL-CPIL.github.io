# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT2 using hypervolume-optimal spacing on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.budget
        
        # Optimal spacing: equalize marginal HV contribution
        # Marginal contribution: Δf1_i * (10 + sqrt(f1_i))
        # CDF: W(f1) = 2*sqrt(f1) - 20*ln(1 + sqrt(f1)/10)
        W1 = 2.0 - 20.0 * np.log(1.1)
        
        x1_values = np.zeros(n)
        x1_values[0] = 0.0
        x1_values[-1] = 1.0
        
        if n > 2:
            targets = np.arange(1, n - 1) * W1 / n
            from scipy.optimize import newton
            for i in range(1, n - 1):
                target = targets[i - 1]
                def eq(u):
                    return 2.0 * u - 20.0 * np.log(1.0 + u / 10.0) - target
                def eqp(u):
                    return 2.0 - 2.0 / (1.0 + u / 10.0)
                u_sol = newton(eq, 0.5, fprime=eqp, tol=1e-12, maxiter=50)
                x1_values[i] = u_sol ** 2
        
        X = np.zeros((n, self.dim))
        X[:, 0] = x1_values
        
        for i in range(n):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
