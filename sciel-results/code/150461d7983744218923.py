import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        M = 10
        T = self.budget // M
        
        # Use a fixed seed for reproducibility? No, the task says "hidden seeded shift".
        # The evaluator creates the instance. Our random numbers should be random.
        # However, using a fixed seed in our code is fine, it just makes our search deterministic given the function.
        # Let's use default random.
        
        X = np.random.uniform(-5, 5, (M, N))
        V = np.zeros((M, N))
        
        F = np.array([func(x) for x in X])
        
        Pbest = X.copy()
        Fbest = F.copy()
        
        gbest_idx = np.argmin(Fbest)
        Gbest = Pbest[gbest_idx].copy()
        Fgbest = Fbest[gbest_idx]
        
        best_overall_x = Gbest.copy()
        best_overall_f = Fgbest
        
        c1 = 1.5
        c2 = 1.5
        
        no_improve_count = 0
        max_no_improve = 10
        
        for t in range(T):
            omega = 0.9 - (0.9 - 0.4) * (t / max(1, T - 1))
            
            r1 = np.random.rand(M, N)
            r2 = np.random.rand(M, N)
            
            V = omega * V + c1 * r1 * (Pbest - X) + c2 * r2 * (Gbest[None, :] - X)
            
            X = X + V
            X = np.clip(X, -5, 5)
            
            F = np.array([func(x) for x in X])
            
            improved = F < Fbest
            Pbest[improved] = X[improved]
            Fbest[improved] = F[improved]
            
            current_gbest_idx = np.argmin(Fbest)
            if Fbest[current_gbest_idx] < Fgbest:
                Gbest = Pbest[current_gbest_idx].copy()
                Fgbest = Fbest[current_gbest_idx]
                best_overall_x = Gbest.copy()
                best_overall_f = Fgbest
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= max_no_improve:
                # Randomize worst half
                sorted_idx = np.argsort(Fbest)
                worst_half = sorted_idx[M//2:]
                X[worst_half] = np.random.uniform(-5, 5, (len(worst_half), N))
                V[worst_half] = 0
                # Re-evaluate? No, we will evaluate at the start of next loop? 
                # Actually, we already evaluated X. But X was changed.
                # We should evaluate the new X.
                # But we are in the middle of the loop. 
                # Let's just let the next iteration handle it, or evaluate now?
                # Evaluating now costs budget. We have used M evals this iteration.
                # If we randomize, we should evaluate the new points.
                # But we already used our budget for this iteration.
                # So we just set X and V, and the next iteration will evaluate them.
                # This is fine.
                no_improve_count = 0

        return best_overall_x, best_overall_f
