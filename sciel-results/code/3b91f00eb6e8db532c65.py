import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
        # Objective function
        def objective(x):
            # x: array of first N-1 gaps
            if np.any(x <= 1e-6):
                return 1e10
            deltas = np.empty(N)
            deltas[:N-1] = x
            deltas[N-1] = 2 * np.pi - np.sum(x)
            if deltas[N-1] <= 1e-6:
                return 1e10
            
            phi = np.zeros(N)
            phi[1:] = np.cumsum(deltas[:N-1])
            
            min_num = float('inf')
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        a = phi[j] - phi[i]
                        b = phi[k] - phi[j]
                        c = 2 * np.pi - a - b
                        val = np.sin(a) + np.sin(b) + np.sin(c)
                        if val < min_num:
                            min_num = val
                        
            den = np.sum(np.sin(deltas))
            if den <= 0:
                return 1e10
            return -(min_num / den)

        bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
        
        best_x = None
        best_score = -float('inf')
        
        rng = np.random.default_rng(42)
        
        # Initial guesses
        inits = []
        # 1. Uniform
        inits.append(np.full(N-1, 2 * np.pi / N))
        
        # 2. Random Dirichlet samples
        for _ in range(20):
            alpha = rng.random(N) + 0.5
            gaps = alpha / np.sum(alpha) * 2 * np.pi
            inits.append(gaps[:N-1])
            
        for init in inits:
            # Clip init to bounds to help optimizer start
            init_clipped = np.clip(init, 1e-4, 2*np.pi - 1e-4)
            # Adjust last gap implicitly? No, Nelder-Mead will handle.
            # But if sum(init_clipped) > 2pi, the last gap is negative.
            # We should ensure the initial point is feasible.
            # Simple check: if sum(x) < 2pi - 1e-5
            if np.sum(init_clipped) > 2 * np.pi - 1e-5:
                # Scale down
                scale = (2 * np.pi - 1e-5) / np.sum(init_clipped)
                init_clipped = init_clipped * scale
                # Re-clip?
                
            result = minimize(
                objective, 
                init_clipped, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={
                    'xatol': 1e-12, 
                    'fatol': 1e-12, 
                    'maxiter': 5000, 
                    'adaptive': True
                }
            )
            
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_x = result.x
        
        if best_x is None:
            best_x = np.full(N-1, 2 * np.pi / N)
            
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_x
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
        
        angles = np.zeros(N)
        angles[1:] = np.cumsum(opt_deltas[:N-1])
        
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        return points
