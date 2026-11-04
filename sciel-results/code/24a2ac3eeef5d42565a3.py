import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14

def construct():
        # Pre-compute indices for all triples
        # i < j < k
        triples = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    triples.append((i, j, k))
        triples = np.array(triples) # shape (364, 3)
        
        def objective(x):
            # x: first N-1 gaps
            # Check validity
            if np.any(x <= 0):
                return 1e10
                
            deltas = np.empty(N)
            deltas[:N-1] = x
            deltas[N-1] = 2 * np.pi - np.sum(x)
            
            if deltas[N-1] <= 0:
                return 1e10
                
            # Cumulative angles
            phi = np.zeros(N)
            phi[1:] = np.cumsum(deltas[:N-1])
            
            # Get angles for triples
            # triples[:, 0], triples[:, 1], triples[:, 2]
            p_i = phi[triples[:, 0]]
            p_j = phi[triples[:, 1]]
            p_k = phi[triples[:, 2]]
            
            a = p_j - p_i
            b = p_k - p_j
            c = 2 * np.pi - a - b
            
            # Area numerator ~ sin(a) + sin(b) + sin(c)
            areas = np.sin(a) + np.sin(b) + np.sin(c)
            
            min_num = np.min(areas)
            
            # Hull area denominator
            den = np.sum(np.sin(deltas))
            
            if den <= 0:
                return 1e10
                
            ratio = min_num / den
            return -ratio # Maximize ratio -> Minimize negative ratio

        # Bounds for DE
        # Each gap must be > 0.
        # To avoid the last gap being negative, we can constrain the sum of first N-1 to be < 2*pi.
        # DE doesn't support linear constraints easily.
        # We can set upper bound for each x_i to 2*pi - (N-2)*epsilon.
        # But sum constraint is tricky.
        # Alternative: Use a transformation?
        # Or just rely on the penalty.
        
        # A safer bound:
        # If one gap is very large, others must be small.
        # Let's use bounds (1e-5, 2*pi - 1e-5) for all.
        # The penalty function handles the sum constraint violation.
        
        bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
        
        # Run Differential Evolution
        # seed=42 for reproducibility
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-10,
            init='sobol' # Better sampling
        )
        
        best_x = de_result.x
        
        # Polish with L-BFGS-B or Nelder-Mead
        # L-BFGS-B is better for smooth parts, but objective is min of smooth functions (non-diff).
        # Powell or Nelder-Mead is safer for non-smooth.
        # Let's use Nelder-Mead for polishing.
        
        polish_result = minimize(
            objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000, 'adaptive': True}
        )
        
        if polish_result.success:
            opt_deltas = np.empty(N)
            opt_deltas[:N-1] = polish_result.x
            opt_deltas[N-1] = 2 * np.pi - np.sum(polish_result.x)
        else:
            opt_deltas = np.empty(N)
            opt_deltas[:N-1] = best_x
            opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
            
        # Ensure valid
        if np.any(opt_deltas <= 0):
             # Fallback to regular
             opt_deltas = np.full(N, 2 * np.pi / N)

        angles = np.zeros(N)
        angles[1:] = np.cumsum(opt_deltas[:N-1])
        
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        return points
