import numpy as np
from scipy.optimize import differential_evolution

def construct():
    import time
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    def objective(x):
        P = x.reshape(8, 2)
        # Vectorized distance calculation
        # P[:, np.newaxis, :] - P[np.newaxis, :, :] gives 8x8x2
        # This creates an 8x8x2 array.
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # dists_sq is 8x8. Diagonal is 0.
        # We need upper triangle
        i, j = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i, j])
        
        if np.any(dists < 1e-9):
            return 1e10 # Minimize, so large value is bad
            
        max_d = np.max(dists)
        if max_d < 1e-9:
            return 1e10
            
        log_prod = np.sum(np.log(dists))
        val = log_prod - 28 * np.log(max_d)
        return -val # Minimize negative
        
    # Initial bounds
    # We can bound coordinates to [-2, 2] to help convergence?
    # The optimal diameter is 2. So points are within radius 1 or so.
    # But we don't know the center.
    # Let's use [-5, 5] to be safe.
    bounds = [(-5, 5)] * 16
    
    # Run differential evolution
    try:
        res = differential_evolution(objective, bounds, maxiter=2000, popsize=20, tol=1e-8, seed=42, polish=True)
        best_x = res.x
        best_score_val = -res.fun # This is the log-score part
        # Convert to actual score
        # Score = exp(best_score_val) / 8**8 ?
        # Wait, the objective was log_prod - 28 log(max_d).
        # The actual score is (prod dists) / (max_d ** 28) / 8**8 ?
        # No, the problem says: "score is Delta/n^n, where Delta is the product ... after scaling to diameter 2".
        # If we scale to diameter 2, the new distances are $d'_{ij} = 2 d_{ij} / D$.
        # $\Delta = \prod (2 d_{ij} / D) = 2^{28} \prod d_{ij} / D^{28}$.
        # Score $= \Delta / 8^8 = 2^{28} / 8^8 \cdot (\prod d_{ij} / D^{28})$.
        # $2^{28} / 8^8 = 2^{28} / (2^3)^8 = 2^{28} / 2^{24} = 2^4 = 16$.
        # So Score $= 16 \cdot \exp(\log_prod - 28 \log D)$.
        # My objective minimized $-(\log_prod - 28 \log D)$.
        # So $val = \log_prod - 28 \log D$.
        # Score $= 16 \cdot \exp(val)$.
        
        # Check time
        if time.time() - start_time > time_limit:
             # Return regular octagon if timeout?
             # But we should return best found so far.
             pass

        # We need to return the points scaled to diameter 2?
        # The interface says: "construct() -> 8 distinct points [x, y] ... score is ... after scaling".
        # So we can return ANY points. The evaluator scales them.
        # So we just return best_x reshaped.
        
        # However, we should verify the score to be sure.
        P = best_x.reshape(8, 2)
        # Calculate score manually
        dists = []
        max_d = 0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(P[i] - P[j])
                dists.append(d)
                if d > max_d: max_d = d
        prod = np.prod(dists)
        score = (prod / (max_d ** 28)) * 16.0
        
        # If score is not better than 1.0, maybe DE didn't converge well or got stuck.
        # We can try a few random restarts if time permits.
        
    except Exception as e:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    # Convert to list of lists
    return [[float(P[i][0]), float(P[i][1])] for i in range(8)]
