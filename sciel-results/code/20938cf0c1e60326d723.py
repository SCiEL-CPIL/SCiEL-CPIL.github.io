"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()

    def radius(x):
        """Compute the maximum common radius for given centers."""
        x = x.reshape(N, 2)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Distance to boundary
        bounds = 0.5 - np.abs(x)
        min_bound = np.min(bounds)
        return min(min_pair, min_bound)

    def neg_radius(x):
        return -radius(x)

    # Hard-disk expansion: iteratively push apart overlapping circles
    def expand(x, max_iter=2000, tol=1e-10):
        x = x.copy().reshape(N, 2)
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            # Compute pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Find minimum distance pair
            min_dist = np.min(dists)
            r = min_dist / 2.0
            
            # Also check boundary
            bounds = 0.5 - np.abs(x)
            min_bound = np.min(bounds)
            r = min(r, min_bound)
            
            # For each pair closer than 2r, push apart
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2*r - 1e-12 and d > 1e-12:
                        # Push apart
                        direction = (x[i] - x[j]) / d
                        push = (2*r - d) / 2.0
                        x[i] += direction * push
                        x[j] -= direction * push
                        moved = True
                # Boundary check
                for k in range(2):
                    if x[i, k] < -0.5 + r:
                        x[i, k] = -0.5 + r
                        moved = True
                    elif x[i, k] > 0.5 - r:
                        x[i, k] = 0.5 - r
                        moved = True
            
            if not moved:
                break
        
        return x.reshape(-1)

    # SLSQP polish
    def slsqp_polish(x, max_iter=500):
        x = x.copy()
        
        def objective(xv):
            xv = xv.reshape(N, 2)
            diff = xv[:, np.newaxis, :] - xv[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            bounds = 0.5 - np.abs(xv)
            min_bound = np.min(bounds)
            return -min(min_pair, min_bound)
        
        # Constraints: all pairs at least 2r apart, and within boundary
        # We'll use the current radius as a lower bound
        r_current = -objective(x)
        
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def pair_constraint(xv, i=i, j=j):
                    xv = xv.reshape(N, 2)
                    d = np.sqrt(np.sum((xv[i] - xv[j])**2))
                    return d - 2*r_current
                cons.append({'type': 'ineq', 'fun': pair_constraint})
        
        for i in range(N):
            for k in range(2):
                def bound_constraint(xv, i=i, k=k):
                    xv = xv.reshape(N, 2)
                    return xv[i, k] + r_current + 0.5
                def bound_constraint2(xv, i=i, k=k):
                    xv = xv.reshape(N, 2)
                    return 0.5 - r_current - xv[i, k]
                cons.append({'type': 'ineq', 'fun': bound_constraint})
                cons.append({'type': 'ineq', 'fun': bound_constraint2})
        
        try:
            res = minimize(objective, x, method='SLSQP', bounds=[(-0.5, 0.5)]*2*N,
                          constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-12})
            if res.success and -res.fun > r_current - 1e-12:
                return res.x
        except:
            pass
        return x
    
    # Run expansion
    x_expanded = expand(x0)
    r_expanded = radius(x_expanded.reshape(N, 2))
    best_x = x_expanded.copy()
    best_r = r_expanded

    # Try SLSQP polish
    if time.time() < deadline - 20.0:
        x_polished = slsqp_polish(x_expanded, max_iter=300)
        r_polished = radius(x_polished.reshape(N, 2))
        if r_polished > best_r:
            best_x = x_polished
            best_r = r_polished

    # Try more expansions with different initial conditions
    for trial in range(3):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb initial grid
        np.random.seed(trial * 42 + 123)
        perturbation = np.random.randn(N, 2) * 0.01
        x_pert = (x0 + perturbation).flatten()
        
        x_exp = expand(x_pert, max_iter=1500)
        r_exp = radius(x_exp.reshape(N, 2))
        
        if r_exp > best_r:
            best_x = x_exp
            best_r = r_exp
        
        # Polish
        if time.time() < deadline - 15.0:
            x_pol = slsqp_polish(x_exp, max_iter=200)
            r_pol = radius(x_pol.reshape(N, 2))
            if r_pol > best_r:
                best_x = x_pol
                best_r = r_pol

    # Final expansion on best
    if time.time() < deadline - 10.0:
        x_final = expand(best_x, max_iter=1000)
        r_final = radius(x_final.reshape(N, 2))
        if r_final > best_r:
            best_x = x_final
            best_r = r_final

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
