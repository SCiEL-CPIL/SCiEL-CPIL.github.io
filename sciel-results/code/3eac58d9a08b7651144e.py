import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    n = 8
    # Objective: Minimize -sum(log(distances))
    # Constraints: distances <= 1
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_dists = dists[mask]
        # If any distance is 0, return large penalty
        if np.any(valid_dists < 1e-12):
            return 1e10
        return -np.sum(np.log(valid_dists))
    
    def constraint_diam(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_dists = dists[mask]
        # We want valid_dists <= 1
        # SLSQP expects g(x) >= 0
        return 1 - np.max(valid_dists)
        
    # We need to ensure all pairs satisfy the constraint, not just the max.
    # SLSQP handles scalar constraints easily, but vector constraints are better.
    # Let's use a vector of constraints for all pairs.
    
    def constraints_vec(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_dists = dists[mask]
        return 1 - valid_dists

    # To avoid singularities and help the optimizer, we can also add a small penalty for points being too close
    # But the log objective handles that naturally.

    best_score = -np.inf
    best_pts = None

    def get_dists(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return dists[mask]

    def de_objective(x):
        valid_dists = get_dists(x)
        if np.any(valid_dists < 1e-12):
            return 1e10
        penalty = np.sum(np.maximum(0, valid_dists - 1.0)**2)
        return -np.sum(np.log(valid_dists)) + 1000.0 * penalty

    # 1. Global optimization with Differential Evolution
    bounds = [(-1, 1)] * (n * 2)
    try:
        de_res = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=500, 
            popsize=20,
            tol=1e-8,
            polish=False
        )
        if de_res.fun < 1e9:
            try:
                res = minimize(
                    objective, 
                    de_res.x, 
                    method='SLSQP', 
                    constraints=[{'type': 'ineq', 'fun': constraints_vec}],
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
                if res.success:
                    c = constraints_vec(res.x)
                    if np.all(c >= -1e-9):
                        val = -res.fun
                        if val > best_score:
                            best_score = val
                            best_pts = res.x.reshape(n, 2)
            except Exception:
                pass
    except Exception:
        pass

    # 2. Local optimization from Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    octagon = np.column_stack((np.cos(angles), np.sin(angles)))
    octagon_scaled = octagon * 0.5
    x0_oct = octagon_scaled.flatten()
    
    try:
        res_oct = minimize(
            objective, 
            x0_oct, 
            method='SLSQP', 
            constraints=[{'type': 'ineq', 'fun': constraints_vec}],
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        if res_oct.success:
            c = constraints_vec(res_oct.x)
            if np.all(c >= -1e-9):
                val = -res_oct.fun
                if val > best_score:
                    best_score = val
                    best_pts = res_oct.x.reshape(n, 2)
    except Exception:
        pass

    # 3. Local optimization from Random Starts
    np.random.seed(42)
    for i in range(5):
        r = np.random.rand(n) ** 0.5
        theta = np.random.rand(n) * 2 * np.pi
        x0 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        x0_flat = x0.flatten()
        
        try:
            res = minimize(
                objective, 
                x0_flat, 
                method='SLSQP', 
                constraints=[{'type': 'ineq', 'fun': constraints_vec}],
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success:
                c = constraints_vec(res.x)
                if np.all(c >= -1e-9):
                    val = -res.fun
                    if val > best_score:
                        best_score = val
                        best_pts = res.x.reshape(n, 2)
        except Exception:
            continue
            
    # Also try the octagon result explicitly in case random fails or is worse
    # We already have score_oct. But we need to compare it in the same "space".
    # The objective value is the log product.
    # Let's just compute the score for the best_pts found.
    
    if best_pts is None:
        # Fallback to scaled octagon
        best_pts = octagon_scaled
    
    # Now we have best_pts with diameter <= 1.
    # The problem states: "scaling the set to diameter 2".
    # So we need to scale best_pts such that its max pairwise distance is 2.
    
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dists[mask]
    current_diam = np.max(valid_dists)
    
    if current_diam > 1e-12:
        scale_factor = 2.0 / current_diam
        final_pts = best_pts * scale_factor
    else:
        # Degenerate case, fallback to regular octagon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        final_pts = np.column_stack((np.cos(angles), np.sin(angles)))

    return final_pts.tolist()
