# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using a Global Grid Search followed by Newton Polish.
    """
    x0 = np.array(x0, dtype=float)
    
    # Helper: Newton step with central difference Jacobian
    def newton_polish(x_start, max_steps=20):
        x = x_start.copy()
        for _ in range(max_steps):
            try:
                Fx = F(x)
                norm_F = np.linalg.norm(Fx)
                if norm_F <= 1e-8:
                    return x, True
                
                # Central difference Jacobian for better accuracy
                h = 1e-5
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    return x, False
                
                # Damping
                step = 1.0
                improved = False
                for _ in range(10):
                    x_new = x + step * dx
                    try:
                        Fx_new = F(x_new)
                        norm_new = np.linalg.norm(Fx_new)
                    except Exception:
                        step *= 0.5
                        continue
                    
                    if norm_new < norm_F:
                        x = x_new
                        improved = True
                        break
                    else:
                        step *= 0.5
                
                if not improved:
                    return x, False
            except Exception:
                return x, False
        
        # Final check
        try:
            return x, np.linalg.norm(F(x)) <= 1e-8
        except Exception:
            return x, False

    # Helper: Evaluate F and return norm, handling exceptions
    def eval_F(x):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return np.linalg.norm(Fx)
        except Exception:
            return np.inf

    # 1. Generate candidate points
    # We will try grids of increasing radius to handle cases where x0 is far from root
    radii = [0.1, 1.0, 10.0, 100.0, 1000.0, 1e6]
    candidates = []
    
    # Always include x0
    candidates.append(x0)
    
    for r in radii:
        # Create a grid in a square of side 2r centered at x0
        # 5x5 grid = 25 points per scale
        steps = np.linspace(-r, r, 5)
        for dx in steps:
            for dy in steps:
                pt = x0 + np.array([dx, dy])
                candidates.append(pt)
        
        # Also add points on the circle for better coverage?
        # 4 cardinal points
        candidates.append(x0 + np.array([r, 0]))
        candidates.append(x0 + np.array([-r, 0]))
        candidates.append(x0 + np.array([0, r]))
        candidates.append(x0 + np.array([0, -r]))
        
        # Add diagonal points
        r_diag = r * np.sqrt(2)
        candidates.append(x0 + np.array([r_diag, r_diag]))
        candidates.append(x0 + np.array([-r_diag, r_diag]))
        candidates.append(x0 + np.array([r_diag, -r_diag]))
        candidates.append(x0 + np.array([-r_diag, -r_diag]))

    # 2. Score candidates by residual
    # We want to try the best ones first.
    # To save calls, we evaluate them in order of likelihood (smaller radius first).
    # But we need to sort by residual to pick the best for Newton.
    # Since the number of candidates is small (~150), we can evaluate all and sort.
    
    scored_candidates = []
    for pt in candidates:
        norm = eval_F(pt)
        scored_candidates.append((norm, pt))
        
        # Early exit if we find a very good candidate
        # If norm is small enough, Newton will likely converge.
        # However, we might find an even better one later? Unlikely if grid is dense.
        # Let's not exit early to ensure we find the global best among the grid.
        
    # Sort by norm
    scored_candidates.sort(key=lambda x: x[0])
    
    # 3. Polish the best candidates with Newton
    # Try the top 5 candidates
    for norm, pt in scored_candidates[:5]:
        if norm > 1e2: # If residual is huge, Newton might not help, but try anyway?
            # If the best candidate has huge residual, we might be in the wrong basin.
            # But we tried larger radii. If all are huge, we are stuck.
            pass
            
        x_sol, success = newton_polish(pt)
        
        if success:
            return x_sol
            
    # 4. Fallback: Return the point with the smallest residual found
    if scored_candidates:
        return scored_candidates[0][1]
    
    return x0
# EVOLVE-BLOCK-END
