"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def generate_points(params):
        """
        Generates 16 points in a 4x4 grid structure with modulations.
        params: [shear_x, mod_amp_x, mod_amp_y, shear_y]
        
        Base grid:
        x_base = col_index - 1.5
        y_base = row_index - 1.5
        
        Modulations:
        x = x_base + shear_x * y_base + mod_amp_x * sin(2*pi * row_index / 4)
        y = y_base + shear_y * x_base + mod_amp_y * cos(2*pi * col_index / 4)
        
        This creates a sheared, wavy grid.
        """
        shear_x, mod_amp_x, mod_amp_y, shear_y = params
        
        rows = np.arange(4)
        cols = np.arange(4)
        
        # Create meshgrid of base coordinates
        # We want 4 rows and 4 cols
        # y varies by row, x varies by col
        y_base = rows[:, np.newaxis] - 1.5  # shape (4, 1)
        x_base = cols[np.newaxis, :] - 1.5  # shape (1, 4)
        
        # Expand to (4, 4)
        y_base = np.broadcast_to(y_base, (4, 4))
        x_base = np.broadcast_to(x_base, (4, 4))
        
        # Apply shear
        x_sheared = x_base + shear_x * y_base
        y_sheared = y_base + shear_y * x_base
        
        # Apply sinusoidal modulation
        # Modulate x based on row index (horizontal waves)
        x_mod = x_sheared + mod_amp_x * np.sin(2 * np.pi * y_base / 4.0)
        
        # Modulate y based on col index (vertical waves)
        y_mod = y_sheared + mod_amp_y * np.cos(2 * np.pi * x_base / 4.0)
        
        points = np.column_stack([x_mod.ravel(), y_mod.ravel()])
        
        # Normalize: Center and scale such that min distance is approx 1?
        # Actually, for the ratio score, normalization doesn't matter.
        # But for numerical stability in DE, keeping coords bounded is good.
        # Let's just return the raw points. The score function handles the ratio.
        
        return points

    def objective(params):
        pts = generate_points(params)
        # Check for distinct points
        if np.any(np.isclose(get_sq_dists(pts), 0)):
            return 1e10
        return score_raw(pts)

    # Bounds for the 4 parameters
    # Shear: -1 to 1
    # Mod Amp: 0 to 1
    bounds = [
        (-1.0, 1.0),  # shear_x
        (0.0, 1.0),   # mod_amp_x
        (0.0, 1.0),   # mod_amp_y
        (-1.0, 1.0)   # shear_y
    ]
    
    best_params = None
    best_score = 1e10
    
    # Use Differential Evolution
    # popsize=20, maxiter=100 is a reasonable start.
    # We can run multiple restarts if time permits.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            popsize=20,
            maxiter=200,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True, # Use L-BFGS-B to polish the result
            constraints=()
        )
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception as e:
        pass

    # If time remains, try another restart with different seed and bounds
    if time.time() < deadline - 20:
        try:
            # Wider bounds for shear
            bounds2 = [
                (-2.0, 2.0),
                (0.0, 2.0),
                (0.0, 2.0),
                (-2.0, 2.0)
            ]
            res2 = differential_evolution(
                objective, 
                bounds2, 
                seed=123,
                popsize=15,
                maxiter=100,
                tol=1e-8,
                polish=True
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple grid
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        pts = generate_points(best_params)
        
    # Final check and return
    return pts

# EVOLVE-BLOCK-END
