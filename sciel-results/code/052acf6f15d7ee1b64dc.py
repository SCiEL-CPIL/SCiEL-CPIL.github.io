import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import itertools

N = 14
D = 3

def _objective(x):
    pts = x.reshape(N, D)
    d = pdist(pts)
    d_min = np.min(d)
    if d_min < 1e-8:
        return 1e12
    d_max = np.max(d)
    return (d_max / d_min) ** 2

def construct():
    start_time = time.time()
    timeout = 110.0  # 10s margin
    
    # Initial guess: FCC-like structure
    # 8 points at cube vertices (±1, ±1, ±1)
    # 6 points at octahedron vertices (±1, 0, 0), etc.
    # This is the standard FCC unit cell content (if we consider the central atom as one of the octahedron points? No, FCC is corners + face centers).
    # My previous "Cube + Octahedron" is:
    # Cube: (±1, ±1, ±1)
    # Oct: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    # Let's check distances for this specific set.
    # d(Cube-Cube) min: 2 (edge)
    # d(Oct-Oct) min: sqrt(2) (edge of octa)
    # d(Cube-Oct) min: sqrt((1-1)^2 + 1^2 + 1^2) = sqrt(2)
    # So min distance is sqrt(2).
    # Max distance:
    # d(Cube-Cube) max: 2*sqrt(3) (body diag)
    # d(Oct-Oct) max: 2
    # d(Cube-Oct) max: sqrt((1-(-1))^2 + 1^2 + 1^2) = sqrt(4+1+1) = sqrt(6)
    # So max distance is 2*sqrt(3) approx 3.46.
    # Ratio: 3.46 / 1.414 = 2.45.
    # Squared: 6.
    
    # Can we do better?
    # If we scale the cube down relative to the octahedron?
    # Let Cube be (±c, ±c, ±c) and Oct be (±1, 0, 0)...
    # This is what the optimizer will do.
    
    cube = np.array(list(itertools.product([-1, 1], repeat=3)))
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    init_fcc = np.vstack([cube, octa])
    
    # Add small noise
    np.random.seed(42)
    init_noisy = init_fcc + np.random.randn(N, D) * 0.1
    
    best_x = init_noisy.flatten()
    best_score = _objective(best_x)
    
    bounds = [(-5, 5)] * (N * D)
    
    # Run Differential Evolution
    if time.time() - start_time < timeout - 40:
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=300, 
                popsize=20, 
                tol=1e-5,
                polish=False # Do polish separately for time control
            )
            if res.fun < best_score:
                best_x = res.x
                best_score = res.fun
        except Exception:
            pass
            
    # Final Polish
    if time.time() - start_time < timeout - 5:
        try:
            res2 = minimize(
                _objective, 
                best_x, 
                method='Powell', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'xtol': 1e-15}
            )
            if res2.fun < best_score:
                best_x = res2.x
                best_score = res2.fun
        except Exception:
            pass
            
    return best_x.reshape(N, D)
