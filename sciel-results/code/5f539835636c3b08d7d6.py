"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute all C(11,3) = 165 triplets
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)))
_T_I = _TRIPLES[:, 0]
_T_J = _TRIPLES[:, 1]
_T_K = _TRIPLES[:, 2]


def _min_area(points):
    """Minimum triangle area among all C(N,3) triples."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return float(areas.min())


def _soft_min_area(points, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    min_val = np.min(areas)
    return min_val - np.log(np.sum(np.exp(-beta * (areas - min_val)))) / beta


def _params_to_points(params):
    """Convert 22 parameters to 11 points using barycentric mapping.
    Each point uses 2 params (u, v) mapped to barycentric coords."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        # Clamp to ensure validity
        u = np.clip(u, 0.0, 0.99)
        v = np.clip(v, 0.0, 0.99)
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        # Barycentric: (1-u-v)*C + u*A + v*B
        # A=(0,0), B=(1,0), C=(0.5, H)
        points[i, 0] = v + 0.5 * (1 - u - v)
        points[i, 1] = H * (1 - u - v)
    return points


def _is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -1e-9) and 
            np.all(y <= np.sqrt(3) * x + 1e-9) and 
            np.all(y <= np.sqrt(3) * (1 - x) + 1e-9))


def _objective_soft(params):
    """Objective function: negative soft-min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_soft_min_area(points, beta=50.0)


def _objective_exact(params):
    """Objective function: negative exact min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_min_area(points)


def _simulated_annealing(params0, start_time, deadline, rng):
    """Simulated annealing on smooth surrogate landscape."""
    current_params = params0.copy()
    current_score = _objective_soft(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    # Temperature schedule
    T0 = 0.1
    Tend = 0.001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Exponential cooling
        T = T0 * (Tend / T0) ** (step / n_steps)
        
        # Generate perturbation
        perturbation = rng.normal(0, 0.05, size=2 * N)
        new_params = current_params + perturbation
        
        # Evaluate
        new_score = _objective_soft(new_params)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = current_score - new_score  # positive means worse
            if delta < 0:
                delta = 0
            prob = np.exp(-delta / max(T, 1e-10))
            if rng.random() < prob:
                current_params = new_params
                current_score = new_score
    
    return best_params, -best_score


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_params = None
    best_score = -np.inf
    
    # Strategy 1: Differential evolution with soft objective
    bounds = [(0, 1)] * (2 * N)
    
    try:
        de_res = differential_evolution(
            _objective_soft,
            bounds,
            maxiter=100,
            popsize=30,
            tol=1e-7,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        if de_res.fun < 1e6:
            exact_score = -_objective_exact(de_res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_params = de_res.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Simulated annealing from best DE result
    if best_params is not None and time.time() < DEADLINE - 30:
        sa_params, sa_score = _simulated_annealing(best_params, start_time, DEADLINE, rng)
        exact_sa_score = -_objective_exact(sa_params)
        if exact_sa_score > best_score:
            best_score = exact_sa_score
            best_params = sa_params.copy()
    
    # Strategy 3: Local refinement with Nelder-Mead on exact objective
    if best_params is not None and time.time() < DEADLINE - 10:
        try:
            res = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < 1e6:
                exact_res_score = -res.fun
                if exact_res_score > best_score:
                    best_score = exact_res_score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Multiple random restarts if time permits
    if time.time() < DEADLINE - 20:
        for restart in range(5):
            if time.time() > DEADLINE - 10:
                break
            random_params = rng.uniform(0, 1, size=2 * N)
            try:
                res = minimize(
                    _objective_soft,
                    random_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfev': 5000}
                )
                if res.fun < 1e6:
                    exact_score = -_objective_exact(res.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Fallback
    if best_params is None:
        points = np.zeros((N, 2))
        positions = [
            (0.5, H * 0.8),
            (0.25, H * 0.5),
            (0.75, H * 0.5),
            (0.1, H * 0.2),
            (0.5, H * 0.2),
            (0.9, H * 0.2),
            (0.35, H * 0.65),
            (0.65, H * 0.65),
            (0.2, H * 0.35),
            (0.8, H * 0.35),
            (0.5, H * 0.45),
        ]
        for i, (x, y) in enumerate(positions):
            points[i] = [x, y]
        return points
    
    points = _params_to_points(best_params)
    
    # Ensure points are inside triangle (clamp if needed)
    for i in range(N):
        x, y = points[i]
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x if x > 0 else 0
            if x == 0:
                y = 0
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) if x < 1 else 0
            if x == 1:
                y = 0
        points[i] = [x, y]
    
    return points
