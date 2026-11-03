"""Packing 50 equal circles in a unit circle using Delaunay-based stochastic gradient ascent."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import warnings

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import warnings
import time

def _compute_packing_radius(x):
    """Compute the packing radius for a configuration."""
    x = x.reshape(-1, 2)
    n = len(x)
    if n < 2:
        return 1.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    radii = np.sqrt(np.sum(x**2, axis=1))
    min_bound_clear = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_bound_clear)

def _get_delaunay_edges(x):
    """Get Delaunay triangulation edges as a list of (i, j) pairs."""
    x = x.reshape(-1, 2)
    try:
        tri = Delaunay(x)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        # Fallback: return all pairs if triangulation fails
        n = len(x)
        return [(i, j) for i in range(n) for j in range(i+1, n)]

def _initial_hexagonal():
    """Generate a good initial hexagonal packing."""
    points = []
    # Try multiple spacings to get exactly N points
    for spacing in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        pts = []
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing)**2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            radii = np.sqrt(np.sum(pts**2, axis=1))
            idx = np.argsort(radii)[:N]
            return pts[idx]
    # Fallback: random
    return _initial_random()

def _initial_random(seed=42):
    """Generate random initial configuration inside unit circle."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= 0.81:
                points.append([x, y])
                break
    return np.array(points)

def _stochastic_gradient_step(x, edges, step_size, active_eps=1e-3):
    """
    One stochastic gradient ascent step using Delaunay edges.
    Only consider constraints from Delaunay edges + boundary.
    """
    x = x.reshape(-1, 2)
    n = len(x)
    
    # Compute clearances for Delaunay edges
    edge_clears = []
    for (i, j) in edges:
        d = np.linalg.norm(x[i] - x[j])
        if d > 1e-10:
            edge_clears.append((d / 2.0, i, j, 'pair'))
    
    # Boundary clearances
    radii = np.sqrt(np.sum(x**2, axis=1))
    bound_clears = [(1.0 - radii[i], i, -1, 'bound') for i in range(n)]
    
    all_clears = edge_clears + bound_clears
    
    if not all_clears:
        return x
    
    r_min = min(c[0] for c in all_clears)
    
    # Compute gradient from active constraints
    grad = np.zeros_like(x)
    
    for (c, i, j, ctype) in all_clears:
        if c < r_min + active_eps:
            if ctype == 'pair':
                d = np.linalg.norm(x[i] - x[j])
                if d > 1e-10:
                    direction = (x[i] - x[j]) / d
                    grad[i] += direction / 2.0
                    grad[j] += -direction / 2.0
            else:  # bound
                r = radii[i]
                if r > 1e-10:
                    direction = -x[i] / r
                    grad[i] += direction
    
    # Normalize and apply step
    norm = np.sqrt(np.sum(grad**2))
    if norm > 1e-10:
        grad /= norm
        x_new = x + step_size * grad
    else:
        x_new = x.copy()
    
    # Project back to unit circle
    radii_new = np.sqrt(np.sum(x_new**2, axis=1))
    for i in range(n):
        if radii_new[i] > 0.999:
            x_new[i] *= 0.999 / radii_new[i]
    
    return x_new

def construct():
    """Construct packing of N equal circles in unit circle using Delaunay-based optimization."""
    # Start with hexagonal initial configuration
    x = _initial_hexagonal()
    
    best_x = x.copy()
    best_r = _compute_packing_radius(x)
    
    # Main optimization loop
    total_iterations = 5000
    start_time = time.time()
    max_time = 100  # seconds, leave margin for evaluation
    
    # Adaptive step size
    step_size = 0.01
    patience = 0
    best_r_so_far = best_r
    no_improve_count = 0
    
    for iteration in range(total_iterations):
        # Check time
        if time.time() - start_time > max_time:
            break
        
        # Get Delaunay edges
        edges = _get_delaunay_edges(x)
        
        # Take a gradient step
        x_new = _stochastic_gradient_step(x, edges, step_size)
        
        # Evaluate
        r_new = _compute_packing_radius(x_new)
        
        # Accept if better, or with some probability if worse (simulated annealing style)
        if r_new > best_r_so_far:
            x = x_new
            best_r_so_far = r_new
            no_improve_count = 0
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.02, 0.05)
        else:
            # Accept with decreasing probability
            accept_prob = 0.1 * (1.0 - iteration / total_iterations)
            if np.random.random() < accept_prob:
                x = x_new
            else:
                # Reject, decrease step size
                step_size *= 0.95
            no_improve_count += 1
        
        # Update best
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
        
        # Decrease step size periodically
        if iteration % 1000 == 999:
            step_size *= 0.5
        
        # If stuck for too long, try a small random perturbation
        if no_improve_count > 200:
            x = best_x.copy() + np.random.normal(0, 0.01, size=(N, 2))
            # Project back
            radii = np.sqrt(np.sum(x**2, axis=1))
            for i in range(N):
                if radii[i] > 0.99:
                    x[i] *= 0.99 / radii[i]
            no_improve_count = 0
            step_size = 0.01
    
    # Final polish: run more iterations with smaller step
    if time.time() - start_time < max_time - 5:
        step_size = 0.001
        for iteration in range(2000):
            if time.time() - start_time > max_time:
                break
            edges = _get_delaunay_edges(x)
            x_new = _stochastic_gradient_step(x, edges, step_size, active_eps=1e-4)
            r_new = _compute_packing_radius(x_new)
            if r_new > best_r:
                x = x_new
                best_r = r_new
                best_x = x_new.copy()
            else:
                x = x_new  # still accept, we're in polish mode
        
        # Re-evaluate
        r_final = _compute_packing_radius(x)
        if r_final > best_r:
            best_r = r_final
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
