"""Rewritten program using topological initialization and parallel L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

# Try to import helper functions
try:
    from autoresearch_suite.chem_clusters import dzugutov_energy, dzugutov_energy_gradient
    HAS_HELPERS = True
except ImportError:
    HAS_HELPERS = False

# Dzugutov Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _manual_dzugutov_energy(positions):
    """Calculate Dzugutov energy manually."""
    n = len(positions)
    if n < 2:
        return 0.0
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-8:
                return 1e10
            if r < a:
                v1 = A * (r**(-m) - B) * np.exp(c / (r - a))
                v2 = B * np.exp(d / (r - b))
                energy += v1 + v2
            elif r < b:
                v2 = B * np.exp(d / (r - b))
                energy += v2
    return energy

def _manual_dzugutov_gradient(positions):
    """Calculate Dzugutov gradient manually."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-8:
                # Avoid division by zero, add random small vector
                grad[i] += np.random.uniform(-1, 1, 3) * 1e5
                grad[j] -= np.random.uniform(-1, 1, 3) * 1e5
                continue
            
            dv_dr = 0.0
            if r < a:
                # d/dr [ A(r^-m - B)exp(c/(r-a)) ]
                term1 = -m * r**(-m-1)
                term2 = (r**(-m) - B) * (-c / (r - a)**2)
                dv1 = A * (term1 * np.exp(c / (r - a)) + term2 * np.exp(c / (r - a)))
                # d/dr [ B exp(d/(r-b)) ]
                dv2 = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
                dv_dr = dv1 + dv2
            elif r < b:
                dv2 = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
                dv_dr = dv2
            
            # Force on i from j: -dv/dr * (r_i - r_j) / r
            # Gradient component for i: dv/dr * (r_i - r_j) / r
            factor = dv_dr / r
            grad[i] += factor * diff
            grad[j] -= factor * diff
    return grad

def _get_energy_and_gradient(positions):
    if HAS_HELPERS:
        try:
            e = dzugutov_energy(positions)
            g = dzugutov_energy_gradient(positions)
            return e, g
        except Exception:
            pass
    return _manual_dzugutov_energy(positions), _manual_dzugutov_gradient(positions)

def _generate_icosahedron(n):
    """Generate icosahedral-like structure."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    points = [np.zeros(3)]
    points.extend([v * 1.2 for v in verts]) # Slightly larger to avoid overlap with center?
    
    # Fill with random points if needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.uniform(-1.5, 1.5, 3)
        if np.linalg.norm(p) < 1.8:
             # Check distance
             if all(np.linalg.norm(p - q) > 0.6 for q in points):
                 points.append(p)
    return np.array(points[:n])

def _generate_random(n, seed):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        p = rng.uniform(-2.0, 2.0, 3)
        if all(np.linalg.norm(p - q) > 0.6 for q in points):
            points.append(p)
    return np.array(points)

def _optimize_one(pos, n):
    x0 = pos.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        e, _ = _get_energy_and_gradient(p)
        return e
        
    def gradient(x):
        p = x.reshape(-1, 3)
        _, g = _get_energy_and_gradient(p)
        return g.flatten()
        
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 300, 'gtol': 1e-6})
        return res.fun, res.x.reshape(-1, 3)
    except Exception:
        return float('inf'), pos

def construct(instance):
    n = instance['n']
    start_time = time.time()
    deadline = start_time + 280 # 20s margin
    
    # Generate starts
    ico = _generate_icosahedron(n)
    starts = [ico]
    
    # Add noisy icosahedra
    rng = np.random.default_rng(n)
    for i in range(4):
        noise = rng.normal(0, 0.2, (n, 3))
        starts.append(ico + noise)
        
    # Add randoms
    for i in range(10):
        starts.append(_generate_random(n, seed=n+i))
        
    best_energy = float('inf')
    best_pos = None
    
    # Parallel optimization
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_optimize_one, s, n): s for s in starts}
        
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except Exception:
                pass
                
    if best_pos is None:
        best_pos = ico
        
    # Final polish on the best candidate
    if time.time() < deadline - 10:
        e, pos = _optimize_one(best_pos, n)
        if e < best_energy:
            best_energy = e
            best_pos = pos
            
    return best_pos
# EVOLVE-BLOCK-END
