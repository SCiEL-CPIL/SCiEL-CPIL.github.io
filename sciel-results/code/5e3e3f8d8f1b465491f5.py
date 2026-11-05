"""Dzugutov cluster global minima search with polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A_const = 5.82
B_const = 1.28
a_const = 1.87
b_const = 1.94
c_const = 1.10
d_const = 0.27
m_const = 16.0


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _make_initial_guess(n, rng, style='icosahedral'):
    """
    Generate an initial guess for n atoms.
    style: 'icosahedral' or 'random'
    """
    if style == 'random':
        # Random compact structure
        R = 0.8 * (n / (4/3 * np.pi)) ** (1/3) * 2.0
        points = []
        max_attempts = 1000
        while len(points) < n and max_attempts > 0:
            max_attempts -= 1
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) > R:
                continue
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points.append(p)
        return np.array(points)

    # Icosahedral style
    # Start with 13-atom icosahedron (center + 12 shell)
    # The minimum energy distance for Dzugutov is around r ~ 1.8-2.0
    # Let's set the first shell radius to 1.9
    R1 = 1.9
    core = np.vstack([np.zeros(3), _generate_icosahedron(R1)])
    
    if n <= 13:
        return core[:n]
    
    points = list(core)
    remaining = n - len(points)
    
    # Add remaining atoms in a second shell
    # The second shell in a 55-atom icosahedron is at roughly 2*R1
    R2 = 3.8
    
    for _ in range(remaining):
        # Random point on sphere of radius R2
        u = rng.uniform(-1, 1)
        v = rng.uniform(0, 2 * np.pi)
        theta = np.arccos(u)
        p = np.array([
            R2 * np.sin(theta) * np.cos(v),
            R2 * np.sin(theta) * np.sin(v),
            R2 * np.cos(theta)
        ])
        
        # Check minimum distance to existing points
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        
        if valid:
            points.append(p)
        else:
            # Try a few more times with slightly different radius
            for attempt in range(5):
                R_try = R2 + (attempt - 2) * 0.2
                u = rng.uniform(-1, 1)
                v = rng.uniform(0, 2 * np.pi)
                theta = np.arccos(u)
                p_try = np.array([
                    R_try * np.sin(theta) * np.cos(v),
                    R_try * np.sin(theta) * np.sin(v),
                    R_try * np.cos(theta)
                ])
                valid = True
                for q in points:
                    if np.linalg.norm(p_try - q) < 0.55:
                        valid = False
                        break
                if valid:
                    points.append(p_try)
                    break
            else:
                # Just add it; relaxation will fix overlaps
                points.append(p)
                
    return np.array(points[:n])


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    # Number of restarts
    # We have ~50s per instance. Relaxation takes ~0.1-0.5s.
    # We can do ~20-50 restarts. Let's do 20 icosahedral and 5 random.
    n_ico_restarts = 20
    n_rand_restarts = 5
    
    for i in range(n_ico_restarts + n_rand_restarts):
        if time.time() > deadline:
            break
            
        # Alternate between icosahedral and random
        if i < n_ico_restarts:
            style = 'icosahedral'
            seed = size * 100 + i
        else:
            style = 'random'
            seed = size * 100 + 1000 + (i - n_ico_restarts)
            
        rng = np.random.default_rng(seed)
        
        try:
            points = _make_initial_guess(size, rng, style=style)
            energy, positions = relax(points, potential, None)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
        except Exception as e:
            pass

    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(size)
        points = _make_initial_guess(size, rng, style='random')
        try:
            energy, positions = relax(points, potential, None)
            return positions
        except:
            return points
            
    return best_positions
# EVOLVE-BLOCK-END
