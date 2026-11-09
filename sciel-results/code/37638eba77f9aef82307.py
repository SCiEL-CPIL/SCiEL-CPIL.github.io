"""Dzugutov cluster global minima via basin hopping with polytetrahedral templates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def _dzugutov_energy(positions):
    """Compute Dzugutov energy for given positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-8:
                continue
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _icosahedron(scale=1.0):
    """12 vertices of regular icosahedron, edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    d = np.linalg.norm(v[0] - v[4])
    return v * (scale / d)


def _random_compact(n, rng, spacing=1.0):
    """Random compact cluster with min separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.7
    pts = []
    max_tries = 500 * n
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in pts):
                pts.append(p)
        tries += 1
    # Fill remaining with jittered points
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3) + rng.normal(0, 0.1, 3)
        pts.append(p)
    return np.array(pts)


def _make_templates(n, rng):
    """Generate diverse polytetrahedral-inspired templates."""
    templates = []
    # 1. Random compact
    for _ in range(3):
        templates.append(_random_compact(n, rng, spacing=1.05))
    # 2. Icosahedral core + shell
    if n >= 13:
        core = _icosahedron(1.0)
        if n >= 13:
            t = core.copy()
        else:
            t = core[:n]
        # Add second shell points for larger n
        if n > 12:
            # Face centers of icosahedron as second shell
            face_centers = []
            # Icosahedron faces: 20 triangular faces
            # Approximate: use midpoints of edges scaled outward
            edges = [
                (0,4),(0,8),(0,10),(1,4),(1,8),(1,10),
                (2,5),(2,9),(2,11),(3,5),(3,9),(3,11),
                (4,5),(4,7),(4,9),(5,6),(5,7),(5,8),
                (6,7),(6,9),(6,11),(7,8),(7,10),(8,10),
                (9,10),(9,11),(10,11)
            ]
            for i, j in edges:
                fc = (core[i] + core[j]) / 2.0
                fc = fc * 1.6  # push outward
                face_centers.append(fc)
            face_centers = np.array(face_centers)
            # Add face centers to fill up to n
            extra = min(n - len(t), len(face_centers))
            if extra > 0:
                t = np.vstack([t, face_centers[:extra]])
        if len(t) >= n:
            templates.append(t[:n])
        elif len(t) > 0:
            # Pad with random
            while len(t) < n:
                t = np.vstack([t, rng.uniform(-2, 2, 3)])
            templates.append(t)
    # 3. Polytetrahedral-like: multiple small icosahedral clusters
    if n >= 13:
        # Two overlapping icosahedra
        c1 = _icosahedron(0.9)
        c2 = _icosahedron(0.9) + np.array([1.5, 0, 0])
        t = np.vstack([c1, c2])
        if len(t) >= n:
            templates.append(t[:n])
        else:
            while len(t) < n:
                t = np.vstack([t, rng.uniform(-2, 2, 3)])
            templates.append(t)
    # 4. FCC-like cubic
    if n >= 4:
        side = int(np.ceil(n ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(pts) < n:
                        pts.append([i, j, k])
        if len(pts) >= n:
            templates.append(np.array(pts[:n], dtype=float) * 1.0)
    return templates


def _basin_hopping_step(pos, rng, step_size=0.15, max_iter=500, tol=1e-8):
    """
    Basin hopping: random displacement + local relaxation via gradient descent.
    Returns (energy, positions) after local minimization.
    """
    n = len(pos)
    e, relaxed = relax(pos, 'dzugutov')
    return e, relaxed


def _gradient_descent(pos, max_iter=2000, tol=1e-10, lr_init=0.01, lr_decay=0.999):
    """
    Simple gradient descent with adaptive learning rate using dzugutov_energy_gradient.
    Returns (energy, positions).
    """
    n = len(pos)
    lr = lr_init
    best_e = _dzugutov_energy(pos)
    best_pos = pos.copy()
    
    for it in range(max_iter):
        grad = dzugutov_energy_gradient(pos)
        # Check gradient norm
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Adaptive step size
        step = lr / (gnorm + 1e-8)
        new_pos = pos - step * grad
        
        # Check minimum separation constraint
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(new_pos[i] - new_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            new_e = _dzugutov_energy(new_pos)
            if new_e < best_e:
                pos = new_pos
                best_e = new_e
                best_pos = pos.copy()
            else:
                lr *= 0.9
        else:
            lr *= 0.9
        
        if lr < 1e-6:
            break
    
    # Final relaxation via the provided relax function
    e, relaxed = relax(best_pos, 'dzugutov')
    return e, relaxed


def _bh_search(n, rng, time_budget=35.0):
    """
    Basin hopping with polytetrahedral seeds and gradient-based local minimization.
    """
    # Initialize with diverse templates
    templates = _make_templates(n, rng)
    
    # Relax all templates to get starting basins
    initial_basins = []
    for t in templates:
        if len(t) == n:
            e, pos = relax(t, 'dzugutov')
            initial_basins.append((e, pos))
    
    # Fill with random if needed
    min_basins = 6
    while len(initial_basins) < min_basins:
        t = _random_compact(n, rng, spacing=1.05)
        e, pos = relax(t, 'dzugutov')
        initial_basins.append((e, pos))
    
    # Sort by energy
    initial_basins.sort(key=lambda x: x[0])
    
    best_e, best_pos = initial_basins[0]
    start = time.time()
    
    # Basin hopping loop
    pop_size = len(initial_basins)
    population = initial_basins[:]
    
    while time.time() - start < time_budget:
        if time.time() > _DEADLINE - 25:
            break
        
        # Pick a parent from the population (tournament selection)
        idx1 = rng.integers(pop_size)
        idx2 = rng.integers(pop_size)
        parent_idx = idx1 if population[idx1][0] < population[idx2][0] else idx2
        parent_e, parent_pos = population[parent_idx]
        
        # Generate a perturbed child
        # Use larger steps than GA mutation to explore new basins
        sigma = rng.uniform(0.1, 0.3)
        child_pos = parent_pos + rng.normal(0, sigma, parent_pos.shape)
        
        # Local minimization via gradient descent + relax
        child_e, child_relaxed = _gradient_descent(child_pos, max_iter=500, tol=1e-9)
        
        # Update population
        if child_e < best_e:
            best_e, best_pos = child_e, child_relaxed
        
        # Replace worst member of population if child is better
        worst_idx = np.argmax([x[0] for x in population])
        if child_e < population[worst_idx][0]:
            population[worst_idx] = (child_e, child_relaxed)
        
        # Occasionally add a fresh random start to maintain diversity
        if rng.random() < 0.15:
            t = _random_compact(n, rng, spacing=1.05)
            e, pos = relax(t, 'dzugutov')
            population[worst_idx] = (e, pos)
    
    return best_pos


def _hybrid_search(n, rng, time_budget=35.0):
    """
    Hybrid: start with basin hopping, then refine with GA-like local moves.
    """
    # Phase 1: Basin hopping for broad exploration
    bh_pos = _bh_search(n, rng, time_budget=time_budget * 0.6)
    bh_e = _dzugutov_energy(bh_pos)
    
    # Phase 2: Local refinement around the best solution
    # Use small random perturbations + gradient descent
    best_e, best_pos = bh_e, bh_pos
    start = time.time()
    remaining = time_budget * 0.4
    
    while time.time() - start < remaining:
        if time.time() > _DEADLINE - 20:
            break
        
        # Small perturbation
        sigma = rng.uniform(0.03, 0.08)
        child_pos = best_pos + rng.normal(0, sigma, best_pos.shape)
        
        # Gradient descent
        child_e, child_relaxed = _gradient_descent(child_pos, max_iter=300, tol=1e-10)
        
        if child_e < best_e:
            best_e, best_pos = child_e, child_relaxed
    
    return best_pos


def construct(instance):
    """Best of hybrid search on Dzugutov clusters."""
    size = instance["n"]
    rng = np.random.default_rng(size * 1000 + 42)

    # Time budget per instance
    remaining = _DEADLINE - time.time()
    time_budget = max(5.0, min(45.0, remaining / 1.5))

    result = _hybrid_search(size, rng, time_budget=time_budget)
    return result
# EVOLVE-BLOCK-END
