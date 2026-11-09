"""Dzugutov cluster global minima via GA with polytetrahedral templates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


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


def _blend_crossover(a, b, rng):
    """Blend crossover: child = alpha*a + (1-alpha)*b."""
    alpha = rng.uniform(0.3, 0.7)
    return alpha * a + (1 - alpha) * b


def _mutate(x, rng, sigma=0.15):
    """Gaussian mutation."""
    return x + rng.normal(0, sigma, x.shape)


def _ga_search(n, rng, time_budget=35.0):
    """Genetic algorithm with local relaxation."""
    pop_size = 24
    # Initialize population
    templates = _make_templates(n, rng)
    pop = []
    for t in templates:
        if len(t) == n:
            e, pos = relax(t, 'dzugutov')
            pop.append((e, pos))
    # Fill with random
    while len(pop) < pop_size:
        t = _random_compact(n, rng, spacing=1.05)
        e, pos = relax(t, 'dzugutov')
        pop.append((e, pos))

    # Sort and keep best
    pop.sort(key=lambda x: x[0])
    best_e, best_pos = pop[0]

    gen = 0
    start = time.time()
    elite_size = 4

    while time.time() - start < time_budget:
        gen += 1
        # Selection: tournament
        def select():
            idx1 = rng.integers(pop_size)
            idx2 = rng.integers(pop_size)
            return pop[idx1] if pop[idx1][0] < pop[idx2][0] else pop[idx2]

        new_pop = []
        # Elitism
        for i in range(min(elite_size, len(pop))):
            new_pop.append(pop[i])

        # Generate offspring
        while len(new_pop) < pop_size:
            parent1 = select()
            parent2 = select()
            child = _blend_crossover(parent1[1], parent2[1], rng)
            child = _mutate(child, rng, sigma=0.12)
            e, pos = relax(child, 'dzugutov')
            new_pop.append((e, pos))

        pop = new_pop
        pop.sort(key=lambda x: x[0])
        if pop[0][0] < best_e:
            best_e, best_pos = pop[0]

        # Check deadline
        if time.time() > _DEADLINE - 30:
            break

    return best_pos


def construct(instance):
    """Best of GA search on Dzugutov clusters."""
    size = instance["n"]
    rng = np.random.default_rng(size * 1000 + 42)

    # Time budget per instance
    remaining = _DEADLINE - time.time()
    time_budget = max(5.0, min(45.0, remaining / 1.5))

    result = _ga_search(size, rng, time_budget=time_budget)
    return result
# EVOLVE-BLOCK-END
