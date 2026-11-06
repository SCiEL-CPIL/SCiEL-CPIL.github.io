"""Dzugutov cluster global minima: discrete icosahedral shell assignment + genetic algorithm + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Safety margin before 330s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _get_icosahedron_vertices():
    """Return 12 unit icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_icosahedron_edge_midpoints():
    """Return 30 unit icosahedron edge midpoint directions."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 0.01:
                edges.append((i, j))
    edge_mids = np.array([(verts_unit[i] + verts_unit[j]) / 2.0 for i, j in edges])
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
    return edge_mids


def _get_icosahedron_face_centers():
    """Return 20 unit icosahedron face center directions."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 2], [0, 2, 6], [0, 6, 1],
        [1, 6, 9], [1, 9, 5], [1, 5, 3], [1, 3, 8],
        [2, 4, 10], [2, 10, 7], [2, 7, 6],
        [3, 5, 11], [3, 11, 10], [3, 10, 4], [3, 4, 8],
        [5, 9, 11], [5, 11, 7], [5, 7, 3],
        [7, 11, 10], [7, 10, 2], [7, 2, 6], [7, 6, 9], [7, 9, 5]
    ]
    face_centers = []
    for f in faces:
        fc = (verts_unit[f[0]] + verts_unit[f[1]] + verts_unit[f[2]]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    return np.array(face_centers)


def _build_from_shell_assignment(shell_types, radii, n):
    """Build positions from shell type assignments and radii.
    
    shell_types: list of shell type indices (0=center, 1=verts, 2=edges, 3=faces)
    radii: list of radius values for each shell
    n: number of atoms
    """
    verts = _get_icosahedron_vertices()
    edges = _get_icosahedron_edge_midpoints()
    faces = _get_icosahedron_face_centers()
    
    points = []
    for stype, r in zip(shell_types, radii):
        if stype == 0:  # center
            points.append([0.0, 0.0, 0.0])
        elif stype == 1:  # 12 vertices
            points.extend((verts * r).tolist())
        elif stype == 2:  # 30 edge midpoints
            points.extend((edges * r).tolist())
        elif stype == 3:  # 20 face centers
            points.extend((faces * r).tolist())
    
    points = np.array(points)
    
    if len(points) < n:
        # Add remaining atoms on outer shell
        extra = n - len(points)
        outer_r = max(radii) * 1.3 if radii else 1.0
        rng = np.random.default_rng(42)
        for _ in range(extra):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * outer_r
            points = np.vstack([points, p])
    elif len(points) > n:
        points = points[:n]
    
    return points


def _genetic_search(n, potential, parameter, pop_size=30, generations=20, seed=0):
    """Genetic algorithm over icosahedral shell configurations."""
    rng = np.random.default_rng(seed)
    
    # Shell types: 0=center, 1=12 verts, 2=30 edges, 3=20 faces
    # We'll encode a configuration as (shell_types, radii)
    # For n atoms, we need to choose which shells to use and their radii
    
    def _random_config():
        """Generate a random shell configuration."""
        # Start with center
        shell_types = [0]
        radii = [0.0]
        remaining = n - 1
        
        # Add shells
        while remaining > 0:
            # Choose shell type
            if remaining >= 30:
                options = [1, 2, 3]
            elif remaining >= 20:
                options = [1, 3]
            elif remaining >= 12:
                options = [1]
            else:
                options = [1]  # Will overflow, handle later
            
            stype = options[rng.integers(len(options))]
            # Choose radius
            base_r = 0.4 + 0.15 * len(shell_types)
            r = base_r + rng.uniform(-0.05, 0.05)
            shell_types.append(stype)
            radii.append(r)
            
            # Subtract atoms
            if stype == 1:
                remaining -= 12
            elif stype == 2:
                remaining -= 30
            elif stype == 3:
                remaining -= 20
        
        return shell_types, radii
    
    def _evaluate(config):
        """Evaluate a configuration: build positions, relax, return energy."""
        shell_types, radii = config
        try:
            positions = _build_from_shell_assignment(shell_types, radii, n)
            # Check minimum distance
            min_d = np.inf
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_d:
                        min_d = d
            if min_d < 0.5:
                return 1e10
            energy, relaxed = relax(positions, potential, parameter)
            return energy
        except:
            return 1e10
    
    def _crossover(c1, c2):
        """Crossover two configurations."""
        st1, r1 = c1
        st2, r2 = c2
        # Randomly combine shells
        new_st = []
        new_r = []
        for i in range(max(len(st1), len(st2))):
            if i < len(st1) and (i < len(st2) or rng.random() < 0.5):
                if i < len(st2) and rng.random() < 0.5:
                    new_st.append(st2[i])
                    new_r.append(r2[i] + rng.normal(0, 0.03))
                else:
                    new_st.append(st1[i])
                    new_r.append(r1[i] + rng.normal(0, 0.03))
            elif i < len(st2):
                new_st.append(st2[i])
                new_r.append(r2[i] + rng.normal(0, 0.03))
        return (new_st, new_r)
    
    def _mutate(config, rate=0.3):
        """Mutate a configuration."""
        st, r = config
        new_st = list(st)
        new_r = list(r)
        for i in range(len(new_st)):
            if rng.random() < rate:
                # Mutate radius
                new_r[i] += rng.normal(0, 0.08)
                if new_r[i] < 0.2:
                    new_r[i] = 0.2
            if rng.random() < rate * 0.3:
                # Mutate shell type
                new_st[i] = int(rng.integers(0, 4))
        # Occasionally add or remove a shell
        if rng.random() < 0.2 and len(new_st) > 1:
            idx = int(rng.integers(1, len(new_st)))
            new_st.pop(idx)
            new_r.pop(idx)
        elif rng.random() < 0.1 and len(new_st) < 5:
            new_st.append(int(rng.integers(1, 4)))
            new_r.append(new_r[-1] + 0.2 + rng.uniform(0, 0.1))
        return (new_st, new_r)
    
    # Initialize population
    population = [_random_config() for _ in range(pop_size)]
    energies = [_evaluate(c) for c in population]
    
    best_config = population[int(np.argmin(energies))]
    best_energy = min(energies)
    
    for gen in range(generations):
        if _time_remaining() < 30:
            break
        
        # Sort by energy
        order = np.argsort(energies)
        population = [population[i] for i in order]
        energies = [energies[i] for i in order]
        
        # Update best
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_config = population[0]
        
        # Create new population
        new_population = []
        # Elitism: keep top 5
        for i in range(min(5, pop_size)):
            new_population.append(population[i])
        
        # Fill rest with crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            i1 = int(rng.integers(0, min(10, pop_size)))
            i2 = int(rng.integers(0, min(10, pop_size)))
            if energies[i1] < energies[i2]:
                parent1 = population[i1]
            else:
                parent1 = population[i2]
            i3 = int(rng.integers(0, min(10, pop_size)))
            i4 = int(rng.integers(0, min(10, pop_size)))
            if energies[i3] < energies[i4]:
                parent2 = population[i3]
            else:
                parent2 = population[i4]
            
            if rng.random() < 0.7:
                child = _crossover(parent1, parent2)
            else:
                child = parent1
            
            child = _mutate(child)
            new_population.append(child)
        
        population = new_population
        energies = [_evaluate(c) for c in population]
    
    # Final relaxation of best
    positions = _build_from_shell_assignment(best_config[0], best_config[1], n)
    try:
        energy, relaxed = relax(positions, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_config_positions = relaxed
        else:
            best_config_positions = positions
    except:
        best_config_positions = positions
    
    return best_energy, best_config_positions


def _basin_hopping(positions, potential, parameter, n_kicks=10, kick_sigma=0.15, seed=0):
    """Basin hopping around a starting position."""
    rng = np.random.default_rng(seed)
    best_energy, best_pos = positions.copy(), positions.copy()
    e0, _ = relax(positions, potential, parameter)
    best_energy = e0
    
    for i in range(n_kicks):
        if _time_remaining() < 20:
            break
        perturbed = positions + rng.normal(0, kick_sigma, positions.shape)
        try:
            energy, new_pos = relax(perturbed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = new_pos.copy()
        except:
            pass
    
    return best_energy, best_pos


def _random_compact(n, seed, spacing=1.1, radius_factor=0.75):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.8 * spacing
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Construct low-energy Dzugutov cluster using discrete icosahedral shell GA + basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Genetic algorithm over icosahedral shell configurations
    ga_energy, ga_pos = _genetic_search(size, potential, parameter, pop_size=25, generations=15, seed=size)
    if ga_energy < best_energy:
        best_energy = ga_energy
        best_positions = ga_pos.copy()
    
    if _time_remaining() < 80:
        return best_positions
    
    # Strategy 2: Basin hopping around GA result
    bh_energy, bh_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=10, kick_sigma=0.12, seed=size
    )
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_positions = bh_pos.copy()
    
    if _time_remaining() < 70:
        return best_positions
    
    # Strategy 3: Second GA run with different seed
    ga2_energy, ga2_pos = _genetic_search(size, potential, parameter, pop_size=20, generations=10, seed=size + 100)
    if ga2_energy < best_energy:
        best_energy = ga2_energy
        best_positions = ga2_pos.copy()
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 4: Basin hopping around second GA
    bh2_energy, bh2_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_sigma=0.15, seed=size + 200
    )
    if bh2_energy < best_energy:
        best_energy = bh2_energy
        best_positions = bh2_pos.copy()
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 5: Random compact + relax
    rc_points = _random_compact(size, seed=size, spacing=1.1, radius_factor=0.75)
    try:
        energy, positions = relax(rc_points, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    except:
        pass
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 6: Basin hopping around random compact
    bh3_energy, bh3_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_sigma=0.18, seed=size + 300
    )
    if bh3_energy < best_energy:
        best_energy = bh3_energy
        best_positions = bh3_pos.copy()
    
    if _time_remaining() < 30:
        return best_positions
    
    # Strategy 7: Third GA with different parameters
    ga3_energy, ga3_pos = _genetic_search(size, potential, parameter, pop_size=20, generations=10, seed=size + 400)
    if ga3_energy < best_energy:
        best_energy = ga3_energy
        best_positions = ga3_pos.copy()
    
    if _time_remaining() < 20:
        return best_positions
    
    # Strategy 8: Final basin hopping
    bh4_energy, bh4_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_sigma=0.10, seed=size + 500
    )
    if bh4_energy < best_energy:
        best_energy = bh4_energy
        best_positions = bh4_pos.copy()
    
    return best_positions
# EVOLVE-BLOCK-END
