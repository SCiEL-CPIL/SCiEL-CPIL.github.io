<<<<<<< SEARCH
def _relax_with_timeout(positions, potential, parameter, timeout=15.0):
    """Relax positions with a timeout guard."""
    start = time.time()
    try:
        energy, pos = relax(positions.copy(), potential, parameter)
        return energy, pos
    except Exception:
        pass
    # Fallback: gradient descent
    pos = positions.copy()
    lr = 0.01
    for step in range(200):
        if time.time() - start > timeout:
            break
        try:
            grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        pos = pos - lr * grad / (norm + 1e-12) * 0.1
        lr = max(0.001, lr * 0.99)
    energy = _dzugutov_energy(pos)
    return energy, pos


def _blend_crossover(parent1, parent2, rng):
    """Blend crossover: child = alpha*p1 + (1-alpha)*p2 with per-atom alpha."""
    n = len(parent1)
    alphas = rng.uniform(0.2, 0.8, n)[:, np.newaxis]
    child = alphas * parent1 + (1.0 - alphas) * parent2
    return child


def _mutate(positions, rng, sigma=0.1):
    """Gaussian mutation."""
    return positions + rng.normal(0, sigma, positions.shape)


def _ga_search(n, potential, parameter, pop_size=8, generations=15, time_budget=40.0):
    """Genetic algorithm with polytetrahedral templates."""
    start_time = time.time()
    
    # Initialize population with diverse templates
    templates = []
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.1))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.15))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.05))
    templates.append(_generate_decahedral(n, base_spacing=1.1))
    templates.append(_generate_compact_random(n, seed=42))
    templates.append(_generate_compact_random(n, seed=123))
    templates.append(_generate_fcc(n, spacing=1.1))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.2))
    
    # Relax all templates and evaluate
    population = []
    energies = []
    for t in templates:
        if time.time() - start_time > time_budget * 0.5:
            break
        try:
            e, pos = _relax_with_timeout(t, potential, parameter, timeout=8.0)
            population.append(pos)
            energies.append(e)
        except Exception:
            population.append(t)
            energies.append(_dzugutov_energy(t))
    
    if not population:
        return templates[0], _dzugutov_energy(templates[0])
    
    best_idx = np.argmin(energies)
    best_pos = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    # GA evolution
    rng = np.random.default_rng(n)
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Selection: keep top half
        sorted_idx = np.argsort(energies)
        keep = sorted_idx[:pop_size // 2]
        parents = [population[i] for i in keep]
        parent_energies = [energies[i] for i in keep]
        
        # Generate offspring
        new_population = []
        new_energies = []
        
        # Elitism: keep best
        new_population.append(population[sorted_idx[0]])
        new_energies.append(energies[sorted_idx[0]])
        
        for _ in range(pop_size - 1):
            if time.time() - start_time > time_budget:
                break
            
            # Crossover
            i = rng.randint(0, len(parents))
            j = rng.randint(0, len(parents))
            if i == j:
                j = (j + 1) % len(parents)
            child = _blend_crossover(parents[i], parents[j], rng)
            
            # Mutation
            if rng.random() < 0.5:
                child = _mutate(child, rng, sigma=0.08)
            
            # Relax
            try:
                e, pos = _relax_with_timeout(child, potential, parameter, timeout=5.0)
                new_population.append(pos)
                new_energies.append(e)
            except Exception:
                new_population.append(child)
                new_energies.append(_dzugutov_energy(child))
        
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx].copy()
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster via GA with polytetrahedral templates."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check remaining time
    remaining = _remaining()
    if remaining < 10:
        # Emergency fallback
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            energy, pos = _relax_with_timeout(seed, potential, parameter, timeout=5.0)
            return pos
        except Exception:
            return _generate_mackay_icosahedron(size, base_spacing=1.1)
    
    # Allocate time budget for GA
    ga_budget = min(remaining * 0.6, 45.0)
    
    # Run GA
    try:
        best_pos, best_energy = _ga_search(size, potential, parameter, 
                                           pop_size=8, generations=12, 
                                           time_budget=ga_budget)
    except Exception:
        # Fallback to simple seed
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy, best_pos = _relax_with_timeout(seed, potential, parameter, timeout=10.0)
        except Exception:
            best_pos = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy = _dzugutov_energy(best_pos)
    
    # Final refinement: perturb and relax around best
    remaining2 = _remaining()
    if remaining2 > 15:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if _remaining() < 10:
                break
            perturbed = best_pos + rng2.normal(0, 0.04, best_pos.shape)
            try:
                energy, positions = _relax_with_timeout(perturbed, potential, parameter, timeout=8.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions
            except Exception:
                continue
    
    return best_pos
# EVOLVE-BLOCK-END
=======
def _basin_hopping_chain(n, initial_pos, rng, max_steps=20, time_budget=30.0, perturb_scale=0.15):
    """Single basin hopping chain with L-BFGS-B local minimization."""
    start_time = time.time()
    current_pos = initial_pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    
    # First local minimization
    current_energy, current_pos = _local_minimize_l_bfgs(current_pos, timeout=8.0)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb
        scale = perturb_scale * (1.0 - 0.5 * step / max_steps)
        perturbed = current_pos + rng.normal(0, scale, current_pos.shape)
        
        # Local minimize
        perturbed_energy, perturbed_pos = _local_minimize_l_bfgs(perturbed, timeout=8.0)
        
        # Metropolis acceptance
        if perturbed_energy < current_energy:
            current_energy = perturbed_energy
            current_pos = perturbed_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            delta_e = perturbed_energy - current_energy
            if delta_e < 0.1:  # Small penalty, accept with some probability
                if rng.random() < 0.3:
                    current_energy = perturbed_energy
                    current_pos = perturbed_pos
    
    return best_energy, best_pos


def _parallel_basin_hopping(n, n_chains=4, time_budget=40.0, n_threads=4):
    """Parallel basin hopping from multiple diverse starting points."""
    start_time = time.time()
    
    # Generate diverse initial templates
    templates = []
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.1))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.05))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.15))
    templates.append(_generate_compact_random(n, seed=42))
    templates.append(_generate_compact_random(n, seed=123))
    templates.append(_generate_compact_random(n, seed=456))
    
    # Add more templates if we have room
    for i in range(3):
        templates.append(_generate_compact_random(n, seed=1000 + i * 777))
    
    # Use only as many as we need
    templates = templates[:max(n_chains, 4)]
    
    # Run parallel basin hopping
    best_energy = float('inf')
    best_pos = None
    
    def run_chain(args):
        chain_idx, template = args
        rng = np.random.default_rng(n * 100 + chain_idx)
        return _basin_hopping_chain(n, template, rng, 
                                     max_steps=15, 
                                     time_budget=time_budget * 0.8,
                                     perturb_scale=0.12)
    
    # Run in parallel with thread pool
    with ThreadPoolExecutor(max_workers=min(n_threads, n_chains)) as executor:
        futures = []
        for i, template in enumerate(templates[:n_chains]):
            if time.time() - start_time > time_budget * 0.7:
                break
            futures.append(executor.submit(run_chain, (i, template)))
        
        for future in as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except Exception:
                continue
    
    if best_pos is None:
        # Fallback: relax first template
        e, pos = _local_minimize_l_bfgs(templates[0], timeout=10.0)
        best_energy = e
        best_pos = pos
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster via parallel basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check remaining time
    remaining = _remaining()
    if remaining < 15:
        # Emergency fallback
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            energy, pos = _local_minimize_l_bfgs(seed, timeout=8.0)
            return pos
        except Exception:
            return _generate_mackay_icosahedron(size, base_spacing=1.1)
    
    # Allocate time budget for basin hopping
    bh_budget = min(remaining * 0.7, 45.0)
    
    # Run parallel basin hopping
    try:
        best_energy, best_pos = _parallel_basin_hopping(size, 
                                                         n_chains=4,
                                                         time_budget=bh_budget,
                                                         n_threads=4)
    except Exception:
        # Fallback to simple seed
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy, best_pos = _local_minimize_l_bfgs(seed, timeout=10.0)
        except Exception:
            best_pos = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy = _dzugutov_energy(best_pos)
    
    # Final refinement: perturb and relax around best
    remaining2 = _remaining()
    if remaining2 > 20:
        rng2 = np.random.default_rng(42)
        for _ in range(3):
            if _remaining() < 12:
                break
            perturbed = best_pos + rng2.normal(0, 0.06, best_pos.shape)
            try:
                energy, positions = _local_minimize_l_bfgs(perturbed, timeout=8.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions
            except Exception:
                continue
    
    return best_pos
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
