<<<<<<< SEARCH
def construct(instance):
    """Construct low-energy cluster using defect-seeded Mackay icosahedron initialization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Precompute Mackay structures
    mackay_55_pos, mackay_55_labels = _build_mackay_55_with_labels(r_inner=1.0, r_outer=1.9)
    mackay_13_pos = _build_mackay_13(r=1.0)
    
    # Generate multiple diverse seeds
    n_seeds = 15
    seeds = []
    
    for seed_idx in range(n_seeds):
        # Vary the icosahedron parameters
        r_inner = 0.95 + 0.05 * (seed_idx % 3)
        r_outer = 1.85 + 0.05 * (seed_idx % 3)
        
        base_pos, base_labels = _build_mackay_55_with_labels(r_inner=r_inner, r_outer=r_outer)
        
        if size == 55:
            candidate = base_pos
        elif size < 55:
            n_remove = 55 - size
            # Different removal strategies for different seeds
            if seed_idx % 3 == 0:
                candidate = _remove_atoms_by_priority(base_pos, base_labels, n_remove, 'outer_first', rng)
            elif seed_idx % 3 == 1:
                candidate = _remove_atoms_by_priority(base_pos, base_labels, n_remove, 'random', rng)
            else:
                # Remove specific high-symmetry atoms first
                # Remove face centers first, then edge midpoints, then vertices
                candidate = _remove_atoms_by_priority(base_pos, base_labels, n_remove, 'outer_first', rng)
        else:  # size > 55
            n_add = size - 55
            candidate = _add_atoms_to_mackay(base_pos, n_add, rng, base_r_outer=r_outer)
        
        if len(candidate) != size:
            # Try to fix by adding/removing
            if len(candidate) < size:
                candidate = _add_atoms_to_mackay(candidate, size - len(candidate), rng, base_r_outer=r_outer)
            elif len(candidate) > size:
                # Remove excess
                center = np.mean(candidate, axis=0)
                dists = np.linalg.norm(candidate - center, axis=1)
                order = np.argsort(dists)[::-1]
                to_remove = set(order[:len(candidate) - size].tolist())
                candidate = np.array([candidate[i] for i in range(len(candidate)) if i not in to_remove])
        
        if len(candidate) == size:
            seeds.append(candidate)
    
    # Add some 13-atom-based seeds for smaller clusters
    if size <= 55:
        for seed_idx in range(5):
            r = 0.95 + 0.05 * (seed_idx % 3)
            base_13 = _build_mackay_13(r=r)
            
            if size == 13:
                seeds.append(base_13)
            elif size < 13:
                # Remove from 13
                n_remove = 13 - size
                # Remove from outer shell
                center = np.mean(base_13, axis=0)
                dists = np.linalg.norm(base_13 - center, axis=1)
                order = np.argsort(dists)[::-1]
                to_remove = set(order[:n_remove].tolist())
                candidate = np.array([base_13[i] for i in range(13) if i not in to_remove])
                if len(candidate) == size:
                    seeds.append(candidate)
            else:  # 13 < size <= 55
                n_add = size - 13
                candidate = _add_atoms_to_mackay(base_13, n_add, rng, base_r_outer=1.9)
                if len(candidate) == size:
                    seeds.append(candidate)
    
    # Add some random compact seeds as fallback
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.75
    for _ in range(5):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        if len(points) == size:
            seeds.append(np.array(points))
    
    # Relax all seeds and keep the best
    for candidate in seeds:
        if len(candidate) != size:
            continue
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Second pass: try gradient-based refinement on the best result
    if best_positions is not None:
        try:
            e2, pos2 = _relax_with_gradient(best_positions, potential, parameter, max_iter=500)
            if e2 < best_energy:
                best_energy = e2
                best_positions = pos2
        except:
            pass
    
    # Third pass: try local search with small perturbations on the best result
    if best_positions is not None:
        for _ in range(10):
            perturbed = best_positions + rng.normal(0, 0.1, size=best_positions.shape)
            try:
                e3, pos3 = relax(perturbed, potential, parameter)
                if e3 < best_energy:
                    best_energy = e3
                    best_positions = pos3
            except:
                continue
    
    if best_positions is None:
        # Fallback: simple grid
        points = []
        for i in range(size):
            p = np.array([i * 1.0, (i // 10) * 1.0, (i // 100) * 1.0])
            points.append(p)
        best_positions = np.array(points)
    
    return best_positions
=======
def construct(instance):
    """Construct low-energy cluster using defect-seeded Mackay icosahedron initialization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    seeds = []
    
    # Generate diverse seeds from Mackay icosahedron with different defect patterns
    n_mackay_seeds = 12
    
    for seed_idx in range(n_mackay_seeds):
        # Vary icosahedron parameters
        r_inner = 0.95 + 0.03 * (seed_idx % 4)
        r_outer = 1.85 + 0.03 * (seed_idx % 4)
        edge_sel = ['greedy', 'pentagon', 'greedy', 'greedy'][seed_idx % 4]
        
        base_pos, base_labels = _build_mackay_55_with_labels(r_inner=r_inner, r_outer=r_outer, edge_selection=edge_sel)
        
        if size == 55:
            candidate = base_pos
        elif size < 55:
            n_remove = 55 - size
            # Different defect strategies
            strategy = ['symmetric', 'random', 'symmetric', 'random'][seed_idx % 4]
            candidate = _apply_defects(base_pos, base_labels, n_remove, strategy, rng)
        else:  # size > 55
            n_add = size - 55
            candidate = _add_atoms_smart(base_pos, n_add, rng, base_r_outer=r_outer)
        
        if len(candidate) == size:
            seeds.append(candidate)
    
    # Add seeds from 13-atom icosahedron for smaller clusters
    if size <= 50:
        for seed_idx in range(6):
            r = 0.95 + 0.03 * (seed_idx % 3)
            base_13 = _build_mackay_13(r=r)
            
            if size == 13:
                seeds.append(base_13)
            elif size < 13:
                n_remove = 13 - size
                center = np.mean(base_13, axis=0)
                dists = np.linalg.norm(base_13 - center, axis=1)
                order = np.argsort(dists)[::-1]
                to_remove = set(order[:n_remove].tolist())
                candidate = np.array([base_13[i] for i in range(13) if i not in to_remove])
                if len(candidate) == size:
                    seeds.append(candidate)
            else:
                n_add = size - 13
                candidate = _add_atoms_smart(base_13, n_add, rng, base_r_outer=1.9)
                if len(candidate) == size:
                    seeds.append(candidate)
    
    # Add random compact seeds as fallback
    spacing = 1.0
    radius = spacing * size ** (1.0 / 3.0) * 0.8
    for _ in range(4):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        if len(points) == size:
            seeds.append(np.array(points))
    
    # Relax all seeds and keep the best
    for candidate in seeds:
        if len(candidate) != size:
            continue
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Multi-stage refinement on the best result
    if best_positions is not None:
        try:
            e2, pos2 = _multi_stage_relax(best_positions, potential, parameter, max_stages=2)
            if e2 < best_energy:
                best_energy = e2
                best_positions = pos2
        except:
            pass
        
        # Gradient refinement
        try:
            e3, pos3 = _gradient_refine(best_positions, potential, parameter, max_iter=150)
            if e3 < best_energy:
                best_energy = e3
                best_positions = pos3
        except:
            pass
        
        # Final perturbation search
        for _ in range(8):
            perturbed = best_positions + rng.normal(0, 0.08, size=best_positions.shape)
            try:
                e4, pos4 = relax(perturbed, potential, parameter)
                if e4 < best_energy:
                    best_energy = e4
                    best_positions = pos4
            except:
                continue
    
    if best_positions is None:
        # Fallback: simple grid
        points = []
        for i in range(size):
            p = np.array([i * 1.0, (i // 10) * 1.0, (i // 100) * 1.0])
            points.append(p)
        best_positions = np.array(points)
    
    return best_positions
>>>>>>> REPLACE
