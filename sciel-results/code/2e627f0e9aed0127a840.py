<<<<<<< SEARCH
def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Best result tracking
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Layer count patterns to try, with various label sequences
    # Pattern: list of (n_layers, label_sequence_template)
    # We'll generate label sequences with various patterns
    
    patterns = []
    
    # Alternating starting with L
    for n in [5, 7, 9, 11, 13, 15, 17, 19]:
        if n <= max_layers:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
            patterns.append((n, labels))
    
    # Alternating starting with H
    for n in [5, 7, 9, 11, 13, 15, 17, 19]:
        if n <= max_layers:
            labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
            patterns.append((n, labels))
    
    # L H H L repeating (period 4)
    for n in [8, 12, 16]:
        if n <= max_layers:
            labels = ['L' if i % 4 < 1 else ('H' if i % 4 < 3 else 'L') for i in range(n)]
            patterns.append((n, labels))
    
    # H L L H repeating (period 4)
    for n in [8, 12, 16]:
        if n <= max_layers:
            labels = ['H' if i % 4 < 1 else ('L' if i % 4 < 3 else 'H') for i in range(n)]
            patterns.append((n, labels))
    
    # Single layer
    patterns.append((1, ['L']))
    patterns.append((1, ['H']))
    
    # Two layers
    patterns.append((2, ['L', 'H']))
    patterns.append((2, ['H', 'L']))
    patterns.append((2, ['L', 'L']))
    patterns.append((2, ['H', 'H']))
    
    # Three layers
    patterns.append((3, ['L', 'H', 'L']))
    patterns.append((3, ['H', 'L', 'H']))
    patterns.append((3, ['L', 'L', 'H']))
    patterns.append((3, ['H', 'H', 'L']))
    
    # Sort patterns by expected difficulty (smaller first for quick wins, but prioritize mid-range)
    # Prioritize 7-13 layers which tend to do well for broadband
    def pattern_priority(p):
        n = p[0]
        if 7 <= n <= 13:
            return 0
        elif 5 <= n < 7 or 13 < n <= 17:
            return 1
        else:
            return 2
    
    patterns.sort(key=pattern_priority)
    
    # For each pattern, do multiple random starts + NM polish
    for n_layers, labels in patterns:
        if time.time() > deadline - 30:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Multiple random starts
        n_starts = 12
        for start_i in range(n_starts):
            if time.time() > deadline - 25:
                break
            
            # Generate initial guess
            if start_i == 0:
                # Quarter-wave at center
                center_wl = 0.5 * (band_nm[0] + band_nm[1])
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
            elif start_i == 1:
                # Quarter-wave at red end
                x0 = np.array([band_nm[1] / (4.0 * n) for n in indices])
            elif start_i == 2:
                # Quarter-wave at blue end
                x0 = np.array([band_nm[0] / (4.0 * n) for n in indices])
            elif start_i == 3:
                # Random but constrained
                x0 = np.random.uniform(0, max_layer_nm, size=n_layers)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            elif start_i == 4:
                # Small random
                x0 = np.random.uniform(0, 100, size=n_layers)
            elif start_i == 5:
                # Gradient-like: decreasing thicknesses
                x0 = np.linspace(max_layer_nm * 0.8, max_layer_nm * 0.2, n_layers)
            elif start_i == 6:
                # Gradient-like: increasing thicknesses
                x0 = np.linspace(max_layer_nm * 0.2, max_layer_nm * 0.8, n_layers)
            elif start_i == 7:
                # All same
                x0 = np.full(n_layers, max_total_nm / n_layers)
            elif start_i == 8:
                # Alternating large/small
                x0 = np.array([max_layer_nm * 0.6 if i % 2 == 0 else max_layer_nm * 0.2 for i in range(n_layers)])
            elif start_i == 9:
                # Alternating small/large
                x0 = np.array([max_layer_nm * 0.2 if i % 2 == 0 else max_layer_nm * 0.6 for i in range(n_layers)])
            elif start_i == 10:
                # Random with constraint
                x0 = np.random.uniform(0, max_layer_nm, size=n_layers)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            else:
                # Very small random
                x0 = np.random.uniform(0, 50, size=n_layers)
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Evaluate initial
            f0 = f(x0)
            
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
            
            if f0 >= 100.0:
                continue  # Constraint violated, skip NM
            
            # NM polish with time-limited budget
            time_per_start = max(0.5, min(5.0, (deadline - 20 - time.time()) / max(1, n_starts - start_i)))
            nm_deadline = time.time() + time_per_start
            
            try:
                x_opt, f_opt = _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-9, 
                                             time_deadline=nm_deadline)
            except Exception:
                x_opt, f_opt = x0, f0
            
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels.copy()
                best_thicknesses = x_opt.copy()
            
            # Early exit if very good
            if best_score < 0.3:
                break
    
    # Final fallback
    if best_labels is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
=======
def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Phase 1: Genetic algorithm to find good region in label+thickness space
    ga_deadline = start_time + 100.0  # Leave 40s for local polish
    best_labels, best_thicknesses, best_score = _genetic_algorithm(
        wavelengths_nm, ambient_idx, substrate_idx, materials,
        max_layers, max_total_nm, max_layer_nm, ga_deadline,
        pop_size=35, elite_size=5, mutation_rate=0.12
    )
    
    # Phase 2: Local polish with Nelder-Mead on the best GA result
    if best_labels is not None and time.time() < deadline - 5:
        indices = [materials[label] for label in best_labels]
        bounds = [(0.0, float(max_layer_nm))] * len(best_labels)
        
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        
        x0 = np.clip(best_thicknesses, 0, max_layer_nm)
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        
        try:
            x_opt, f_opt = _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-9,
                                         time_deadline=deadline - 3)
            if f_opt < best_score:
                best_score = f_opt
                best_thicknesses = x_opt
        except Exception:
            pass
    
    # Final fallback
    if best_labels is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
>>>>>>> REPLACE
