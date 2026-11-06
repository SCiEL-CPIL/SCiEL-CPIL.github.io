<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: consecutive exponents with all 1s
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
    
    # Simulated annealing with structural moves
    random.seed(42)
    np.random.seed(42)
    
    def _evaluate(poly):
        return _square_nonzero_count_fast(poly)
    
    def _sa_optimize(initial_poly, max_iters=50000):
        """Simulated annealing to minimize nonzero terms in square."""
        current = list(initial_poly)
        current_score = _evaluate(current)
        best = list(current)
        best_score = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.05
        T = T_start
        
        # Coefficient choices - more diverse
        coeff_choices = [-12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Cooled temperature
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = list(current)
            
            if move_type < 0.45:
                # Change one exponent - try nearby values first for local search
                i = random.randint(0, 11)
                old_exp = neighbor[i][0]
                if random.random() < 0.6:
                    # Local move: try nearby exponent
                    offset = random.choice([-3, -2, -1, 1, 2, 3, -5, 5])
                    new_exp = old_exp + offset
                    if new_exp < 0 or new_exp > 40:
                        continue
                else:
                    new_exp = random.randint(0, 40)
                if new_exp == old_exp:
                    continue
                # Check for collision
                if any(neighbor[j][0] == new_exp for j in range(12) if j != i):
                    continue
                neighbor[i] = (new_exp, neighbor[i][1])
            elif move_type < 0.75:
                # Change one coefficient
                i = random.randint(0, 11)
                new_coeff = random.choice(coeff_choices)
                if new_coeff == 0 or new_coeff == neighbor[i][1]:
                    continue
                neighbor[i] = (neighbor[i][0], new_coeff)
            elif move_type < 0.90:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = neighbor[i]
                ej, cj = neighbor[j]
                neighbor[i] = (ej, ci)
                neighbor[j] = (ei, cj)
            else:
                # Perturb two coefficients simultaneously
                i, j = random.sample(range(12), 2)
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci == 0 or new_cj == 0:
                    continue
                neighbor[i] = (neighbor[i][0], new_ci)
                neighbor[j] = (neighbor[j][0], new_cj)
            
            if not _validate(neighbor):
                continue
            
            neighbor_score = _evaluate(neighbor)
            delta = neighbor_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            
            if best_score <= 12:
                break
        
        return best, best_score
    
    # Multiple random restarts with simulated annealing
    num_restarts = 45
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial polynomials
        if r == 0:
            # Start from seed
            initial = list(seed_poly)
        elif r == 1:
            # Arithmetic progression with signs
            step = random.randint(1, 3)
            start = random.randint(0, 40 - 11 * step)
            signs = [random.choice([-1, 1]) for _ in range(12)]
            initial = [(start + i * step, signs[i]) for i in range(12)]
        elif r == 2:
            # Two clusters
            exps = sorted(random.sample(range(0, 41), 12))
            # Make first 6 close together, last 6 close together
            initial = [(exps[i], random.choice([-1, 1])) for i in range(12)]
        elif r == 3:
            # Structured: pairs with same coefficient magnitude
            base_exps = sorted(random.sample(range(0, 41), 6))
            offsets = [random.randint(1, 5) for _ in range(6)]
            exps = []
            for be in base_exps:
                exps.append(be)
                exps.append(be + offsets[base_exps.index(be)])
            exps = sorted(set(exps))
            while len(exps) < 12:
                exps.append(random.randint(0, 40))
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = []
            for i in range(6):
                c = random.choice([-2, -1, 1, 2])
                coeffs.extend([c, c])
            random.shuffle(coeffs)
            initial = list(zip(exps, coeffs))
        elif r == 4:
            # Palindromic structure
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(6)]
            coeffs = coeffs + coeffs[::-1]
            initial = list(zip(exps, coeffs))
        else:
            # Random
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            initial = list(zip(exps, coeffs))
        
        if not _validate(initial):
            continue
        
        score = _evaluate(initial)
        if score < best_score:
            best_score = score
            best_poly = list(initial)
        
        # Run simulated annealing - allocate more iterations when time permits
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        max_iters = min(50000, int(remaining * 3500))
        local_poly, local_score = _sa_optimize(initial, max_iters=max(1000, max_iters))
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly
        
        if best_score <= 12:
            break
    
    # Final local improvement on best found
    if best_poly is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 3:
            final_poly, final_score = _sa_optimize(best_poly, max_iters=int(remaining * 5000))
            if final_score < best_score:
                best_score = final_score
                best_poly = final_poly
    
    # Convert to coefficient list
    if best_poly is None:
        best_poly = seed_poly
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
=======
def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: consecutive exponents with all 1s
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count_fast(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
    
    # Simulated annealing with structural moves
    random.seed(42)
    np.random.seed(42)
    
    def _evaluate(poly):
        return _square_nonzero_count_fast(poly)
    
    def _sa_optimize(initial_poly, max_iters=50000):
        """Simulated annealing to minimize nonzero terms in square."""
        current = list(initial_poly)
        current_score = _evaluate(current)
        best = list(current)
        best_score = current_score
        
        # Temperature schedule
        T_start = 8.0
        T_end = 0.01
        T = T_start
        
        # Coefficient choices - optimized for cancellation
        coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4, -6, -5, 5, 6, -8, 8]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Cooled temperature
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = list(current)
            
            if move_type < 0.40:
                # Change one exponent - try nearby values first for local search
                i = random.randint(0, 11)
                old_exp = neighbor[i][0]
                if random.random() < 0.65:
                    # Local move: try nearby exponent
                    offset = random.choice([-2, -1, 1, 2, -3, 3, -4, 4, -6, 6])
                    new_exp = old_exp + offset
                    if new_exp < 0 or new_exp > 40:
                        continue
                else:
                    new_exp = random.randint(0, 40)
                if new_exp == old_exp:
                    continue
                # Check for collision
                if any(neighbor[j][0] == new_exp for j in range(12) if j != i):
                    continue
                neighbor[i] = (new_exp, neighbor[i][1])
            elif move_type < 0.70:
                # Change one coefficient
                i = random.randint(0, 11)
                new_coeff = random.choice(coeff_choices)
                if new_coeff == 0 or new_coeff == neighbor[i][1]:
                    continue
                neighbor[i] = (neighbor[i][0], new_coeff)
            elif move_type < 0.85:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = neighbor[i]
                ej, cj = neighbor[j]
                neighbor[i] = (ej, ci)
                neighbor[j] = (ei, cj)
            elif move_type < 0.93:
                # Perturb two coefficients simultaneously
                i, j = random.sample(range(12), 2)
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci == 0 or new_cj == 0:
                    continue
                neighbor[i] = (neighbor[i][0], new_ci)
                neighbor[j] = (neighbor[j][0], new_cj)
            else:
                # Change two exponents simultaneously
                i, j = random.sample(range(12), 2)
                new_ei = random.randint(0, 40)
                new_ej = random.randint(0, 40)
                # Check collisions
                if new_ei == new_ej:
                    continue
                if any(neighbor[k][0] == new_ei for k in range(12) if k != i and k != j):
                    continue
                if any(neighbor[k][0] == new_ej for k in range(12) if k != i and k != j):
                    continue
                if new_ei == neighbor[j][0] or new_ej == neighbor[i][0]:
                    continue
                neighbor[i] = (new_ei, neighbor[i][1])
                neighbor[j] = (new_ej, neighbor[j][1])
            
            if not _validate(neighbor):
                continue
            
            neighbor_score = _evaluate(neighbor)
            delta = neighbor_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            
            if best_score <= 12:
                break
        
        return best, best_score
    
    def _generate_factor_based_initial():
        """Generate initial polynomial as product structure for cancellation."""
        # Try to construct P = A * B where A and B have 6 terms each
        # with exponent supports that create overlapping sums in A*A, B*B, A*B
        for attempt in range(50):
            # Choose two sets of 6 exponents
            set_a = sorted(random.sample(range(0, 41), 6))
            set_b = sorted(random.sample(range(0, 41), 6))
            
            # Ensure all 12 exponents are distinct
            if len(set(set_a) & set(set_b)) > 0:
                continue
            all_exps = set_a + set_b
            if len(set(all_exps)) != 12:
                continue
            
            # Assign coefficients to encourage cancellation
            # Strategy: use pairs with opposite signs at related positions
            coeffs_a = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
            coeffs_b = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
            
            # Try to make some cross-terms cancel
            # If a_i * b_j and a_k * b_l have same sum, try opposite signs
            poly = [(e, c) for e, c in zip(set_a, coeffs_a)] + \
                   [(e, c) for e, c in zip(set_b, coeffs_b)]
            
            if not _validate(poly):
                continue
            
            score = _evaluate(poly)
            if score < 20:
                return poly
        
        return None
    
    # Multiple random restarts with simulated annealing
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial polynomials
        if r == 0:
            # Start from seed
            initial = list(seed_poly)
        elif r == 1:
            # Arithmetic progression with alternating signs
            step = random.randint(1, 3)
            start = random.randint(0, 40 - 11 * step)
            signs = [(-1)**i for i in range(12)]
            initial = [(start + i * step, signs[i]) for i in range(12)]
        elif r == 2:
            # Factor-based initialization
            fb = _generate_factor_based_initial()
            if fb is not None:
                initial = fb
            else:
                exps = sorted(random.sample(range(0, 41), 12))
                initial = [(exps[i], random.choice([-1, 1])) for i in range(12)]
        elif r == 3:
            # Structured: pairs with same coefficient magnitude, opposite signs
            base_exps = sorted(random.sample(range(0, 41), 6))
            offsets = [random.randint(1, 4) for _ in range(6)]
            exps = []
            coeffs = []
            for be, off in zip(base_exps, offsets):
                exps.append(be)
                exps.append(be + off)
                c = random.choice([1, 2, 3])
                coeffs.extend([c, -c])
            exps = sorted(set(exps))
            if len(exps) < 12:
                initial = None
            else:
                # Rebuild properly
                pairs = [(be, off) for be, off in zip(base_exps, offsets)]
                all_exps = []
                all_coeffs = []
                for be, off in pairs:
                    all_exps.append(be)
                    all_exps.append(be + off)
                    c = random.choice([1, 2, 3])
                    all_coeffs.extend([c, -c])
                if len(set(all_exps)) == 12 and all(0 <= e <= 40 for e in all_exps):
                    initial = list(zip(all_exps, all_coeffs))
                    random.shuffle(initial)
                else:
                    initial = None
            if initial is None:
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                initial = list(zip(exps, coeffs))
        elif r == 4:
            # Palindromic with sign flips
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(6)]
            coeffs = coeffs + [-c for c in coeffs[::-1]]
            initial = list(zip(exps, coeffs))
        elif r == 5:
            # Two interleaved arithmetic progressions
            step1 = random.randint(1, 2)
            step2 = random.randint(1, 2)
            start1 = random.randint(0, 30)
            start2 = random.randint(0, 30)
            exps1 = [start1 + i * step1 for i in range(6)]
            exps2 = [start2 + i * step2 for i in range(6)]
            all_exps = exps1 + exps2
            if len(set(all_exps)) == 12 and all(0 <= e <= 40 for e in all_exps):
                coeffs1 = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                coeffs2 = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                initial = list(zip(all_exps, coeffs1 + coeffs2))
            else:
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
                initial = list(zip(exps, coeffs))
        else:
            # Random with bias toward small coefficients
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-2, -1, -1, 1, 1, 2]) for _ in range(12)]
            initial = list(zip(exps, coeffs))
        
        if not _validate(initial):
            continue
        
        score = _evaluate(initial)
        if score < best_score:
            best_score = score
            best_poly = list(initial)
        
        # Run simulated annealing - allocate more iterations when time permits
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        max_iters = min(60000, int(remaining * 4000))
        local_poly, local_score = _sa_optimize(initial, max_iters=max(1000, max_iters))
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly
        
        if best_score <= 12:
            break
    
    # Final local improvement on best found
    if best_poly is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 3:
            final_poly, final_score = _sa_optimize(best_poly, max_iters=int(remaining * 5000))
            if final_score < best_score:
                best_score = final_score
                best_poly = final_poly
    
    # Convert to coefficient list
    if best_poly is None:
        best_poly = seed_poly
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
>>>>>>> REPLACE
