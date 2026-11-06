# EVOLVE-BLOCK-START
def construct():
    """
    Find pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Two classes (m1, r1) and (m2, r2) are disjoint iff r1 != r2 mod gcd(m1, m2).
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    rng = random.Random(42)
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute gcd table indexed by modulus value
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def is_valid(m, r, selected):
        """Check if (m, r) is valid against a list of selected (m, r) pairs."""
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
    
    def calc_score(selected):
        score = Fraction(0)
        for m, r in selected:
            score += Fraction(1, m)
        return score
        
    def polish_solution(selected):
        """
        Deterministic polish:
        1. Add any missing moduli if a valid residue exists.
        2. Swap larger moduli for smaller ones if removing the larger unblocks the smaller.
        """
        selected = [list(x) for x in selected]
        selected_mods = set(m for m, _ in selected)
        
        # Step 1: Greedy Add
        for m in range(min_m, max_m + 1):
            if m in selected_mods:
                continue
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    selected_mods.add(m)
                    break
        
        # Step 2: Iterative Swap (Large for Small)
        # We iterate until no improvement is possible
        improved = True
        while improved:
            improved = False
            # Iterate over current selected moduli
            # We want to remove large m to add small m
            # So sort by m descending
            selected_sorted = sorted(selected, key=lambda x: -x[0])
            
            for i in range(len(selected_sorted)):
                m_large, r_large = selected_sorted[i]
                # Temporarily remove m_large
                temp_selected = [x for j, x in enumerate(selected_sorted) if j != i]
                temp_mods = set(m for m, _ in temp_selected)
                
                # Try to add any m_small < m_large that is not in temp_mods
                # We only need to check if it fits. If it fits, we swap.
                # Since m_small < m_large, 1/m_small > 1/m_large, so score increases.
                found_swap = False
                for m_small in range(min_m, m_large):
                    if m_small in temp_mods:
                        continue
                    # Check if m_small can be added to temp_selected
                    for r_small in range(m_small):
                        if is_valid(m_small, r_small, temp_selected):
                            # Perform swap
                            selected_sorted[i] = [m_small, r_small]
                            # Reconstruct selected list
                            selected = [list(x) for x in selected_sorted]
                            selected_mods = set(m for m, _ in selected)
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if improved:
                    break # Restart the while loop
            
        return selected

    def greedy_construct(strategy='first', rng_seed=None):
        """Greedy: for each m in increasing order, pick a valid residue."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = rng
        
        selected = []
        for m in range(min_m, max_m + 1):
            if strategy == 'first':
                for r in range(m):
                    if is_valid(m, r, selected):
                        selected.append([m, r])
                        break
            elif strategy == 'random':
                valid_residues = [r for r in range(m) if is_valid(m, r, selected)]
                if valid_residues:
                    r = local_rng.choice(valid_residues)
                    selected.append([m, r])
            elif strategy == 'min_conflict':
                # Choose residue that minimizes future conflicts
                best_r = None
                best_conflicts = float('inf')
                for r in range(m):
                    if not is_valid(m, r, selected):
                        continue
                    conflicts = 0
                    for future_m in range(m + 1, max_m + 1):
                        g = gcd_table[m][future_m]
                        blocked = future_m // g
                        conflicts += blocked
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_r = r
                if best_r is not None:
                    selected.append([m, best_r])
        return selected
    
    def simulated_annealing(initial_solution, time_budget):
        """Simulated Annealing to improve the solution."""
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        
        start_time = time.time()
        
        # Precompute set of selected moduli for quick lookup
        selected_mods = set(m for m, _ in current_sol)
        
        while time.time() - start_time < time_budget and T > min_T:
            # Generate a neighbor
            move_type = rng.random()
            
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False
            
            if move_type < 0.5:
                # Add a modulus
                m_add = rng.randint(min_m, max_m)
                if m_add in selected_mods:
                    continue
                
                found_r = False
                for _ in range(10):
                    r_add = rng.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += Fraction(1, m_add)
                        valid_move = True
                        found_r = True
                        break
                
                if not found_r:
                    continue
                    
            elif move_type < 0.7:
                # Remove a modulus
                if not new_sol:
                    continue
                idx = rng.randint(0, len(new_sol) - 1)
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= Fraction(1, m_rem)
                valid_move = True
                
            else:
                # Change residue
                if not new_sol:
                    continue
                idx = rng.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                
                found_r = False
                for _ in range(10):
                    new_r = rng.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    temp_sol = [x for i, x in enumerate(new_sol) if i != idx]
                    if is_valid(m_chg, new_r, temp_sol):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        found_r = True
                        break
                
                if not found_r:
                    continue
            
            if not valid_move:
                continue
                
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta.numerator / (delta.denominator * T)):
                current_sol = new_sol
                current_score = new_score
                
                selected_mods = set(m for m, _ in current_sol)
                
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]
            
            T *= cooling
            
        return best_sol, best_score
    
    # Phase 1: Multiple greedy starts with different strategies
    best_selected = None
    best_score = Fraction(0)
    
    strategies = ['first', 'random', 'min_conflict']
    for strategy in strategies:
        selected = greedy_construct(strategy)
        # Polish the greedy result immediately
        polished = polish_solution(selected)
        score = calc_score(polished)
        if score > best_score:
            best_score = score
            best_selected = polished
    
    # More random restarts
    start_time = time.time()
    for attempt in range(20):
        if time.time() - start_time > 10:
            break
        selected = greedy_construct('random', rng_seed=attempt)
        polished = polish_solution(selected)
        score = calc_score(polished)
        if score > best_score:
            best_score = score
            best_selected = polished
    
    # Phase 2: Simulated Annealing on best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_selected, best_score = simulated_annealing(best_selected, min(remaining_time - 2, 40))
        # Polish the SA result
        best_selected = polish_solution(best_selected)
        best_score = calc_score(best_selected)
    
    # Phase 3: More local search iterations with different starting points
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            selected = greedy_construct('random', rng_seed=100 + attempt)
            polished = polish_solution(selected)
            score = calc_score(polished)
            if score > best_score - 0.5:
                budget = min(deadline - time.time() - 3, 20)
                if budget > 2:
                    new_selected, new_score = simulated_annealing(polished, budget)
                    new_selected = polish_solution(new_selected)
                    new_score = calc_score(new_selected)
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = new_selected
    
    # Convert to output format
    result = [[m, r] for m, r in best_selected]
    return result
# EVOLVE-BLOCK-END
