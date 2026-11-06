<<<<<<< SEARCH
import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 115s to be safe
    deadline = time.time() + 115
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    moduli = list(range(min_m, max_m + 1))
    
    # Helper to check validity of (m, r) against a set of (m, r) pairs
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(m, selected):
        valid_rs = []
        for r in range(m):
            if is_valid(m, r, selected):
                valid_rs.append(r)
        return valid_rs

    def calculate_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_construct(randomize=False):
        selected = []
        # Process moduli in ascending order (highest weight first)
        for m in moduli:
            valid_rs = get_valid_residues(m, selected)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    # Heuristic: pick r that minimizes conflicts with next 5 moduli
                    best_r = valid_rs[0]
                    min_conflicts = float('inf')
                    for r in valid_rs:
                        conflicts = 0
                        next_moduli = [mm for mm in moduli if mm > m][:5]
                        for mm in next_moduli:
                            g = gcd_table[m][mm]
                            conflicts += mm // g
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            best_r = r
                    r = best_r
                selected.append((m, r))
        return selected

    def local_search(initial_selected):
        selected = [tuple(p) for p in initial_selected]
        current_score = calculate_score(selected)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any unselected modulus
            for m in moduli:
                if m in [s[0] for s in selected]:
                    continue
                valid_rs = get_valid_residues(m, selected)
                if valid_rs:
                    new_selected = selected + [(m, valid_rs[0])]
                    new_score = current_score + 1.0 / m
                    if new_score > current_score:
                        selected = new_selected
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 2. Try to swap a selected modulus with a smaller unselected one
            selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
            
            swap_found = False
            for m_old, r_old in selected_sorted:
                for m_new in moduli:
                    if m_new >= m_old:
                        break
                    if m_new in [s[0] for s in selected]:
                        continue
                    
                    temp_selected = [s for s in selected if s[0] != m_old]
                    valid_rs = get_valid_residues(m_new, temp_selected)
                    if valid_rs:
                        new_score = current_score - 1.0 / m_old + 1.0 / m_new
                        if new_score > current_score:
                            selected = temp_selected + [(m_new, valid_rs[0])]
                            current_score = new_score
                            improved = True
                            swap_found = True
                            break
                if swap_found:
                    break
            
            if improved:
                continue
                
            # 3. Random kick
            if time.time() < deadline - 10:
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected = [s for s in selected if s != removed]
                    current_score = calculate_score(selected)
                    improved = True

        return selected, current_score

    def try_fill_all_randomized():
        # Inspiration 1: Try to find a valid residue assignment for ALL moduli
        # This is the optimal solution if feasible.
        # Use randomized greedy with restarts.
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
                
            sol = {}
            success = True
            
            # Iterate from smallest to largest m (most constrained first)
            for m in range(min_m, max_m + 1):
                # Generate random permutation of residues
                residues = list(range(m))
                random.shuffle(residues)
                
                assigned = False
                for r in residues:
                    valid = True
                    for m2, r2 in sol.items():
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            valid = False
                            break
                            
                    if valid:
                        sol[m] = r
                        assigned = True
                        break
                        
                if not assigned:
                    success = False
                    break
            
            if success:
                return [[m, r] for m, r in sol.items()]
                
        return None

    best_solution = []
    best_score = 0.0
    
    # Phase 1: Try to get the full set (optimal if feasible)
    full_sol = try_fill_all_randomized()
    if full_sol:
        # Verify and score
        # It's guaranteed valid by construction, but let's double check score
        score = calculate_score(full_sol)
        if score > best_score:
            best_score = score
            best_solution = full_sol
            
    # Phase 2: If full set not found or we want to improve (though full set is max possible score sum 1/m),
    # we run the Greedy + Local Search approach which might find a high-scoring subset 
    # if the full set is infeasible, or confirm the full set.
    # Actually, if full set is feasible, it IS the optimal solution because all weights are positive.
    # So if we found it, we are done.
    # If we didn't find it, it might be infeasible, or just hard to find.
    # The greedy+local search approach finds the best subset.
    
    if not best_solution:
        # Run multiple iterations of Greedy + Local Search
        num_restarts = 50
        
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            is_rand = (i % 2 == 1)
            initial = greedy_construct(randomize=is_rand)
            
            sol, score = local_search(initial)
            
            if score > best_score:
                best_score = score
                best_solution = sol

    # Convert to required format
    if best_solution:
        return [[m, r] for m, r in best_solution]
    
    # Fallback: simple greedy
    initial = greedy_construct(randomize=False)
    return [[m, r] for m, r in initial]
=======
import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 115s to be safe
    deadline = time.time() + 115
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    moduli = list(range(min_m, max_m + 1))
    n_moduli = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}
    
    # Precompute valid residues for each modulus
    # valid_residues[m] = list of all residues 0..m-1
    all_residues = {m: list(range(m)) for m in moduli}
    
    # Precompute conflict matrix: conflict[i][j][r_i][r_j] = True if conflict
    # This is too large, so we'll compute on the fly but cache gcd results
    
    def check_conflict(m1, r1, m2, r2):
        g = gcd_table[m1][m2]
        return (r1 % g) == (r2 % g)
    
    def count_conflicts(state, m_idx, r):
        """Count how many conflicts (m_idx, r) has with current state"""
        m = moduli[m_idx]
        conflicts = 0
        for j in range(n_moduli):
            if j == m_idx or state[j] is None:
                continue
            m2 = moduli[j]
            r2 = state[j]
            if check_conflict(m, r, m2, r2):
                conflicts += 1
        return conflicts
    
    def get_valid_residues_for(m_idx, state):
        """Get all valid residues for modulus m_idx given current state"""
        m = moduli[m_idx]
        valid = []
        for r in all_residues[m]:
            if count_conflicts(state, m_idx, r) == 0:
                valid.append(r)
        return valid
    
    def calculate_score(state):
        score = 0.0
        for i in range(n_moduli):
            if state[i] is not None:
                score += 1.0 / moduli[i]
        return score
    
    def is_valid_state(state):
        """Check if state is fully valid (no conflicts among assigned moduli)"""
        for i in range(n_moduli):
            if state[i] is None:
                continue
            for j in range(i + 1, n_moduli):
                if state[j] is None:
                    continue
                if check_conflict(moduli[i], state[i], moduli[j], state[j]):
                    return False
        return True
    
    def simulated_annealing():
        """Simulated annealing to find high-scoring valid assignment"""
        # Initialize: greedy construction
        state = [None] * n_moduli
        
        # Greedy init: process moduli in order, pick random valid residue
        random.shuffle(moduli)
        for m in moduli:
            m_idx = mod_index[m]
            valid_rs = get_valid_residues_for(m_idx, state)
            if valid_rs:
                state[m_idx] = random.choice(valid_rs)
        
        current_score = calculate_score(state)
        best_state = state[:]
        best_score = current_score
        
        # Annealing parameters
        T_start = 1.0
        T_end = 0.001
        cooling_rate = 0.9995
        T = T_start
        
        while T > T_end and time.time() < deadline - 5:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Change residue of an assigned modulus
                assigned = [i for i in range(n_moduli) if state[i] is not None]
                if assigned:
                    idx = random.choice(assigned)
                    old_r = state[idx]
                    new_r = random.randint(0, moduli[idx] - 1)
                    if new_r != old_r:
                        state[idx] = new_r
                        new_score = calculate_score(state)
                        delta = new_score - current_score
                        # Also check if we introduced conflicts
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = state[:]
                        else:
                            state[idx] = old_r
                            
            elif move_type < 0.6:
                # Add an unassigned modulus
                unassigned = [i for i in range(n_moduli) if state[i] is None]
                if unassigned:
                    idx = random.choice(unassigned)
                    valid_rs = get_valid_residues_for(idx, state)
                    if valid_rs:
                        r = random.choice(valid_rs)
                        state[idx] = r
                        new_score = current_score + 1.0 / moduli[idx]
                        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = state[:]
                        else:
                            state[idx] = None
                            
            elif move_type < 0.8:
                # Remove an assigned modulus
                assigned = [i for i in range(n_moduli) if state[i] is not None]
                if assigned:
                    idx = random.choice(assigned)
                    old_r = state[idx]
                    state[idx] = None
                    new_score = current_score - 1.0 / moduli[idx]
                    if random.random() < math.exp((new_score - current_score) / T):
                        current_score = new_score
                    else:
                        state[idx] = old_r
                        
            else:
                # Swap: remove one, add another
                assigned = [i for i in range(n_moduli) if state[i] is not None]
                unassigned = [i for i in range(n_moduli) if state[i] is None]
                if assigned and unassigned:
                    remove_idx = random.choice(assigned)
                    add_idx = random.choice(unassigned)
                    old_r = state[remove_idx]
                    state[remove_idx] = None
                    valid_rs = get_valid_residues_for(add_idx, state)
                    if valid_rs:
                        r = random.choice(valid_rs)
                        state[add_idx] = r
                        new_score = current_score - 1.0 / moduli[remove_idx] + 1.0 / moduli[add_idx]
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = state[:]
                        else:
                            state[add_idx] = None
                            state[remove_idx] = old_r
                            
            T *= cooling_rate
        
        return best_state, best_score
    
    def greedy_with_backtracking():
        """Greedy with limited backtracking"""
        state = [None] * n_moduli
        current_score = 0.0
        
        # Process moduli in order of weight (1/m), so smallest m first
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            valid_rs = get_valid_residues_for(m_idx, state)
            if valid_rs:
                # Pick the residue that minimizes future conflicts
                best_r = valid_rs[0]
                min_future_conflicts = float('inf')
                for r in valid_rs:
                    # Count how many future moduli would be blocked
                    future_conflicts = 0
                    for j in range(m_idx + 1, n_moduli):
                        if state[j] is not None:
                            continue
                        g = gcd_table[m][moduli[j]]
                        # How many residues of moduli[j] are blocked?
                        blocked = 0
                        for r2 in all_residues[moduli[j]]:
                            if (r % g) == (r2 % g):
                                blocked += 1
                        future_conflicts += blocked
                    if future_conflicts < min_future_conflicts:
                        min_future_conflicts = future_conflicts
                        best_r = r
                state[m_idx] = best_r
                current_score += 1.0 / m
        return state, current_score
    
    best_solution = []
    best_score = 0.0
    
    # Phase 1: Greedy with backtracking
    state, score = greedy_with_backtracking()
    if score > best_score:
        best_score = score
        best_solution = state[:]
    
    # Phase 2: Simulated annealing from multiple starts
    num_sa_runs = 10
    for i in range(num_sa_runs):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if i % 3 == 0:
            # Greedy init
            state, score = greedy_with_backtracking()
        elif i % 3 == 1:
            # Random init
            state = [None] * n_moduli
            for m_idx in range(n_moduli):
                if random.random() < 0.8:
                    r = random.randint(0, moduli[m_idx] - 1)
                    state[m_idx] = r
            score = calculate_score(state)
        else:
            # Sparse random init
            state = [None] * n_moduli
            for m_idx in range(n_moduli):
                if random.random() < 0.5:
                    r = random.randint(0, moduli[m_idx] - 1)
                    state[m_idx] = r
            score = calculate_score(state)
        
        sa_state, sa_score = simulated_annealing()
        
        if sa_score > best_score:
            best_score = sa_score
            best_solution = sa_state[:]
    
    # Phase 3: Local search refinement on best solution
    if best_solution and time.time() < deadline - 5:
        state = best_solution[:]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try to add any unassigned modulus
            for m_idx in range(n_moduli):
                if state[m_idx] is not None:
                    continue
                valid_rs = get_valid_residues_for(m_idx, state)
                if valid_rs:
                    r = valid_rs[0]
                    state[m_idx] = r
                    new_score = current_score + 1.0 / moduli[m_idx]
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try to improve residue assignments
            for m_idx in range(n_moduli):
                if state[m_idx] is None:
                    continue
                current_r = state[m_idx]
                # Try all residues
                best_r = current_r
                best_conflicts = count_conflicts(state, m_idx, current_r)
                for r in all_residues[moduli[m_idx]]:
                    if r == current_r:
                        continue
                    c = count_conflicts(state, m_idx, r)
                    if c < best_conflicts:
                        best_conflicts = c
                        best_r = r
                if best_r != current_r:
                    state[m_idx] = best_r
                    improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_solution = state[:]
    
    # Convert to required format
    result = []
    for i in range(n_moduli):
        if best_solution[i] is not None:
            result.append([moduli[i], best_solution[i]])
    
    return result
>>>>>>> REPLACE

