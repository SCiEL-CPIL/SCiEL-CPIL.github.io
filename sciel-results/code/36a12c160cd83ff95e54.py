import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
    
    # Precompute 1/m as floats for speed
    inv_m = [1.0 / m for m in moduli]
    
    def get_score(selected):
        return sum(1.0 / m for m, r in selected)
    
    def is_compatible_idx(idx_m, r, selected_idx_res):
        """Check if (idx_m, r) is compatible with all entries in selected_idx_res."""
        for idx_prime, r_prime in selected_idx_res:
            g = gcd_mat[idx_m][idx_prime]
            if r % g == r_prime % g:
                return False
        return True
        
    def greedy_fill(seed, priority_order=None):
        """
        Greedy fill with optional priority order.
        priority_order: list of indices in the order to try. If None, use natural order.
        """
        random.seed(seed)
        selected_idx_res = []
        selected_set = set()
        
        if priority_order is None:
            order = list(range(n_mods))
        else:
            order = priority_order
        
        for idx_m in order:
            m = moduli[idx_m]
            
            # Try residues in random order
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if is_compatible_idx(idx_m, r, selected_idx_res):
                    selected_idx_res.append((idx_m, r))
                    selected_set.add(idx_m)
                    found = True
                    break
            
            # If not found, skip this modulus
        
        # Convert back to (m, r)
        result = []
        for idx_m, r in selected_idx_res:
            result.append([moduli[idx_m], r])
        return result
    
    def simulated_annealing(initial_solution, seed, max_iters=5000, t_start=10.0, t_end=0.01, cooling_rate=0.999):
        """
        Simulated Annealing to optimize the set of residue classes.
        """
        random.seed(seed)
        
        # State: list of [m, r]
        current_sol = [list(x) for x in initial_solution]
        current_score = get_score(current_sol)
        
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        # Map m -> index for quick lookup
        def get_idx(m):
            return m - 20
            
        # Build a set of selected indices for fast membership check
        selected_indices = set()
        for m, r in current_sol:
            selected_indices.add(get_idx(m))
            
        T = t_start
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.5:
                # Move 1: Change residue of an existing modulus, then try to insert a new one
                if not current_sol:
                    continue
                    
                # Pick a random element from current_sol
                i = random.randint(0, len(current_sol) - 1)
                m, r_old = current_sol[i]
                idx_m = get_idx(m)
                
                # Try a new random residue
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                
                # Check compatibility with others
                compatible = True
                for j, (m_j, r_j) in enumerate(current_sol):
                    if j == i:
                        continue
                    idx_j = get_idx(m_j)
                    g = gcd_mat[idx_m][idx_j]
                    if r_new % g == r_j % g:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Apply residue change (score unchanged)
                current_sol[i][1] = r_new
                
                # Now try to insert a new modulus that was previously blocked
                missing = [i for i in range(n_mods) if i not in selected_indices]
                if missing:
                    # Prioritize smaller moduli (higher 1/m)
                    missing.sort(key=lambda i: -inv_m[i])
                    candidates = missing[:15]
                    idx_new = random.choice(candidates)
                    m_new = moduli[idx_new]
                    
                    # Try to find a compatible residue
                    found_r = None
                    for _ in range(8):
                        r_try = random.randint(0, m_new - 1)
                        # Check against all current entries
                        ok = True
                        for j, (m_j, r_j) in enumerate(current_sol):
                            idx_j = get_idx(m_j)
                            g = gcd_mat[idx_new][idx_j]
                            if r_try % g == r_j % g:
                                ok = False
                                break
                        if ok:
                            found_r = r_try
                            break
                    
                    if found_r is not None:
                        # Insert the new modulus
                        current_sol.append([m_new, found_r])
                        selected_indices.add(idx_new)
                        delta = inv_m[idx_new]
                        current_score += delta
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_sol = [list(x) for x in current_sol]
                    else:
                        # Keep the residue change even if no insertion succeeded
                        # (it might help future moves)
                        if current_score > best_score:
                            best_score = current_score
                            best_sol = [list(x) for x in current_sol]
                        
            elif move_type < 0.8:
                # Move 2: Insert a new modulus
                missing = [i for i in range(n_mods) if i not in selected_indices]
                if not missing:
                    continue
                
                # Prioritize smaller moduli (higher 1/m)
                # Pick from top 20 missing by 1/m
                missing.sort(key=lambda i: -inv_m[i])
                candidates = missing[:20]
                idx_m = random.choice(candidates)
                m = moduli[idx_m]
                
                # Try to find a compatible residue
                found_r = None
                for _ in range(10):
                    r = random.randint(0, m - 1)
                    ok = True
                    for j, (m_j, r_j) in enumerate(current_sol):
                        idx_j = get_idx(m_j)
                        g = gcd_mat[idx_m][idx_j]
                        if r % g == r_j % g:
                            ok = False
                            break
                    if ok:
                        found_r = r
                        break
                
                if found_r is None:
                    continue
                
                # Apply insertion
                current_sol.append([m, found_r])
                selected_indices.add(idx_m)
                
                delta = inv_m[idx_m]
                new_score = current_score + delta
                
                # Acceptance
                if delta > 0:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_sol = [list(x) for x in current_sol]
                else:
                    # Should not happen for insertion since 1/m > 0
                    pass
                    
            else:
                # Move 3: Delete a modulus
                if not current_sol:
                    continue
                
                # Prefer deleting larger moduli (lower 1/m)
                # Sort current_sol by 1/m ascending
                sorted_sol = sorted(current_sol, key=lambda x: inv_m[get_idx(x[0])])
                # Pick from bottom 10 (largest m)
                candidates = sorted_sol[:10]
                m_del, r_del = random.choice(candidates)
                idx_del = get_idx(m_del)
                
                # Find and remove
                for i, (m, r) in enumerate(current_sol):
                    if m == m_del and r == r_del:
                        current_sol.pop(i)
                        break
                
                selected_indices.remove(idx_del)
                
                delta = -inv_m[idx_del]
                new_score = current_score + delta
                
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_sol = [list(x) for x in current_sol]
                else:
                    # Reject: Undo
                    current_sol.append([m_del, r_del])
                    selected_indices.add(idx_del)
                    # current_score remains unchanged
                    
            # Cool down
            T *= cooling_rate
            if T < t_end:
                T = t_end
                
        return best_sol

    best_solution = []
    best_score = -1
    
    # Strategy: run greedy fills to get good initial states, then SA
    num_greedy = 0
    
    while time.time() < deadline - 15.0:
        seed = num_greedy + 1
        
        # Create a priority order: bias toward small moduli
        indices = list(range(n_mods))
        
        if random.random() < 0.7:
            # Sort by 1/m descending with some noise
            indices.sort(key=lambda i: -inv_m[i] + random.uniform(-0.01, 0.01))
        else:
            random.shuffle(indices)
        
        sol = greedy_fill(seed, priority_order=indices)
        score = get_score(sol)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
        num_greedy += 1
        
        # Run SA on the current best (or this new one if it's better)
        # We run SA on the best solution found so far to refine it
        if num_greedy % 2 == 0 or score > best_score:
            # Use remaining time budget for SA
            sa_sol = simulated_annealing(best_solution, seed + 10000, max_iters=30000, t_start=5.0, t_end=0.01, cooling_rate=0.99995)
            sa_score = get_score(sa_sol)
            
            if sa_score > best_score:
                best_score = sa_score
                best_solution = sa_sol
    
    # Final polish if time allows
    if best_solution and time.time() < deadline - 5.0:
        final_sol = simulated_annealing(best_solution, 99999, max_iters=80000, t_start=2.0, t_end=0.001, cooling_rate=0.99998)
        final_score = get_score(final_sol)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol
    
    if not best_solution:
        return [[20 * j, j - 1] for j in range(1, 11)]
        
    return best_solution
