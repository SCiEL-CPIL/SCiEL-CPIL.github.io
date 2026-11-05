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
    
    def local_search(solution, seed, max_iters=5000):
        """
        Local search: try swap, insert, and delete moves.
        solution: list of [m, r]
        Returns improved solution.
        """
        random.seed(seed)
        
        # Build index mapping: idx -> (m, r)
        sol_dict = {}
        for m, r in solution:
            idx = m - 20
            sol_dict[idx] = (m, r)
        
        current_score = get_score(solution)
        best_score = current_score
        best_sol = list(solution)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            move_type = random.choice(['swap', 'insert', 'delete'])
            
            if move_type == 'swap':
                # Pick a selected index and try a different residue
                if not sol_dict:
                    continue
                idx_m = random.choice(list(sol_dict.keys()))
                m, r_old = sol_dict[idx_m]
                
                # Try a random new residue
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                
                # Check compatibility
                compatible = True
                for idx_prime, (m_prime, r_prime) in sol_dict.items():
                    if idx_prime == idx_m:
                        continue
                    g = gcd_mat[idx_m][idx_prime]
                    if r_new % g == r_prime % g:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Apply swap
                new_solution = list(solution)
                for i, (m_s, r_s) in enumerate(new_solution):
                    if m_s == m and r_s == r_old:
                        new_solution[i] = [m, r_new]
                        break
                
                new_score = get_score(new_solution)
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_sol = new_solution
                    sol_dict[idx_m] = (m, r_new)
                    current_score = new_score
            
            elif move_type == 'insert':
                # Try to insert a missing modulus
                missing = [i for i in range(n_mods) if i not in sol_dict]
                if not missing:
                    continue
                
                # Prioritize smaller moduli (higher 1/m)
                missing.sort(key=lambda i: -inv_m[i])
                idx_m = random.choice(missing[:10])  # Pick from top 10 missing
                m = moduli[idx_m]
                
                # Try random residues
                found = False
                for _ in range(20):
                    r = random.randint(0, m - 1)
                    if is_compatible_idx(idx_m, r, [(i, sol_dict[i][1]) for i in sol_dict]):
                        # Insert
                        new_solution = best_sol + [[m, r]]
                        new_score = get_score(new_solution)
                        if new_score > best_score + 1e-15:
                            best_score = new_score
                            best_sol = new_solution
                            sol_dict[idx_m] = (m, r)
                            current_score = new_score
                        found = True
                        break
                
                if found:
                    continue
            
            elif move_type == 'delete':
                # Try removing a modulus (useful if it blocks better ones)
                if len(sol_dict) <= 1:
                    continue
                
                # Prefer removing larger moduli (lower 1/m)
                items = list(sol_dict.items())
                items.sort(key=lambda x: inv_m[x[0]])  # Ascending 1/m, so large m first
                idx_m = random.choice([i for i, _ in items[:10]])
                m, r = sol_dict[idx_m]
                
                new_solution = [x for x in best_sol if not (x[0] == m and x[1] == r)]
                new_score = get_score(new_solution)
                
                # Only accept if we can re-insert something better later
                # For now, just track but don't accept pure deletions unless they enable insertions
                # Skip pure deletions for now
        
        return best_sol
    
    best_solution = []
    best_score = -1
    
    # Strategy: run greedy fills with different priority orders, then local search
    num_greedy = 0
    num_local = 0
    
    while time.time() < deadline - 10.0:
        if time.time() > deadline:
            break
        
        seed = num_greedy + 1
        
        # Create a priority order: bias toward small moduli but with randomness
        # Use a mix: first K moduli in random order, then rest in random order
        indices = list(range(n_mods))
        
        # Bias: smaller moduli have higher priority
        # Use a weighted shuffle
        if random.random() < 0.5:
            # Half the time: sort by 1/m descending with some noise
            indices.sort(key=lambda i: -inv_m[i] + random.uniform(-0.01, 0.01))
        else:
            # Other half: fully random
            random.shuffle(indices)
        
        sol = greedy_fill(seed, priority_order=indices)
        score = get_score(sol)
        
        if score > best_score:
            best_score = score
            best_solution = sol
        
        num_greedy += 1
        
        # Do local search on the current best
        if num_greedy % 5 == 0 or score > best_score:
            local_sol = local_search(best_solution, seed + 10000, max_iters=2000)
            local_score = get_score(local_sol)
            
            if local_score > best_score:
                best_score = local_score
                best_solution = local_sol
            
            num_local += 1
    
    # Final local search pass with more iterations if time allows
    if best_solution and time.time() < deadline - 5.0:
        final_sol = local_search(best_solution, 99999, max_iters=10000)
        final_score = get_score(final_sol)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol
    
    if not best_solution:
        return [[20 * j, j - 1] for j in range(1, 11)]
        
    return best_solution
