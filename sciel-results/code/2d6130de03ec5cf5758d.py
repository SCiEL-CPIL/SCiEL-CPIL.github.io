import random
import time
import math

def construct():
    start_time = time.time()
    # Hard deadline with safety margin (evaluator kills at 90s)
    deadline = start_time + 50.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    # Precompute GCD table for speed
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(2, max_mod + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    def is_compatible(selected, r, m):
        """Check if (r, m) is disjoint from all classes in selected."""
        for r2, m2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
        
    def find_residue(selected, m):
        """Find a valid residue for modulus m given selected classes."""
        for r in range(m):
            if is_compatible(selected, r, m):
                return r
        return None

    def greedy_solve(order):
        """Greedy construction based on a specific order of moduli."""
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_residue(selected, m)
            if r is not None:
                selected.append((r, m))
        return selected

    def light_local_search(selected):
        """Try to add missing moduli to an existing valid set. Single pass."""
        selected = [p[:] for p in selected]
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]
        
        # Try adding missing moduli in increasing order
        for m in missing:
            if time.time() > deadline:
                break
            r = find_residue(selected, m)
            if r is not None:
                selected.append((r, m))
                selected_moduli.add(m)
        
        return selected

    best_solution = []
    best_size = 0
    
    # 1. Initial baseline: Greedy with increasing moduli
    baseline = greedy_solve(moduli)
    best_solution = baseline
    best_size = len(baseline)
    
    # 2. Try different orders
    # Order by number of divisors (descending - most constrained first)
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        return count + 1
        
    moduli_by_divs = sorted(moduli, key=num_divisors, reverse=True)
    sol2 = greedy_solve(moduli_by_divs)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_solution = sol2

    # 3. Randomized restarts with greedy + light local search
    # Limit restarts to ensure we finish in time.
    # Each restart is ~O(N^2 * M). 
    # We'll do as many as we can within the deadline, but cap at 1000 to be safe.
    max_restarts = 1000
    restarts = 0
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        order = moduli[:]
        random.shuffle(order)
        
        # Greedy pass
        current = greedy_solve(order)
        
        # Light local search fill pass
        current = light_local_search(current)
        
        if len(current) > best_size:
            best_size = len(current)
            best_solution = [p[:] for p in current]
            
        # If we hit the theoretical max (127), we can stop? 
        if best_size == n_moduli:
            break

    # Final polish: try to improve best_solution with a more targeted local search if time permits
    if time.time() < deadline - 5:
        best_solution = [p[:] for p in best_solution]
        best_moduli_set = set(m for _, m in best_solution)
        
        # Try to improve best_solution by removing one and adding two
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline - 5:
                    break
                temp_sol = best_solution[:i] + best_solution[i+1:]
                temp_moduli = set(m for _, m in temp_sol)
                
                missing = [m for m in moduli if m not in temp_moduli]
                # Try to add 2 missing moduli
                added_count = 0
                for m in missing:
                    if time.time() > deadline - 5:
                        break
                    r = find_residue(temp_sol, m)
                    if r is not None:
                        temp_sol.append((r, m))
                        temp_moduli.add(m)
                        added_count += 1
                        if added_count == 2:
                            break
                            
                if added_count > 1:
                    best_solution = temp_sol
                    best_moduli_set = temp_moduli
                    improved = True
                    break # Restart loop with new state
                    
    # Return the best solution found
    if not best_solution:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_solution]
