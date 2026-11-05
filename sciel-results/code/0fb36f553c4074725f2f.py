import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    MAX_M = 128
    # Based on analysis, only even moduli can form large disjoint sets
    # because if an odd number is included, it must share a factor with all evens,
    # which is impossible if 2 is included. If 2 is excluded, the set size is smaller.
    # Max size of clique in non-coprime graph is 64 (all evens).
    # We try to solve for all evens, then prune if necessary.
    
    moduli = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    n = len(moduli)
    
    # Precompute gcd for all pairs of moduli
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def solve_csp(order):
        """
        Backtracking solver with forward checking.
        order: list of indices to assign in this order.
        """
        selected = []
        
        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == len(order):
                return True
            
            idx = order[pos]
            m = moduli[idx]
            
            # Try residues in a specific order. 
            # Heuristic: try residues that are "far" from existing ones?
            # Or just 0, 1, ...
            # Randomizing can help escape bad local optima if we restart.
            # But for a single deterministic run, 0..m-1 is fine.
            
            # Optimization: Filter residues early
            # We can pre-filter based on the first few selected items if needed, 
            # but simple check is O(K).
            
            for r in range(m):
                if check_residue(idx, r, selected):
                    selected.append((idx, r))
                    if backtrack(pos + 1):
                        return True
                    selected.pop()
            return False
        
        return backtrack(0)

    # We want to maximize the number of classes.
    # Start with all 64 even moduli.
    # If solvable, return 64.
    # If not, remove one and try again.
    
    # Order: Most constrained first (highest degree in conflict graph?)
    # Degree is just number of other moduli. All have 63 neighbors.
    # Maybe order by modulus size? Smaller moduli have smaller domains.
    # MRV: Smallest domain first. Domain size is m.
    # So sort by modulus ascending.
    
    current_moduli = moduli[:]
    best_solution = []
    best_len = 0
    
    # Try to solve for all 64
    if solve_csp(sorted(range(len(current_moduli)), key=lambda i: current_moduli[i])):
        best_len = len(current_moduli)
        # Reconstruct solution
        # Note: solve_csp modifies a local 'selected' list in its closure? 
        # No, I need to return the solution.
        
        # Let's rewrite solve_csp to return the solution
        pass

    # Better implementation:
    def find_solution(moduli_list):
        n_local = len(moduli_list)
        if n_local == 0:
            return []
        
        # Re-index
        indices = list(range(n_local))
        # Sort by modulus size (MRV heuristic)
        indices.sort(key=lambda i: moduli_list[i])
        
        selected = []
        
        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == len(indices):
                return True
            
            idx = indices[pos]
            m = moduli_list[idx]
            
            # Try residues
            for r in range(m):
                ok = True
                for (sel_idx, sel_r) in selected:
                    g = math.gcd(moduli_list[idx], moduli_list[sel_idx])
                    if (r - sel_r) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append((idx, r))
                    if backtrack(pos + 1):
                        return True
                    selected.pop()
            return False
        
        if backtrack(0):
            return [(moduli_list[idx], r) for (idx, r) in selected]
        return None

    # Try all evens
    solution = find_solution(moduli)
    if solution:
        return [[r, m] for (m, r) in solution]
        
    # If all evens fail, we need to remove some.
    # Which ones to remove?
    # The "hardest" constraints are likely those with small gcds (gcd=2).
    # Removing a modulus that is involved in many gcd=2 conflicts might help.
    
    # We can use a greedy removal strategy.
    # Start with all evens.
    # While no solution:
    #   Find the modulus that appears in the most "conflict" pairs in the failure trace?
    #   Or just remove the largest modulus?
    #   Or random removal?
    
    # Given the time, a simple iterative removal of the largest modulus might work,
    # or random removal.
    
    current_set = moduli[:]
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        sol = find_solution(current_set)
        if sol:
            return [[r, m] for (m, r) in sol]
            
        # Remove a random modulus
        if len(current_set) > 1:
            idx_to_remove = random.randint(0, len(current_set) - 1)
            current_set.pop(idx_to_remove)
            
    # Fallback: Return the largest set we could verify?
    # Or just return a known small set.
    # Actually, if find_solution fails, it returns None.
    # We should keep the best solution found in previous iterations if we were doing local search.
    # But here we are just pruning.
    
    # If we pruned down to a small set and it worked, we return it.
    # If we ran out of time, we return the last valid one?
    # We didn't store intermediate valid ones because we only return on success.
    
    # Let's do a local search approach if exact backtracking is too slow.
    # But backtracking on 64 vars with domains up to 128 should be fast if the solution is dense.
    # The problem is likely that NO solution exists for all 64.
    
    # Let's assume the max is around 50-60.
    
    # If we reach here, we failed to find a solution for the current_set.
    # This shouldn't happen if we reduce enough.
    
    return [[0, 2]]

# Test the logic mentally
# The backtracking might be slow if it explores many dead ends.
# We can add a random restart to the residue ordering.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    
    def find_solution(moduli_list, seed=0):
        n_local = len(moduli_list)
        if n_local == 0:
            return []
        
        indices = list(range(n_local))
        indices.sort(key=lambda i: moduli_list[i])
        
        selected = []
        random.seed(seed)
        
        # Precompute gcds for this subset
        gcd_sub = [[0]*n_local for _ in range(n_local)]
        for i in range(n_local):
            for j in range(i+1, n_local):
                g = math.gcd(moduli_list[i], moduli_list[j])
                gcd_sub[i][j] = g
                gcd_sub[j][i] = g

        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == len(indices):
                return True
            
            idx = indices[pos]
            m = moduli_list[idx]
            
            # Generate residues in random order to help with restarts
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                ok = True
                for (sel_idx, sel_r) in selected:
                    g = gcd_sub[idx][sel_idx]
                    if (r - sel_r) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append((idx, r))
                    if backtrack(pos + 1):
                        return True
                    selected.pop()
            return False
        
        if backtrack(0):
            return [(moduli_list[idx], r) for (idx, r) in selected]
        return None

    # Try to solve for all 64 evens with a few random restarts
    for s in range(10):
        if time.time() > deadline - 5:
            break
        sol = find_solution(moduli, seed=s)
        if sol:
            return [[r, m] for (m, r) in sol]

    # If all 64 fail, prune
    current_set = moduli[:]
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        if len(current_set) == 0:
            break
            
        sol = find_solution(current_set, seed=attempt)
        if sol:
            return [[r, m] for (m, r) in sol]
            
        # Remove a random modulus
        idx_to_remove = random.randint(0, len(current_set) - 1)
        current_set.pop(idx_to_remove)
        
    # Fallback
    return [[0, 2]]

print(construct())
