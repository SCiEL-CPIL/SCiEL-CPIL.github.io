import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)
    
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])

    def get_valid_residues_fast(selected, idx):
        m = candidates[idx]
        blocked = [False] * m
        
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            if g == 1:
                return []
            
            base = sel_r % g
            for r in range(base, m, g):
                blocked[r] = True
        
        return [r for r in range(m) if not blocked[r]]

    def greedy_add_missing(sol):
        """Add any missing moduli that can be fit."""
        sol_idx_set = set(idx for idx, _ in sol)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for idx in range(n):
                if idx in sol_idx_set:
                    continue
                valid = get_valid_residues_fast(sol, idx)
                if valid:
                    r = random.choice(valid)
                    sol.append((idx, r))
                    sol_idx_set.add(idx)
                    changed = True
                    break
        return sol

    def perturb(sol, k=1):
        """Change residues of k random moduli."""
        if not sol:
            return sol
        sol = list(sol)
        sol_idx_set = set(idx for idx, _ in sol)
        indices = list(sol_idx_set)
        kick_indices = random.sample(indices, min(k, len(indices)))
        
        # Remove kicked
        remaining = [(idx, r) for idx, r in sol if idx not in kick_indices]
        
        # Re-add kicked with new random valid residues
        for idx in kick_indices:
            valid = get_valid_residues_fast(remaining, idx)
            if valid:
                r = random.choice(valid)
                remaining.append((idx, r))
            else:
                # If we can't re-insert, keep old? Or drop?
                # Dropping reduces size. Let's try to find a valid one.
                # If none, we might have to accept a smaller set or revert.
                # For now, if no valid, we just skip adding it back, effectively removing it.
                # This is a risk.
                pass
                
        return remaining

    def local_search(sol):
        """Hill climbing with perturbations."""
        current_sol = sol
        best_sol = list(sol)
        best_len = len(sol)
        
        while time.time() < deadline - 5:
            # Try to add missing
            current_sol = greedy_add_missing(current_sol)
            
            if len(current_sol) > best_len:
                best_sol = list(current_sol)
                best_len = len(current_sol)
                
            # Perturb
            # Choose k based on stagnation? Just fixed small k.
            k = random.randint(1, 2)
            neighbor = perturb(current_sol, k)
            
            # Local search on neighbor
            neighbor = greedy_add_missing(neighbor)
            
            # Accept if better or equal (with some probability if equal to escape plateaus)
            if len(neighbor) > len(current_sol):
                current_sol = neighbor
            elif len(neighbor) == len(current_sol) and random.random() < 0.1:
                current_sol = neighbor
            else:
                # If worse, we might still accept with low prob? 
                # For now, strict hill climbing.
                pass
                
        return best_sol

    random.seed(42)
    best_solution = []
    best_len = 0
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
            
        order = list(range(n))
        random.shuffle(order)
        
        # Initial greedy
        sol = []
        for idx in order:
            valid = get_valid_residues_fast(sol, idx)
            if valid:
                r = random.choice(valid)
                sol.append((idx, r))
                
        # Local search
        sol = local_search(sol)
        
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
            
    # Final polish
    if time.time() < deadline - 2:
        best_solution = greedy_add_missing(best_solution)

    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])
        
    return result
