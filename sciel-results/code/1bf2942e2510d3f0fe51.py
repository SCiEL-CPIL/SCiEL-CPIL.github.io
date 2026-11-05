import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    MAX_M = 128
    # Use even moduli as they are pairwise non-coprime (gcd >= 2)
    # This is the largest possible set of pairwise non-coprime integers in [2, 128]
    candidates = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    n = len(candidates)
    
    if n == 0:
        return []

    # Precompute gcds
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])

    # Backtracking solver
    # domains: list of sets of valid residues for each index
    # assigned: list of (index, residue) or None
    # We use a recursive backtracking with forward checking
    
    def solve():
        # Initial domains
        domains = [set(range(candidates[i])) for i in range(n)]
        
        # Order indices to process. 
        # Heuristic: Process moduli with smaller values first? 
        # Or those with more constraints (higher degree)?
        # Let's try processing in natural order first, or by "tightness".
        # Actually, MRV (Minimum Remaining Values) is dynamic.
        
        # To speed up, we can fix an order of variables for selection,
        # but use MRV to pick which one to branch on next.
        
        unassigned = set(range(n))
        assignment = [None] * n
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            if not unassigned:
                return True
            
            # Select variable with MRV
            min_len = float('inf')
            target_idx = -1
            for idx in unassigned:
                if len(domains[idx]) < min_len:
                    min_len = len(domains[idx])
                    target_idx = idx
                    if min_len == 0:
                        return False
                    if min_len == 1:
                        break # Can't get better than 1
            
            if target_idx == -1:
                return False
            
            # Try each value in domain
            # Sort values to be deterministic? Or random?
            # Let's try values in ascending order.
            values = list(domains[target_idx])
            
            for r in values:
                # Check if this value is consistent with already assigned variables
                # Actually, forward checking maintains consistency in domains.
                # But we must ensure r is in domains[target_idx]. It is.
                
                # Save state for backtracking
                # We need to revert domain changes
                # To be efficient, we can copy domains or use a trail.
                # Given small N, copying sets is okay.
                old_domains = [set(d) for d in domains]
                
                # Assign
                assignment[target_idx] = r
                unassigned.remove(target_idx)
                
                # Propagate
                valid = True
                for other_idx in unassigned:
                    g = gcd_tab[target_idx][other_idx]
                    # Remove values x in domains[other_idx] such that x == r (mod g)
                    # x % g == r % g
                    rem = r % g
                    # Filter domains[other_idx]
                    new_domain = set()
                    for x in domains[other_idx]:
                        if x % g != rem:
                            new_domain.add(x)
                    domains[other_idx] = new_domain
                    if not new_domain:
                        valid = False
                        break
                
                if valid and backtrack():
                    return True
                
                # Undo
                unassigned.add(target_idx)
                assignment[target_idx] = None
                domains = old_domains # Restore
                
            return False

        if backtrack():
            result = []
            for i in range(n):
                result.append([assignment[i], candidates[i]])
            return result
        else:
            return None

    # Attempt to solve for all even moduli
    solution = solve()
    
    if solution:
        return solution

    # Fallback: Greedy random restarts if backtracking fails or times out
    # This ensures we return something valid even if the exact solver struggles
    import random
    random.seed(42)
    
    def is_compatible(selected, new_idx, new_r):
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if (new_r - r) % g == 0:
                return False
        return True

    def solve_greedy_randomized(order):
        selected = []
        for idx in order:
            m = candidates[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    break
        return selected

    best_solution = []
    best_len = 0
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = solve_greedy_randomized(order)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
            
    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])
    return result

