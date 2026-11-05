import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    
    # The set of all even numbers is the largest known pairwise non-coprime subset
    # Any odd number is coprime to 2, so if 2 is included, no odd number can be.
    # The set of evens has size 64.
    # Let's verify if we can assign residues to all 64 evens.
    
    moduli = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    n = len(moduli)
    
    # Precompute gcds
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(moduli[i], moduli[j])
                
    # CSP Solver
    # Variables: indices 0..n-1
    # Domains: initially all residues 0..m-1
    # Constraints: for i != j, r_i not congruent r_j mod gcd(i,j)
    
    # We use a recursive backtracking with forward checking.
    # To speed up, we order variables by modulus size (smaller domains first? No, smaller moduli have smaller domains).
    # Actually, domain size is m. Smaller m means smaller domain.
    # So we should assign residues to smaller moduli first.
    # moduli are already sorted ascending: 2, 4, 6...
    
    # State: 
    # assigned: list of (idx, residue)
    # domains: list of sets of available residues for each idx
    
    # Optimization: 
    # The constraint is r_i != r_j mod g.
    # This means for a fixed r_i, the forbidden residues for j are those in the same class mod g.
    # We can maintain domains as sets and prune.
    
    # To avoid deep recursion overhead in Python, we might need to be careful.
    # But n=64 is small.
    
    # Let's try a greedy assignment first, then backtracking if it fails.
    # Greedy: Pick residues in order.
    
    def try_greedy(order):
        assigned = {}
        domains = [set(range(moduli[i])) for i in range(n)]
        
        for idx in order:
            if not domains[idx]:
                return None
            m = moduli[idx]
            # Pick a residue that doesn't conflict with already assigned ones
            # To be safe, we pick the first one that works.
            # But we must ensure it doesn't block future ones? 
            # Greedy might fail. Backtracking is safer.
            pass
        return None # Placeholder

    def solve_csp():
        # Order: ascending modulus (2, 4, 6...)
        order = list(range(n))
        
        # Initial domains
        domains = [set(range(moduli[i])) for i in range(n)]
        
        # We will use a recursive function
        # To prevent stack overflow or slow recursion, we limit depth or use iterative deepening?
        # No, n=64 is fine for recursion if pruned well.
        
        def backtrack(k):
            if k == n:
                return True
            
            idx = order[k]
            m = moduli[idx]
            
            # Try each residue in the current domain
            # To improve pruning, we might want to try residues that are "central" or random?
            # Let's just iterate.
            for r in list(domains[idx]):
                # Check if r is compatible with all previously assigned (0..k-1)
                # Actually, forward checking ensures domains[idx] only contains valid residues 
                # w.r.t. previously assigned variables IF we pruned correctly.
                # But we are doing forward checking *after* assignment.
                # So at step k, domains[idx] has been pruned by assignments 0..k-1.
                # So any r in domains[idx] is valid w.r.t. 0..k-1.
                
                # Assign r to idx
                domains[idx].remove(r)
                
                # Forward checking: prune domains of future variables
                pruned = []
                valid = True
                for j in range(k + 1, n):
                    j_idx = order[j]
                    g = gcd_tab[idx][j_idx]
                    if g == 1:
                        # Should not happen for evens
                        continue
                    # Forbidden residues in domains[j_idx] are those r_j such that r_j == r mod g
                    forbidden = set()
                    for r_j in domains[j_idx]:
                        if r_j % g == r % g:
                            forbidden.add(r_j)
                    if forbidden:
                        domains[j_idx] -= forbidden
                        if not domains[j_idx]:
                            valid = False
                            break
                        pruned.append((j_idx, forbidden))
                    
                if valid:
                    if backtrack(k + 1):
                        return True
                else:
                    # Restore
                    pass
                
                # Backtrack: restore domains
                domains[idx].add(r)
                if valid:
                    for j_idx, forbidden in pruned:
                        domains[j_idx] |= forbidden
            
            return False

        # Run solver
        if backtrack(0):
            # Extract solution
            result = []
            for i in range(n):
                # The assigned residue is the one that was removed and not re-added?
                # Wait, my backtracking logic removes r, tries, and adds it back.
                # If it returns True, the domain state is not necessarily the solution state.
                # I need to store the solution.
                pass
            return None # Need to fix solution extraction
        
        return None

    # The above CSP implementation is a bit messy for solution extraction.
    # Let's use a simpler, more robust backtracking that stores the path.
    
    def solve_csp_v2():
        order = list(range(n))
        domains = [set(range(moduli[i])) for i in range(n)]
        solution = [-1] * n
        
        def backtrack(k):
            if k == n:
                return True
            
            idx = order[k]
            m = moduli[idx]
            
            for r in list(domains[idx]):
                # Check compatibility with already assigned (0..k-1)
                # Since we use forward checking, domains[idx] is already pruned.
                # But let's double check just in case.
                compatible = True
                for prev_idx in order[:k]:
                    g = gcd_tab[idx][prev_idx]
                    if (r - solution[prev_idx]) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Assign
                solution[idx] = r
                domains[idx].remove(r)
                
                # Forward checking
                pruned = []
                valid = True
                for j in range(k + 1, n):
                    j_idx = order[j]
                    g = gcd_tab[idx][j_idx]
                    if g == 1: continue
                    forbidden = set()
                    for r_j in domains[j_idx]:
                        if r_j % g == r % g:
                            forbidden.add(r_j)
                    if forbidden:
                        domains[j_idx] -= forbidden
                        if not domains[j_idx]:
                            valid = False
                            break
                        pruned.append((j_idx, forbidden))
                    
                if valid:
                    if backtrack(k + 1):
                        return True
                
                # Restore
                domains[idx].add(r)
                if valid:
                    for j_idx, forbidden in pruned:
                        domains[j_idx] |= forbidden
            
            solution[idx] = -1
            return False

        if backtrack(0):
            result = []
            for i in range(n):
                result.append([solution[i], moduli[i]])
            return result
        return None

    # Try to solve for all evens
    sol = solve_csp_v2()
    
    if sol is not None:
        return sol
    
    # If CSP fails (unlikely), fallback to greedy or smaller set
    # But 64 evens should be solvable.
    
    # Fallback: Return a small valid set
    return [[0, 2]]
