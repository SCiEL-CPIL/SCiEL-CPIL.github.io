import math

def construct():
    max_mod = 128
    
    # The largest clique in the non-coprime graph is the set of all even numbers.
    # Let's assume the optimal set of moduli is a subset of the even numbers.
    # We will try to find a valid residue assignment for ALL even numbers first.
    
    candidate_moduli = [m for m in range(2, max_mod + 1) if m % 2 == 0]
    
    # If for some reason evens don't work (unlikely), we could try other cliques,
    # but evens are the largest.
    
    n = len(candidate_moduli)
    if n == 0:
        return [[1, 2]]
        
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(candidate_moduli[i], candidate_moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Initial domains
    domains = [list(range(m)) for m in candidate_moduli]
    
    # AC-3 implementation
    def arc_consistency(domains):
        queue = [(i, j) for i in range(n) for j in range(n) if i != j]
        while queue:
            i, j = queue.pop(0)
            if revise(domains, i, j):
                if not domains[i]:
                    return False
                for k in range(n):
                    if k != i and k != j:
                        queue.append((k, i))
        return True
        
    def revise(domains, i, j):
        revised = False
        g = gcds[i][j]
        # For each value in domains[i], check if there is a supporting value in domains[j]
        # Support means: v_i NOT congruent v_j mod g
        # So v_i is supported if there exists v_j in domains[j] such that (v_i - v_j) % g != 0
        # This is equivalent to: domains[j] contains at least one value NOT in (v_i mod g)
        # Or more simply: if domains[j] has size > 1, it's likely supported.
        # Specifically, v_i is removed if ALL v_j in domains[j] satisfy (v_i - v_j) % g == 0.
        # This happens if domains[j] is a subset of {v | v % g == v_i % g}.
        
        # Optimization: If len(domains[j]) > g, it must contain a value not congruent to v_i mod g?
        # No, if g=2, and domains[j] = {0, 2, 4}, and v_i=1, then 1-0=1 (ok), 1-2=-1 (ok).
        # If domains[j] = {0, 2}, and v_i=0, then 0-0=0 (bad), 0-2=-2 (bad). So 0 is removed.
        
        to_remove = []
        for v_i in domains[i]:
            supported = False
            # Check if there exists v_j in domains[j] such that (v_i - v_j) % g != 0
            # This is false only if for ALL v_j, (v_i - v_j) % g == 0
            # Which means all v_j in domains[j] are congruent to v_i mod g.
            
            # Efficient check:
            # If len(domains[j]) == 1:
            #   v_j = domains[j][0]
            #   if (v_i - v_j) % g == 0: not supported
            # else:
            #   We need to check if domains[j] is contained in the congruence class of v_i mod g.
            #   This is expensive.
            
            # Simple check for now:
            for v_j in domains[j]:
                if (v_i - v_j) % g != 0:
                    supported = True
                    break
            
            if not supported:
                to_remove.append(v_i)
                
        if to_remove:
            for v in to_remove:
                domains[i].remove(v)
            revised = True
            
        return revised

    # Run AC-3
    if not arc_consistency(domains):
        # If AC-3 fails, the set of moduli is infeasible.
        # We might need to drop some moduli.
        # For now, let's try backtracking with the pruned domains.
        # But if a domain is empty, it's infeasible.
        pass
        
    # Backtracking Search
    # Order variables by MRV (Minimum Remaining Values)
    # We need to maintain the order or pick dynamically.
    
    assignment = {}
    
    def backtrack(assigned_vars):
        if len(assigned_vars) == n:
            return True
            
        # Select unassigned variable with smallest domain
        unassigned = [i for i in range(n) if i not in assigned_vars]
        if not unassigned:
            return True
            
        # MRV heuristic
        var = min(unassigned, key=lambda i: len(domains[i]))
        
        # LCV heuristic (Least Constraining Value) - optional, skip for simplicity
        for val in domains[var]:
            # Check consistency with assigned variables
            consistent = True
            for other, other_val in assigned_vars.items():
                g = gcds[var][other]
                if (val - other_val) % g == 0:
                    consistent = False
                    break
                    
            if consistent:
                assigned_vars[var] = val
                # Save state of domains for backtracking?
                # AC-3 is destructive. We need to copy domains or undo.
                # For simplicity and speed, since N=64, copying lists of size up to 128 is cheap.
                old_domains = [list(d) for d in domains]
                
                # We don't re-run AC-3 in every step for speed, just check direct constraints.
                # This is a "forward checking" approach.
                
                if backtrack(assigned_vars):
                    return True
                    
                # Undo
                assigned_vars.pop(var)
                domains[:] = old_domains
                
        return False

    if backtrack({}):
        # Construct result
        result = []
        for i in range(n):
            r = assignment[i]
            m = candidate_moduli[i]
            result.append([r, m])
        return result
    else:
        # Fallback: return a small valid set
        # E.g., just [[1, 2]]
        return [[1, 2]]
