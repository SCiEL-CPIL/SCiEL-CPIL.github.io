import math

def construct():
    max_mod = 128
    
    # The largest set of pairwise non-coprime integers in [2, 128] is the set of even numbers.
    # Size: 64.
    # Let's try to find valid residues for this set using CSP backtracking.
    
    moduli = list(range(2, max_mod + 1, 2))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
    
    # CSP Backtracking
    # variables: 0..n-1
    # assigned: list of residues or None
    assigned = [None] * n
    
    def is_consistent(var, val):
        # Check if assigning `val` to `var` is consistent with all already assigned variables
        for u in range(n):
            if assigned[u] is not None:
                g = gcds[var][u]
                if (val - assigned[u]) % g == 0:
                    return False
        return True

    def solve(idx):
        if idx == n:
            return True
        
        # Heuristic: try to assign variable with most constraints? 
        # For now, just go in order, but we can shuffle or use MRV if needed.
        # Given the structure, order might matter.
        # Let's try to assign smaller moduli first as they have smaller domains.
        # moduli are already sorted ascending.
        
        # Try all values in domain of moduli[idx]
        m = moduli[idx]
        # Optimization: shuffle values to avoid systematic bias? 
        # Or just try 0..m-1.
        for r in range(m):
            if is_consistent(idx, r):
                assigned[idx] = r
                if solve(idx + 1):
                    return True
                assigned[idx] = None
        
        return False

    # To increase success rate, we might need to try different orderings or use AC-3.
    # Simple backtracking might be slow if it gets stuck.
    # Let's add a simple AC-3 prune before backtracking? 
    # Or just try a few random restarts if it fails?
    
    # Given the symmetry, let's just run it.
    # If it takes too long, we need a better heuristic.
    # But 64 variables is small.
    
    success = solve(0)
    
    if success:
        result = [[assigned[i], moduli[i]] for i in range(n)]
        return result
    else:
        # Fallback to greedy if CSP fails (should not happen for evens, but just in case)
        # Or return a smaller set
        # Let's try the set of multiples of 2, but maybe skip some?
        # For now, return the greedy result from before as fallback
        # But we should try to construct a valid set if CSP fails.
        
        # Let's try a smaller clique: multiples of 2, 3, 5? No, coprime issues.
        # Just return the even set if we can find ANY solution.
        # If backtracking fails, it might be due to time limit or no solution.
        # Let's assume a solution exists.
        
        # Fallback: Greedy from original code
        moduli_greedy = list(range(2, max_mod + 1))
        def prime_factors(n):
            pf = set()
            d = 2
            while d * d <= n:
                while n % d == 0:
                    pf.add(d)
                    n //= d
                d += 1
            if n > 1:
                pf.add(n)
            return pf
        
        pf_map = {m: prime_factors(m) for m in moduli_greedy}
        sorted_moduli = sorted(moduli_greedy, key=lambda m: (-len(pf_map[m]), min(pf_map[m]) if pf_map[m] else 999, m))
        
        selected = []
        for m in sorted_moduli:
            can_coexist = True
            for (r_s, m_s) in selected:
                if math.gcd(m, m_s) == 1:
                    can_coexist = False
                    break
            if not can_coexist:
                continue
            
            found = False
            for r in range(m):
                valid = True
                for (r_s, m_s) in selected:
                    g = math.gcd(m, m_s)
                    if (r - r_s) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    found = True
                    break
            
        if selected:
            return selected
        return [[1, 2]]
