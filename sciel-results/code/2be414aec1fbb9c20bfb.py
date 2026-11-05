    import math
    import time
    from fractions import Fraction

    def construct():
        start_time = time.time()
        deadline = start_time + 110.0
        
        moduli = list(range(20, 201))
        n = len(moduli)
        
        # Precompute gcd
        gcd_mat = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = math.gcd(moduli[i], moduli[j])
                gcd_mat[i][j] = g
                gcd_mat[j][i] = g
        
        # Sort indices by weight (1/m) descending
        indices = sorted(range(n), key=lambda i: -1.0/moduli[i])
        
        selected = []
        
        # Heuristic: Try to build a clique
        for idx in indices:
            # Check gcd compatibility
            ok = True
            for sel in selected:
                if gcd_mat[idx][sel] == 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check CSP solvability
            if check_csp(selected + [idx], gcd_mat, moduli):
                selected.append(idx)
                
        if not selected:
            return [[20, 0], [22, 1]]
            
        # Get residues
        res = get_residues(selected, gcd_mat, moduli)
        if res is None:
            return [[20, 0], [22, 1]]
            
        out = []
        for i, idx in enumerate(selected):
            out.append([moduli[idx], res[i]])
            
        return out

    def check_csp(sel, gcd_mat, moduli):
        # MRV Heuristic
        # Domains are implicitly [0, m)
        # We can prune domains based on assigned values
        
        # Simple backtracking with MRV
        # Variables: indices in sel
        # To speed up, we can pre-filter domains?
        
        # Let's use a simple recursive backtracking with forward checking
        # Since sel size is small (< 20 likely), it should be fast.
        
        # Order variables by m (smaller m = smaller domain)
        order = sorted(sel, key=lambda i: moduli[i])
        
        domains = {i: list(range(moduli[i])) for i in sel}
        
        def backtrack(pos):
            if pos == len(order):
                return True
            
            # MRV: pick variable with smallest domain
            # But we are processing in fixed order 'order' for simplicity?
            # No, MRV is better.
            
            # Find unassigned variable with min domain size
            min_var = -1
            min_size = float('inf')
            for i in range(pos, len(order)):
                if i not in domains: # assigned
                    continue
                if len(domains[i]) < min_size:
                    min_size = len(domains[i])
                    min_var = i
            
            if min_var == -1:
                return True # All assigned? (Shouldn't happen if pos logic is correct)
            
            # Actually, the 'pos' logic above is for fixed order.
            # Let's stick to fixed order sorted by m for simplicity first, 
            # as MRV adds complexity.
            
            idx = order[pos]
            
            for r in domains[idx]:
                # Check consistency with already assigned
                consistent = True
                for prev_idx in order[:pos]:
                    g = gcd_mat[idx][prev_idx]
                    # We need to know assigned residue of prev_idx
                    # We need to store assignment
                    pass
                
            # This requires storing assignment.
            
            return False
        
        # Let's implement a cleaner MRV CSP solver
        assignment = {}
        
        def solve_csp_mrv():
            if len(assignment) == len(sel):
                return True
            
            # Select variable with min domain
            var = -1
            min_dom_size = float('inf')
            for i in sel:
                if i in assignment:
                    continue
                # Calculate current domain size
                # Domain of i is {r in [0, m_i) | r != assigned[j] mod gcd(i,j) for all assigned j}
                # We can estimate size or just iterate.
                # For MRV, we just pick the one with smallest m_i initially?
                # Or compute actual available count.
                
                m_i = moduli[i]
                # Count available
                avail = 0
                for r in range(m_i):
                    ok = True
                    for j, r_j in assignment.items():
                        g = gcd_mat[i][j]
                        if r % g == r_j % g:
                            ok = False
                            break
                    if ok:
                        avail += 1
                        if avail > min_dom_size: # Optimization: stop if exceeds min
                            break
                
                if avail < min_dom_size:
                    min_dom_size = avail
                    var = i
            
            if var == -1:
                return True # No unassigned?
                
            if min_dom_size == 0:
                return False
            
            m_var = moduli[var]
            for r in range(m_var):
                # Check if r is valid for var
                ok = True
                for j, r_j in assignment.items():
                    g = gcd_mat[var][j]
                    if r % g == r_j % g:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                assignment[var] = r
                if solve_csp_mrv():
                    return True
                del assignment[var]
            
            return False
        
        return solve_csp_mrv()

    def get_residues(sel, gcd_mat, moduli):
        assignment = {}
        
        def solve_csp_mrv():
            if len(assignment) == len(sel):
                return True
            
            var = -1
            min_dom_size = float('inf')
            for i in sel:
                if i in assignment:
                    continue
                
                m_i = moduli[i]
                avail = 0
                for r in range(m_i):
                    ok = True
                    for j, r_j in assignment.items():
                        g = gcd_mat[i][j]
                        if r % g == r_j % g:
                            ok = False
                            break
                    if ok:
                        avail += 1
                        if avail > min_dom_size:
                            break
                
                if avail < min_dom_size:
                    min_dom_size = avail
                    var = i
            
            if var == -1:
                return True
                
            if min_dom_size == 0:
                return False
            
            m_var = moduli[var]
            for r in range(m_var):
                ok = True
                for j, r_j in assignment.items():
                    g = gcd_mat[var][j]
                    if r % g == r_j % g:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                assignment[var] = r
                if solve_csp_mrv():
                    return True
                del assignment[var]
            
            return False
        
        if solve_csp_mrv():
            return [assignment[i] for i in sel]
        return None
    