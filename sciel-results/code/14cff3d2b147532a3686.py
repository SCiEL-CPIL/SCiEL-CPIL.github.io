"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    # List of all moduli
    moduli = list(range(20, 201))
    # Sort by increasing m to prioritize higher weight (1/m)
    moduli.sort()
    
    selected = []  # list of [m, r]
    
    # Precompute gcds for speed? Or just compute on the fly.
    # N=181, so N^2 is ~32k gcds. Fast.
    
    # Helper to check if a modulus m_new can be satisfied given a set of selected [m, r]
    def is_satisfiable(m_new, current_selected):
        # Check if there exists at least one residue r_new in [0, m_new) 
        # that is compatible with all in current_selected
        # Optimization: We don't need to find the residue, just if one exists.
        # But we need to know if *any* residue works.
        # A residue r_new is blocked if for some (m_s, r_s) in current_selected,
        # r_new % gcd(m_new, m_s) == r_s % gcd(m_new, m_s)
        
        # Efficient check:
        # For each (m_s, r_s), it forbids a set of residues mod m_new.
        # The forbidden residues are those congruent to r_s mod g, where g = gcd(m_new, m_s).
        # This means residues r_new such that r_new % g == r_s % g.
        # In the range [0, m_new), there are m_new/g such residues.
        # We need to know if the union of these forbidden sets covers all [0, m_new).
        
        # Since m_new <= 200, we can just iterate through all r_new in range(m_new)
        # and check compatibility. This is O(m_new * len(selected)).
        # m_new <= 200, len(selected) <= 181. 200*181 = 36200 ops.
        # Doing this for each candidate r in the lookahead is too slow if done naively.
        
        # Let's optimize the lookahead.
        pass

    # Revised Plan:
    # The lookahead "Cost" calculation:
    # For a candidate (m, r), we want to know how many future moduli m' become unsatisfiable.
    # To check if m' is unsatisfiable given selected + (m, r), we check if all residues of m' are blocked.
    # This is expensive to do for all future m'.
    
    # Let's simplify the heuristic:
    # Instead of full unsatisfiability check, count "Conflicts".
    # For a candidate (m, r), for each future m', check if the number of valid residues for m' decreases.
    
    # Given time constraints and Python speed, let's try a simpler distinct approach:
    # **Greedy with Randomized Restarts and Local Search**.
    
    import random
    
    def greedy_solve(order):
        sel = []
        for m in order:
            # Find all valid residues
            valid_rs = []
            for r in range(m):
                ok = True
                for m2, r2 in sel:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            if valid_rs:
                # Heuristic: Pick the residue that minimizes future conflicts?
                # Or just pick random to allow different paths?
                # Let's pick the one that has the "most freedom" for others?
                # Simple heuristic: Pick r=0? No, 0 is often restrictive.
                # Let's pick the residue that is "least common" or similar?
                # Actually, just picking the first valid one is what the current program does.
                # To make it distinct, let's pick the residue that minimizes the sum of gcds with future moduli?
                # Or simply pick a random valid residue to explore different branches.
                # But we want a deterministic best result.
                
                # Let's use a "Least Impact" heuristic:
                # For each candidate r, calculate the number of future moduli m' 
                # that have gcd(m, m') > 1.
                # This is static for m. It doesn't depend on r.
                # So this doesn't help choose r.
                
                # How about: For each candidate r, check how many future moduli m'
                # would have *fewer* valid residues if we pick r?
                # This is the true lookahead.
                
                # Let's implement the true lookahead for the residue choice.
                best_r = None
                min_conflicts = float('inf')
                
                # Precompute future moduli list for this step?
                # The 'order' list is fixed. We are at index i.
                # Future moduli are order[i+1:]
                
                for r in valid_rs:
                    conflicts = 0
                    # Check impact on future moduli
                    for m_future in order[order.index(m)+1:]:
                        # Check if m_future is still satisfiable
                        # A modulus is satisfiable if at least one residue is valid.
                        # We check if *all* residues are blocked.
                        all_blocked = True
                        for r_f in range(m_future):
                            # Check if r_f is compatible with current sel + (m, r)
                            # First check against current sel
                            comp_sel = True
                            for m2, r2 in sel:
                                g = gcd(m_future, m2)
                                if r_f % g == r2 % g:
                                    comp_sel = False
                                    break
                            if not comp_sel:
                                continue
                            # Check against new (m, r)
                            g_new = gcd(m_future, m)
                            if r_f % g_new == r % g_new:
                                comp_sel = False
                            
                            if comp_sel:
                                all_blocked = False
                                break
                        
                        if all_blocked:
                            conflicts += 1
                            
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_r = r
                        
                if best_r is not None:
                    sel.append([m, best_r])
        return sel

    # The greedy_solve above is O(N^3 * M^2) roughly, which might be too slow.
    # N=181. N^3 ~ 5.9M. Inner loop M^2 ~ 40k. 5.9M * 40k is too big.
    
    # We need a faster distinct approach.
    # How about **Simulated Annealing**?
    
    def simulated_annealing():
        # Initialize with a greedy solution
        # Order: increasing m
        init_order = list(range(20, 201))
        current_sel = []
        for m in init_order:
            valid_rs = []
            for r in range(m):
                ok = True
                for m2, r2 in current_sel:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            if valid_rs:
                current_sel.append([m, random.choice(valid_rs)])
        
        current_score = sum(1.0/m for m, r in current_sel)
        best_sel = [x[:] for x in current_sel]
        best_score = current_score
        
        T = 1.0
        cooling = 0.999
        iterations = 50000
        
        for i in range(iterations):
            if T < 1e-4:
                break
            
            # Generate a neighbor
            # Move types:
            # 1. Add a random valid modulus
            # 2. Remove a random modulus
            # 3. Change residue of a random modulus
            
            move_type = random.random()
            new_sel = [x[:] for x in current_sel]
            
            if move_type < 0.3:
                # Add
                candidates = []
                for m in range(20, 201):
                    if not any(x[0] == m for x in new_sel):
                        candidates.append(m)
                if candidates:
                    m = random.choice(candidates)
                    valid_rs = []
                    for r in range(m):
                        ok = True
                        for m2, r2 in new_sel:
                            g = gcd(m, m2)
                            if r % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            valid_rs.append(r)
                    if valid_rs:
                        new_sel.append([m, random.choice(valid_rs)])
                        
            elif move_type < 0.6:
                # Remove
                if new_sel:
                    idx = random.randint(0, len(new_sel)-1)
                    new_sel.pop(idx)
            else:
                # Change residue
                if new_sel:
                    idx = random.randint(0, len(new_sel)-1)
                    m, r_old = new_sel[idx]
                    valid_rs = []
                    for r in range(m):
                        if r == r_old: continue
                        ok = True
                        for j, (m2, r2) in enumerate(new_sel):
                            if j == idx: continue
                            g = gcd(m, m2)
                            if r % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            valid_rs.append(r)
                    if valid_rs:
                        new_sel[idx] = [m, random.choice(valid_rs)]
            
            new_score = sum(1.0/m for m, r in new_sel)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_sel = new_sel
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sel = [x[:] for x in current_sel]
            
            T *= cooling
            
        return best_sel

    # Run SA
    result = simulated_annealing()
    return result

# EVOLVE-BLOCK-END
