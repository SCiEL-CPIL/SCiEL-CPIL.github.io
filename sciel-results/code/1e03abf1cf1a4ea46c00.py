import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute gcds
    gcd_cache = {}
    for m in range(20, 201):
        for mp in range(m + 1, 201):
            gcd_cache[(m, mp)] = math.gcd(m, mp)
            
    all_moduli = list(range(20, 201))

    def compute_costs(m):
        """
        Precompute the penalty cost for each residue r in [0, m)
        based on conflicts with all future moduli m' > m.
        """
        costs = [0.0] * m
        # Iterate over future moduli
        for mp in range(m + 1, 201):
            g = gcd_cache[(m, mp)]
            # For each residue class k modulo g
            for k in range(g):
                # Count of r' in [0, mp) such that r' % g == k
                # r' = k, k+g, ... < mp
                if k < mp:
                    count = (mp - 1 - k) // g + 1
                else:
                    count = 0
                if count == 0:
                    continue
                # Add count to all r in [0, m) such that r % g == k
                # r = k, k+g, k+2g... < m
                # Using slice assignment
                # costs[k::g] += count
                # This creates a new list, which is okay for small m
                # But m is up to 200.
                # Let's do it manually to avoid overhead if g is small
                for r in range(k, m, g):
                    costs[r] += count
        return costs

    def greedy_complete(selected, start_m):
        """
        Complete the selection using greedy look-ahead.
        selected: list of [m, r]
        start_m: next modulus to consider
        """
        # Cache costs for moduli
        # We can compute costs on the fly or precompute for all m.
        # Precomputing for all m is 200 * (200 * 200 / 2) ~ 4M ops. Fast.
        # But costs depend on m.
        # Let's precompute costs for all m in [20, 200] once at the start of construct?
        # No, it's better to do it inside if we want to be safe, but precomputing is faster.
        
        for m in range(start_m, 201):
            if time.time() > deadline:
                break
                
            # Get costs for this m
            costs = compute_costs(m)
            
            # Find valid residues
            valid_residues = []
            for r in range(m):
                is_valid = True
                for (m_prev, r_prev) in selected:
                    g = math.gcd(m, m_prev)
                    if r % g == r_prev % g:
                        is_valid = False
                        break
                if is_valid:
                    valid_residues.append(r)
            
            if not valid_residues:
                continue # Skip this modulus
                
            # Choose best residue (min cost)
            best_r = min(valid_residues, key=lambda r: costs[r])
            
            selected.append([m, best_r])
                
        return selected

    def score(classes):
        return sum(1.0 / m for m, r in classes)

    best_score = 0.0
    best_classes = []
    
    # Branching over m=20 and m=21
    for r20 in range(20):
        if time.time() > deadline:
            break
        sel_20 = [[20, r20]]
        
        for r21 in range(21):
            if time.time() > deadline:
                break
            # gcd(20,21)=1, always valid
            sel_21 = sel_20 + [[21, r21]]
            
            # Run greedy for 22..200
            current_sel = [list(item) for item in sel_21]
            
            final_sel = greedy_complete(current_sel, 22)
            
            s = score(final_sel)
            if s > best_score:
                best_score = s
                best_classes = [list(item) for item in final_sel]
                
    if not best_classes:
        best_classes = greedy_complete([], 20)
        
    return best_classes

# EVOLVE-BLOCK-END
