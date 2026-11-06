import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    
    # Precompute gcd table for all pairs of moduli
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute prime factor counts (omega) for ordering heuristics
    def omega(m):
        count = 0
        d = 2
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    omega_cache = {m: omega(m) for m in moduli}
    
    # Precompute a few deterministic orderings
    orderings = []
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], m)))  # Most constrained first
    orderings.append(sorted(moduli, key=lambda m: (omega_cache[m], m)))   # Least constrained first
    orderings.append(sorted(moduli))                                       # Ascending
    orderings.append(sorted(moduli, reverse=True))                         # Descending
    
    def _find_residue_fast(m_idx, selected_pairs, selected_mod_indices):
        """
        Find a residue r in [0, moduli[m_idx]) that is disjoint from all selected classes.
        selected_pairs: list of (r, m)
        selected_mod_indices: list of indices into moduli for the selected moduli
        """
        m = moduli[m_idx]
        # Build constraints: for each selected class with gcd > 1, we have (g, r_i)
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))
        
        if not constraints:
            return 0  # Any residue works, pick 0
        
        # Try residues in a smart order: start with 0, then try others
        # Actually, for speed, just iterate 0..m-1
        for r in range(m):
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy_pass(order_indices):
        """
        One greedy pass given an ordering of modulus indices.
        Returns (list_of_pairs, list_of_mod_indices)
        """
        selected_pairs = []
        selected_mod_indices = []
        used = set()
        
        for m_idx in order_indices:
            if moduli[m_idx] in used:
                continue
            # Check if this modulus can even be added (gcd must be > 1 with all selected? No.)
            # Actually, if gcd == 1 with any selected modulus, they are automatically disjoint.
            # So we only need to worry about moduli where gcd > 1 with at least one selected.
            # But wait: if gcd(m_new, m_sel) == 1, then (r_new - r_sel) % 1 == 0 is always true?
            # No: gcd=1 means the condition (r1-r2) % g != 0 becomes (r1-r2) % 1 != 0 which is 0 != 0 -> False.
            # Wait, the condition for DISJOINT is (r1-r2) % gcd(m1,m2) != 0.
            # If gcd=1, then (r1-r2) % 1 is always 0. So 0 != 0 is False.
            # This means if gcd=1, the classes are NEVER disjoint?
            # Let me re-read the math.
            # Two arithmetic progressions r1 + k*m1 and r2 + l*m2 intersect iff r1 ≡ r2 (mod gcd(m1,m2)).
            # So they are DISJOINT iff r1 NOT ≡ r2 (mod gcd(m1,m2)).
            # If gcd=1, then r1 ≡ r2 (mod 1) is ALWAYS true (since everything is 0 mod 1).
            # So if gcd=1, the classes ALWAYS intersect. They are NEVER disjoint.
            # This is a crucial insight!
            # Therefore, to have disjoint classes, we MUST have gcd(m1, m2) > 1 for ALL pairs.
            # And then we need to choose residues such that r1 NOT ≡ r2 (mod gcd(m1,m2)).
            
            # So the first filter is: gcd(m_new, m_sel) must be > 1 for ALL selected m_sel.
            compatible = True
            for m_sel_idx in selected_mod_indices:
                if gcd_table[m_idx][m_sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            r = _find_residue_fast(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, moduli[m_idx]))
                selected_mod_indices.append(m_idx)
                used.add(moduli[m_idx])
        
        return selected_pairs, selected_mod_indices
    
    best_pairs = []
    best_count = 0
    
    # Run deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        order_indices = [moduli.index(m) for m in order]
        pairs, _ = greedy_pass(order_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
    
    # Run randomized restarts
    random.seed(42)
    num_restarts = 2000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering of all moduli
        all_indices = list(range(n_mod))
        random.shuffle(all_indices)
        
        pairs, _ = greedy_pass(all_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
    
    # Local search: try to add more moduli to the best solution
    if time.time() < deadline - 5.0 and best_pairs:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_pairs = best_pairs[:]
            current_mod_indices = []
            used = set()
            for (r, m) in current_pairs:
                m_idx = moduli.index(m)
                current_mod_indices.append(m_idx)
                used.add(m)
            
            # Try adding each unused modulus
            for m_idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[m_idx]
                if m in used:
                    continue
                
                # Check gcd compatibility
                compatible = True
                for m_sel_idx in current_mod_indices:
                    if gcd_table[m_idx][m_sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                r = _find_residue_fast(m_idx, current_pairs, current_mod_indices)
                if r is not None:
                    current_pairs.append((r, m))
                    current_mod_indices.append(m_idx)
                    used.add(m)
                    improved = True
            
            if len(current_pairs) > len(best_pairs):
                best_pairs = current_pairs
    
    # Also try a "repair" strategy: start with a random subset and try to complete it
    # This might be overkill, but let's add a few more restarts with different strategies
    if time.time() < deadline - 10.0:
        # Try sorting by descending modulus (larger moduli have more residue options)
        order_desc = sorted(moduli, reverse=True)
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            order_indices = [moduli.index(m) for m in order_desc]
            # Perturb the ordering slightly
            for i in range(len(order_indices) - 1):
                if random.random() < 0.1:
                    j = random.randint(i + 1, len(order_indices) - 1)
                    order_indices[i], order_indices[j] = order_indices[j], order_indices[i]
            
            pairs, _ = greedy_pass(order_indices)
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_pairs = pairs[:]
    
    return [[r, m] for r, m in best_pairs]

# EVOLVE-BLOCK-END
