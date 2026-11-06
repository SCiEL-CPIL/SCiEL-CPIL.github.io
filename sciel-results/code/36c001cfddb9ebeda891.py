import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
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
    
    # Group moduli by their smallest prime factor
    def smallest_prime_factor(m):
        if m % 2 == 0:
            return 2
        d = 3
        while d * d <= m:
            if m % d == 0:
                return d
            d += 2
        return m
    
    spf_cache = {m: smallest_prime_factor(m) for m in moduli}
    prime_groups = {}
    for m in moduli:
        p = spf_cache[m]
        if p not in prime_groups:
            prime_groups[p] = []
        prime_groups[p].append(m)
    
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
        
        # Try residues in a smart order
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
            # Check if this modulus can even be added
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
    
    def local_search_with_swaps(pairs, mod_indices, max_iters=100):
        """
        Local search: try to add more moduli, and if stuck, try swapping residues.
        Optimized to be faster.
        """
        current_pairs = pairs[:]
        current_mod_indices = mod_indices[:]
        used = set(m for (_, m) in current_pairs)
        best_count = len(current_pairs)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            
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
                    best_count = len(current_pairs)
                    break # Restart add loop if we added one
            
            if improved:
                continue
            
            # If no improvement, try swapping residues to free up space
            # Limit swaps to avoid O(N^4) blowup
            if len(current_pairs) < 5: # Only do swaps if set is small, otherwise too expensive
                for i in range(len(current_pairs)):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(i + 1, len(current_pairs)):
                        r_i, m_i = current_pairs[i]
                        r_j, m_j = current_pairs[j]
                        
                        # Try swapping
                        old_i, old_j = r_i, r_j
                        current_pairs[i] = (r_j, m_i)
                        current_pairs[j] = (r_i, m_j)
                        
                        # Check if swap is valid
                        valid = True
                        for k in range(len(current_pairs)):
                            r_k, m_k = current_pairs[k]
                            for l in range(k + 1, len(current_pairs)):
                                r_l, m_l = current_pairs[l]
                                g = gcd_table[moduli.index(m_k)][moduli.index(m_l)]
                                if g > 1 and (r_k - r_l) % g == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            # Revert
                            current_pairs[i] = (old_i, m_i)
                            current_pairs[j] = (old_j, m_j)
                            continue
                        
                        # Try adding more moduli after swap
                        added = False
                        for m_idx in range(n_mod):
                            m = moduli[m_idx]
                            if m in used:
                                continue
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
                                added = True
                                best_count = len(current_pairs)
                                break
                        
                        if not added:
                            # Revert swap
                            current_pairs[i] = (old_i, m_i)
                            current_pairs[j] = (old_j, m_j)
        
        return current_pairs, current_mod_indices
    
    best_pairs = []
    best_count = 0
    
    # Strategy 1: Search within each prime factor family
    for p, group in sorted(prime_groups.items(), key=lambda x: -len(x[1])):
        if time.time() > deadline - 10.0:
            break
        
        # Sort within group by descending modulus (more residue options first)
        group_indices = [moduli.index(m) for m in sorted(group, reverse=True)]
        pairs, mod_idx = greedy_pass(group_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
        
        # Local search within this family
        if len(pairs) > 0:
            improved_pairs, improved_idx = local_search_with_swaps(pairs, mod_idx, max_iters=50)
            if len(improved_pairs) > best_count:
                best_count = len(improved_pairs)
                best_pairs = improved_pairs[:]
    
    # Strategy 2: Deterministic orderings across all moduli
    orderings = []
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], m)))
    orderings.append(sorted(moduli, key=lambda m: (omega_cache[m], m)))
    orderings.append(sorted(moduli))
    orderings.append(sorted(moduli, reverse=True))
    
    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        order_indices = [moduli.index(m) for m in order]
        pairs, mod_idx = greedy_pass(order_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
        
        # Local search
        if len(pairs) > 0:
            improved_pairs, improved_idx = local_search_with_swaps(pairs, mod_idx, max_iters=50)
            if len(improved_pairs) > best_count:
                best_count = len(improved_pairs)
                best_pairs = improved_pairs[:]
    
    # Strategy 3: Randomized restarts
    random.seed(42)
    num_restarts = 500 # Reduced from 1500 to fit time budget
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random ordering
        all_indices = list(range(n_mod))
        random.shuffle(all_indices)
        
        pairs, mod_idx = greedy_pass(all_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
        
        # Occasional local search
        if restart % 20 == 0 and len(pairs) > 0:
            improved_pairs, improved_idx = local_search_with_swaps(pairs, mod_idx, max_iters=20)
            if len(improved_pairs) > best_count:
                best_count = len(improved_pairs)
                best_pairs = improved_pairs[:]
    
    # Final local search on best solution
    if best_pairs and time.time() < deadline - 2.0:
        mod_indices = [moduli.index(m) for (_, m) in best_pairs]
        final_pairs, _ = local_search_with_swaps(best_pairs, mod_indices, max_iters=50)
        if len(final_pairs) > best_count:
            best_pairs = final_pairs
    
    return [[r, m] for r, m in best_pairs]

# EVOLVE-BLOCK-END
