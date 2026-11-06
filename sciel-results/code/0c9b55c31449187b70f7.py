import math
import time
import random

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Pre-compute gcds
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0

    # Heuristic ordering: moduli with more prime factors are more constrained
    # Sort by number of distinct prime factors (ascending means easier first)
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Try multiple orderings
    orderings = [
        sorted(all_mods),  # Ascending
        sorted(all_mods, reverse=True),  # Descending
        sorted(all_mods, key=lambda x: count_prime_factors(x)),  # Fewer prime factors first
        sorted(all_mods, key=lambda x: count_prime_factors(x), reverse=True),  # More prime factors first
    ]

    def solve_with_ordering(moduli, t_limit, seed_val):
        """
        Greedy backtracking solver with limited depth.
        Returns list of (residue, modulus) pairs or None if failed.
        """
        random.seed(seed_val)
        n = len(moduli)
        if n == 0:
            return []
        
        residues = [None] * n
        node_count = 0
        max_nodes = 50000  # Limit to avoid exponential blowup
        
        def backtrack(idx):
            nonlocal node_count
            if time.time() > t_limit or node_count > max_nodes:
                return False
            if idx == n:
                return True
            
            node_count += 1
            m = moduli[idx]
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                valid = True
                for j in range(idx):
                    if not fast_disjoint(r, m, residues[j], moduli[j]):
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
                    # If we've tried many residues and failed, give up on this branch
                    if node_count > max_nodes:
                        return False
            return False
        
        if backtrack(0):
            return [(residues[i], moduli[i]) for i in range(n)]
        return None

    def greedy_construct(moduli, t_limit, seed_val):
        """
        Greedy construction without backtracking.
        Returns list of (residue, modulus) pairs.
        """
        random.seed(seed_val)
        n = len(moduli)
        result = []
        
        for m in moduli:
            if time.time() > t_limit:
                break
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                valid = True
                for (r2, m2) in result:
                    if not fast_disjoint(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    result.append((r, m))
                    break
        
        return result

    def local_search_refine(pairs, t_limit, seed_val):
        """
        Local search to add more pairs or improve existing ones.
        pairs is a list of (residue, modulus).
        Returns refined list.
        """
        random.seed(seed_val)
        n = len(pairs)
        if n == 0:
            return pairs
        
        # Convert to mutable structures
        residues = [p[0] for p in pairs]
        moduli = [p[1] for p in pairs]
        
        def count_valid():
            return len(residues)
        
        best_count = count_valid()
        best_residues = residues[:]
        best_moduli = moduli[:]
        
        while time.time() < t_limit:
            # Try to add a new modulus
            used_moduli = set(moduli)
            available_moduli = [m for m in all_mods if m not in used_moduli]
            
            if available_moduli:
                random.shuffle(available_moduli)
                added = False
                for m_new in available_moduli:
                    residues_to_try = list(range(m_new))
                    random.shuffle(residues_to_try)
                    
                    for r_new in residues_to_try:
                        valid = True
                        for i in range(len(residues)):
                            if not fast_disjoint(r_new, m_new, residues[i], moduli[i]):
                                valid = False
                                break
                        if valid:
                            residues.append(r_new)
                            moduli.append(m_new)
                            added = True
                            break
                    if added:
                        break
            
            if not added:
                # Try to reassign an existing modulus to free up space
                if not residues:
                    break
                
                idx = random.randint(0, len(residues) - 1)
                m = moduli[idx]
                old_r = residues[idx]
                residues_to_try = list(range(m))
                random.shuffle(residues_to_try)
                
                moved = False
                for new_r in residues_to_try:
                    if new_r == old_r:
                        continue
                    valid = True
                    for j in range(len(residues)):
                        if j == idx:
                            continue
                        if not fast_disjoint(new_r, m, residues[j], moduli[j]):
                            valid = False
                            break
                    if valid:
                        residues[idx] = new_r
                        moved = True
                        break
                
                if not moved:
                    # Try to remove a pair and see if we can add two
                    if len(residues) >= 2:
                        remove_idx = random.randint(0, len(residues) - 1)
                        saved_r = residues.pop(remove_idx)
                        saved_m = moduli.pop(remove_idx)
                        
                        # Try to add two new moduli
                        used_moduli = set(moduli)
                        available_moduli = [m for m in all_mods if m not in used_moduli]
                        random.shuffle(available_moduli)
                        
                        added_count = 0
                        for m_new in available_moduli[:5]:  # Limit to 5 attempts
                            residues_to_try = list(range(m_new))
                            random.shuffle(residues_to_try)
                            for r_new in residues_to_try:
                                valid = True
                                for i in range(len(residues)):
                                    if not fast_disjoint(r_new, m_new, residues[i], moduli[i]):
                                        valid = False
                                        break
                                if valid:
                                    residues.append(r_new)
                                    moduli.append(m_new)
                                    added_count += 1
                                    break
                        
                        # Also try to re-add the saved one
                        if saved_m not in moduli:
                            residues_to_try = list(range(saved_m))
                            random.shuffle(residues_to_try)
                            for r_new in residues_to_try:
                                valid = True
                                for i in range(len(residues)):
                                    if not fast_disjoint(r_new, saved_m, residues[i], moduli[i]):
                                        valid = False
                                        break
                                if valid:
                                    residues.append(r_new)
                                    moduli.append(saved_m)
                                    added_count += 1
                                    break
                        
                        # Only keep if we improved
                        if len(residues) > best_count:
                            best_count = len(residues)
                            best_residues = residues[:]
                            best_moduli = moduli[:]
                        else:
                            # Revert
                            if added_count > 0:
                                # This is complex to revert properly, so just restart from best
                                residues = best_residues[:]
                                moduli = best_moduli[:]
                            else:
                                residues.insert(remove_idx, saved_r)
                                moduli.insert(remove_idx, saved_m)
            
            cur_count = count_valid()
            if cur_count > best_count:
                best_count = cur_count
                best_residues = residues[:]
                best_moduli = moduli[:]
        
        return [(best_residues[i], best_moduli[i]) for i in range(len(best_residues))]

    best_pairs = []
    best_len = 0
    
    # Phase 1: Try backtracking with different orderings
    for ordering in orderings:
        if time.time() > deadline - 20.0:
            break
        for seed in range(10):
            if time.time() > deadline - 15.0:
                break
            pairs = solve_with_ordering(ordering, deadline - 10.0, seed)
            if pairs is not None:
                if len(pairs) > best_len:
                    best_pairs = pairs
                    best_len = len(pairs)
                    if best_len == len(all_mods):
                        break
        if best_len == len(all_mods):
            break
    
    # Phase 2: If we didn't get a full solution, try greedy + local search
    if best_len < len(all_mods):
        for ordering in orderings:
            if time.time() > deadline - 5.0:
                break
            for seed in range(20):
                if time.time() > deadline - 2.0:
                    break
                pairs = greedy_construct(ordering, deadline - 1.0, seed)
                if len(pairs) > best_len:
                    best_pairs = pairs
                    best_len = len(pairs)
                
                # Local search refinement
                if time.time() < deadline - 1.0:
                    refined = local_search_refine(pairs, deadline - 0.5, seed)
                    if len(refined) > best_len:
                        best_pairs = refined
                        best_len = len(refined)
    
    if not best_pairs:
        # Fallback
        return [[0, 2]]
    
    result = []
    for (r, m) in best_pairs:
        result.append([r, m])
    
    return result
