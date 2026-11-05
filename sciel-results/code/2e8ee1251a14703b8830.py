import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
            
    # Map modulus value to index
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    def check_residue(idx, r, selected):
        """
        Check if residue r for modulus idx is compatible with all selected.
        selected is a list of (idx2, r2).
        """
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_add_modulus(selected, selected_idx_set, target_idx):
        """
        Try to add target_idx to selected by finding a valid residue.
        Returns True if added, False otherwise.
        """
        m = moduli[target_idx]
        # Try residues in a random order to avoid bias
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if check_residue(target_idx, r, selected):
                selected.append((target_idx, r))
                selected_idx_set.add(target_idx)
                return True
        return False

    def try_greedy(order):
        selected = []
        selected_idx_set = set()
        for idx in order:
            try_add_modulus(selected, selected_idx_set, idx)
        return selected

    best = []
    best_len = 0
    
    # 1. Deterministic Greedy Initializations
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        # Prefer moduli with many small prime factors (more constrained)
        sorted(range(n), key=lambda i: (-len([p for p in [2,3,5,7,11,13,17,19,23] if moduli[i] % p == 0]), moduli[i])),
        # Prefer prime moduli (less constrained)
        sorted(range(n), key=lambda i: (len([p for p in [2,3,5,7,11,13,17,19,23] if moduli[i] % p == 0]), moduli[i])),
    ]
    
    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # 2. Random Restart Greedy
    random.seed(42)
    for restart in range(2000):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # 3. Local Search with Residue Re-assignment
    # We work on a copy of the best solution found so far
    # Strategy:
    # 1. Try to add any unselected modulus.
    # 2. If fail, pick a random unselected modulus, find why it fails (conflicts),
    #    and try to change the residue of a conflicting selected modulus to resolve it.
    
    def local_search_improve(initial_selected):
        selected = list(initial_selected)
        selected_idx_set = set(idx for idx, _ in selected)
        unselected = [i for i in range(n) if i not in selected_idx_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Try to add any unselected modulus
            if unselected:
                # Shuffle to pick a random candidate
                random.shuffle(unselected)
                for target_idx in unselected:
                    if try_add_modulus(selected, selected_idx_set, target_idx):
                        unselected.remove(target_idx)
                        improved = True
                        break
            
            # Phase 2: If no direct addition, try to unblock
            if not improved and unselected:
                # Pick a random target to unblock
                target_idx = random.choice(unselected)
                m_target = moduli[target_idx]
                
                # Find all residues for target that are blocked
                # For each residue r_t, find the set of conflicting indices
                # We want to find an r_t and a conflicting index i such that we can change r_i
                # to make r_t valid.
                
                # Optimization: Just try a few random residues for target
                candidates = list(range(m_target))
                random.shuffle(candidates)
                
                found_fix = False
                for r_t in candidates:
                    # Check conflicts
                    conflicts = []
                    for (idx_i, r_i) in selected:
                        g = gcd_tab[target_idx][idx_i]
                        if (r_t - r_i) % g == 0:
                            conflicts.append((idx_i, r_i))
                    
                    if not conflicts:
                        # Should have been added in Phase 1, but just in case
                        selected.append((target_idx, r_t))
                        selected_idx_set.add(target_idx)
                        if target_idx in unselected:
                            unselected.remove(target_idx)
                        improved = True
                        found_fix = True
                        break
                    
                    # Pick a random conflict to try to resolve
                    # We can try to change the residue of idx_i to something else
                    # that is compatible with all others AND not conflicting with r_t
                    idx_i, r_i = random.choice(conflicts)
                    m_i = moduli[idx_i]
                    
                    # Try to find a new residue r_i' for idx_i
                    # Constraints:
                    # 1. r_i' != r_i (implied if we find a valid one that works)
                    # 2. r_i' compatible with all j in selected \ {idx_i}
                    # 3. r_i' NOT conflicting with r_t for target_idx
                    #    i.e., (r_i' - r_t) % gcd(m_i, m_target) != 0
                    
                    g_it = gcd_tab[idx_i][target_idx]
                    
                    # Generate candidate residues for idx_i
                    # Exclude residues that conflict with r_t
                    invalid_residues = set()
                    for r in range(m_i):
                        if (r - r_t) % g_it == 0:
                            invalid_residues.add(r)
                    
                    # Also exclude residues that conflict with other selected moduli
                    # We can just iterate and check
                    valid_new_r = None
                    residues_to_try = list(range(m_i))
                    random.shuffle(residues_to_try)
                    
                    for r_new in residues_to_try:
                        if r_new in invalid_residues:
                            continue
                        # Check compatibility with all other selected
                        ok = True
                        for (idx_j, r_j) in selected:
                            if idx_j == idx_i:
                                continue
                            g_ij = gcd_tab[idx_i][idx_j]
                            if (r_new - r_j) % g_ij == 0:
                                ok = False
                                break
                        if ok:
                            valid_new_r = r_new
                            break
                    
                    if valid_new_r is not None:
                        # Update selected
                        for k in range(len(selected)):
                            if selected[k][0] == idx_i:
                                selected[k] = (idx_i, valid_new_r)
                                break
                        # Add target
                        selected.append((target_idx, r_t))
                        selected_idx_set.add(target_idx)
                        if target_idx in unselected:
                            unselected.remove(target_idx)
                        improved = True
                        found_fix = True
                        break
                
                if found_fix:
                    continue

            # Phase 3: If stuck, perform a random kick
            # Change a random residue to a random valid value
            # Or swap a modulus
            if selected:
                # Randomly change a residue
                pos = random.randint(0, len(selected) - 1)
                idx_i, r_i = selected[pos]
                m_i = moduli[idx_i]
                
                # Find a valid new residue
                # Try random residues
                valid_new_r = None
                residues_to_try = list(range(m_i))
                random.shuffle(residues_to_try)
                for r_new in residues_to_try:
                    if r_new == r_i:
                        continue
                    ok = True
                    for (idx_j, r_j) in selected:
                        if idx_j == idx_i:
                            continue
                        g_ij = gcd_tab[idx_i][idx_j]
                        if (r_new - r_j) % g_ij == 0:
                            ok = False
                            break
                    if ok:
                        valid_new_r = r_new
                        break
                
                if valid_new_r is not None:
                    selected[pos] = (idx_i, valid_new_r)
                    # Don't mark as improved, just kick
                else:
                    # If no valid residue change, maybe remove a random modulus?
                    # This is risky, might reduce count. Only do if count is small?
                    # For now, just break if stuck to avoid infinite loop of useless kicks
                    pass

        return selected

    # Run local search on the best solution found
    if time.time() < deadline - 10:
        improved_best = local_search_improve(best)
        if len(improved_best) > best_len:
            best = improved_best
            best_len = len(best)

    # Try local search on a few random good solutions too?
    # The current best is likely good enough.
    
    return [[r, moduli[idx]] for (idx, r) in best]
