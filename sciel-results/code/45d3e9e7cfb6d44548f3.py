import math
import time

MAX_MOD = 128
DEADLINE = 85.0  # Safety margin from 90s

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Precompute gcd table
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Build compatibility: for each pair of moduli, which residue pairs are compatible
    # compat[m1][m2] = set of (r1, r2) that are disjoint
    # But this is too large. Instead, for each m1, m2, precompute which residues of m1 are compatible with each residue of m2
    
    # Alternative: For the clique approach, we need a graph where vertices are (m, r)
    # Total vertices: sum(m for m in 2..128) = (2+128)*127/2 = 8215
    # That's manageable with bitsets
    
    # Let's use a different approach: backtracking with MRV and forward checking
    # But with a much better ordering and constraint propagation
    
    def solve_backtrack(moduli_order, t_limit):
        """
        moduli_order: list of moduli in the order we'll assign them
        Returns list of (residue, modulus) or None
        """
        n = len(moduli_order)
        assigned = {}  # modulus -> residue
        # For forward checking: for each unassigned modulus, track which residues are still valid
        # valid_residues[m] = set of residues r for m that are compatible with all assigned
        
        # Initialize valid residues
        valid_residues = {}
        for m in moduli_order:
            valid_residues[m] = set(range(m))
        
        def backtrack(idx):
            if time.time() > t_limit:
                return False
            if idx == n:
                return True
            
            m = moduli_order[idx]
            # MRV: find unassigned modulus with fewest valid residues
            # But we're going in order... let's just use the order
            # Actually, let's pick the one with fewest valid residues among remaining
            
            best_idx = idx
            best_count = len(valid_residues[m])
            for j in range(idx + 1, n):
                mj = moduli_order[j]
                if len(valid_residues[mj]) < best_count:
                    best_count = len(valid_residues[mj])
                    best_idx = j
                    if best_count == 0:
                        break
            
            if best_count == 0:
                return False
            
            # Swap to put best_idx at position idx
            if best_idx != idx:
                moduli_order[idx], moduli_order[best_idx] = moduli_order[best_idx], moduli_order[idx]
            
            m = moduli_order[idx]
            # Try residues, ordered by least constraint on others
            residues = list(valid_residues[m])
            
            # Score each residue by how many valid residues it eliminates from others
            def score_residue(r):
                eliminated = 0
                for j in range(idx + 1, n):
                    mj = moduli_order[j]
                    g = gcd_table[(m, mj)]
                    # Count how many residues in valid_residues[mj] are congruent to r mod g
                    # r mod g == x, so residues rj where rj % g == x are incompatible
                    x = r % g
                    count = 0
                    for rj in valid_residues[mj]:
                        if rj % g == x:
                            count += 1
                    eliminated += count
                return eliminated
            
            residues.sort(key=score_residue)
            
            for r in residues:
                assigned[m] = r
                # Forward checking: update valid residues for remaining moduli
                saved_valid = {}
                ok = True
                for j in range(idx + 1, n):
                    mj = moduli_order[j]
                    g = gcd_table[(m, mj)]
                    x = r % g
                    new_valid = set()
                    for rj in valid_residues[mj]:
                        if rj % g != x:
                            new_valid.add(rj)
                    saved_valid[mj] = valid_residues[mj]
                    valid_residues[mj] = new_valid
                    if not new_valid:
                        ok = False
                        break
                
                if ok and backtrack(idx + 1):
                    return True
                
                # Undo
                del assigned[m]
                for mj, sv in saved_valid.items():
                    valid_residues[mj] = sv
                if best_idx != idx:
                    moduli_order[idx], moduli_order[best_idx] = moduli_order[best_idx], moduli_order[idx]
            
            return False
        
        result = backtrack(0)
        if result:
            return [(assigned[m], m) for m in moduli_order]
        return None
    
    # Order moduli by "difficulty": moduli with many divisors / high gcd with others are harder
    # Heuristic: sort by number of moduli that share a gcd > 1 with them, descending
    def difficulty(m):
        count = 0
        for m2 in all_mods:
            if m2 != m and math.gcd(m, m2) > 1:
                count += 1
        return count
    
    moduli_order = sorted(all_mods, key=difficulty, reverse=True)
    
    best_pairs = []
    best_len = 0
    t_limit = t0 + 30.0
    
    # Try full set first
    pairs = solve_backtrack(moduli_order[:], t_limit)
    if pairs is not None:
        best_pairs = pairs
        best_len = len(pairs)
    
    # If not all, try removing moduli one by one
    if best_len < n_mods:
        # Greedy: start with all, try to solve. If fail, remove the "hardest" and retry
        # But this is expensive. Instead, use a different strategy.
        
        # Try a greedy construction with constraint propagation
        for seed in range(20):
            if time.time() > deadline - 1.0:
                break
            import random
            random.seed(seed)
            mods = all_mods[:]
            random.shuffle(mods)
            
            current_set = []
            for m in mods:
                # Try to find a residue for m compatible with all in current_set
                found = False
                for r in range(m):
                    ok = True
                    for (r2, m2) in current_set:
                        g = gcd_table[(m, m2)]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        current_set.append((r, m))
                        found = True
                        break
                if not found:
                    continue
            
            if len(current_set) > best_len:
                best_pairs = current_set
                best_len = len(current_set)
                if best_len == n_mods:
                    break
    
    if not best_pairs:
        best_pairs = [(0, 2)]
    
    return [[r, m] for r, m in best_pairs]
