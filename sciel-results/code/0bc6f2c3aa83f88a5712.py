import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    
    # Precompute gcd for each pair
    gcd_mat = [[math.gcd(all_moduli[i], all_moduli[j]) for j in range(n)] for i in range(n)]
    
    # For each pair (i,j), precompute the set of residues for modulus i that conflict with each residue of modulus j
    # conflict_mask[i][j][r_i] = bitmask of r_j values in modulus j that conflict with r_i
    # Actually, let's precompute: for pair (i,j) with gcd g, residue r_i conflicts with r_j iff (r_i - r_j) % g == 0
    
    # Better: for each modulus i, precompute for each other modulus j, which residues of i are compatible with which residues of j
    # Use bitsets for speed
    
    # For each pair (i, j), precompute a compatibility matrix as bitsets
    # compat[i][j] is a list of length all_moduli[i], where compat[i][j][r] is a bitmask of compatible residues in modulus j
    
    compat = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g = gcd_mat[i][j]
            if g == 1:
                # All residues compatible
                compat[i][j] = [(1 << all_moduli[j]) - 1] * all_moduli[i]
            else:
                compat[i][j] = []
                for r_i in range(all_moduli[i]):
                    mask = 0
                    for r_j in range(all_moduli[j]):
                        if (r_i - r_j) % g != 0:
                            mask |= (1 << r_j)
                    compat[i][j].append(mask)
    
    # Strategy: Backtracking with forward checking and MRV heuristic
    # We'll try to select as many moduli as possible
    
    best_result = []
    
    def backtrack(selected, available):
        """
        selected: list of (index, residue)
        available: bitmask of moduli still available to add
        """
        nonlocal best_result
        
        if len(selected) > len(best_result):
            best_result = list(selected)
        
        if time.time() > deadline:
            return
        
        if available == 0:
            return
        
        # MRV: pick the available modulus with fewest compatible residues given current selection
        best_idx = -1
        best_count = float('inf')
        best_residues = []
        
        avail_list = []
        for idx in range(n):
            if available & (1 << idx):
                avail_list.append(idx)
        
        for idx in avail_list:
            m = all_moduli[idx]
            # Count how many residues are compatible with all selected
            valid_mask = (1 << m) - 1
            for (sel_idx, sel_r) in selected:
                valid_mask &= compat[idx][sel_idx][sel_r]
            
            count = bin(valid_mask).count('1')
            if count == 0:
                continue
            if count < best_count:
                best_count = count
                best_idx = idx
                best_residues = valid_mask
                if count == 1:
                    break
        
        if best_idx == -1:
            return
        
        # Try residues in best_residues
        m = all_moduli[best_idx]
        # Order residues: try those that are most constraining first (fewest options for others)
        # Actually, just try all valid residues
        
        new_available = available & ~(1 << best_idx)
        
        for r in range(m):
            if not (best_residues & (1 << r)):
                continue
            
            # Forward check: verify that adding this doesn't make any other available modulus impossible
            new_selected = selected + [(best_idx, r)]
            
            feasible = True
            for other_idx in range(n):
                if not (new_available & (1 << other_idx)):
                    continue
                om = all_moduli[other_idx]
                other_valid = (1 << om) - 1
                for (sel_idx2, sel_r2) in new_selected:
                    other_valid &= compat[other_idx][sel_idx2][sel_r2]
                if other_valid == 0:
                    feasible = False
                    break
            
            if feasible:
                backtrack(new_selected, new_available)
                if time.time() > deadline:
                    return
    
    # Start backtracking
    # Initial available: all moduli
    all_available = (1 << n) - 1
    backtrack([], all_available)
    
    if best_result:
        return [[r, all_moduli[idx]] for (idx, r) in best_result]
    else:
        # Fallback: powers of 2
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
