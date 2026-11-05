"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import sys

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * (MAX_M + 1) for _ in range(MAX_M + 1)]
    for m1 in moduli:
        for m2 in moduli:
            gcds[m1][m2] = math.gcd(m1, m2)
            
    # Precompute masks for residues congruent to rem mod g for each modulus m
    mask_incompat = {}
    for m in moduli:
        mask_incompat[m] = {}
        divisors = []
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                divisors.append(d)
                if d != m // d:
                    divisors.append(m // d)
        for g in divisors:
            mask_incompat[m][g] = {}
            for rem in range(g):
                mask = 0
                for r in range(m):
                    if r % g == rem:
                        mask |= (1 << r)
                mask_incompat[m][g][rem] = mask

    # Order moduli
    constraint_deg = {}
    for m in moduli:
        deg = 0
        for m2 in moduli:
            if m != m2 and math.gcd(m, m2) > 1:
                deg += 1
        constraint_deg[m] = deg
    
    ordered_moduli = sorted(moduli, key=lambda m: (-constraint_deg[m], m))
    
    best_solution = []
    best_size = 0
    MAX_POSSIBLE = 64 # Upper bound based on clique size of multiples of 2
    
    def solve(idx, current_size, avail_masks, current_solution):
        nonlocal best_solution, best_size
        
        if current_size == MAX_POSSIBLE:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_solution)
            return

        # Pruning: Upper bound
        ub = current_size
        for i in range(idx, n):
            if avail_masks[i] != 0:
                ub += 1
        if ub <= best_size:
            return
            
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_solution)
            return
            
        m = ordered_moduli[idx]
        mask = avail_masks[idx]
        
        # Option 1: Skip this modulus
        solve(idx + 1, current_size, avail_masks, current_solution)
        
        # Option 2: Assign a residue to this modulus
        if mask != 0:
            m_mask = mask
            while m_mask:
                lsb = m_mask & (-m_mask)
                r = lsb.bit_length() - 1
                m_mask ^= lsb
                
                new_avail = list(avail_masks)
                new_avail[idx] = 0
                
                feasible = True
                for j in range(idx + 1, n):
                    m2 = ordered_moduli[j]
                    if new_avail[j] == 0:
                        continue
                    g = gcds[m][m2]
                    if g == 1:
                        new_avail[j] = 0
                    else:
                        rem = r % g
                        incompat = mask_incompat[m2][g][rem]
                        new_avail[j] &= ~incompat
                        
                solve(idx + 1, current_size + 1, new_avail, current_solution + [(m, r)])
                
    init_masks = [0] * n
    for i, m in enumerate(ordered_moduli):
        init_masks[i] = (1 << m) - 1
        
    solve(0, 0, init_masks, [])
    
    result = [[r, m] for m, r in best_solution]
    return result

# EVOLVE-BLOCK-END
