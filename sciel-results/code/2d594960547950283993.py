import time
from math import gcd

def construct():
    """
    Finds maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two classes (r1,m1) and (r2,m2) are disjoint iff (r1-r2) % gcd(m1,m2) != 0.
    
    Strategy: Branch and bound with bitset domains, MRV heuristic, and LGV ordering.
    """
    start_time = time.time()
    deadline = start_time + 85
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    # Precompute forbidden masks: for variable j and gcd value g,
    # masks[j][g][k] = bitmask of values x in [0, m_j) where x % g == k
    masks = {}
    for j in range(n):
        m_j = all_moduli[j]
        masks[j] = {}
        for i in range(n):
            if i == j:
                continue
            g = gcd_mat[i][j]
            if g <= 1:
                continue
            if g not in masks[j]:
                masks[j][g] = []
                for k in range(g):
                    mask = 0
                    for x in range(m_j):
                        if x % g == k:
                            mask |= (1 << x)
                    masks[j][g].append(mask)
    
    best_size = 0
    best_assignment = {}
    
    def popcount(x):
        return bin(x).count('1')
    
    def backtrack(assigned, domains, depth):
        nonlocal best_size, best_assignment
        
        if time.time() > deadline:
            return
        
        if depth > best_size:
            best_size = depth
            best_assignment = dict(assigned)
        
        # MRV: pick unassigned variable with fewest valid residues
        min_count = float('inf')
        min_var = -1
        for j in range(n):
            if j in assigned:
                continue
            c = popcount(domains[j])
            if c == 0:
                return
            if c < min_count:
                min_count = c
                min_var = j
        
        if min_var == -1:
            return
        
        # Get valid residues for min_var
        domain_mask = domains[min_var]
        valid_rs = []
        mask = domain_mask
        while mask:
            lsb = mask & (-mask)
            r = lsb.bit_length() - 1
            valid_rs.append(r)
            mask ^= lsb
        
        # Sort by least constraining value
        def lgv_score(r):
            score = 0
            for j in range(n):
                if j == min_var or j in assigned:
                    continue
                g = gcd_mat[min_var][j]
                if g > 1:
                    score += all_moduli[j] // g
            return score
        
        valid_rs.sort(key=lgv_score)
        
        for r in valid_rs:
            if time.time() > deadline:
                return
            
            old_domains = list(domains)
            assigned[min_var] = r
            
            feasible = True
            for j in range(n):
                if j == min_var or j in assigned:
                    continue
                g = gcd_mat[min_var][j]
                if g > 1:
                    k = r % g
                    forbidden = masks[j][g][k]
                    domains[j] &= ~forbidden
                    if domains[j] == 0:
                        feasible = False
                        break
            
            if feasible:
                backtrack(assigned, domains, depth + 1)
            
            del assigned[min_var]
            domains = old_domains
    
    initial_domains = [(1 << all_moduli[i]) - 1 for i in range(n)]
    backtrack({}, initial_domains, 0)
    
    if not best_assignment:
        return [[0, 2]]
    
    result = [[best_assignment[idx], all_moduli[idx]] for idx in best_assignment]
    return result

# EVOLVE-BLOCK-END
