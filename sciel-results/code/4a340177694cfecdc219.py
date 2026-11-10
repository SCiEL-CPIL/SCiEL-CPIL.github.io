"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n = len(moduli)
    
    # Precompute gcd
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_table[i][j] = math.gcd(moduli[i], moduli[j])
    
    # Greedy with MRV-like heuristic
    # Process moduli that are most constrained first
    
    def get_constraint_info(m_idx):
        """Return list of (other_idx, gcd) for others with gcd > 1"""
        return [(j, gcd_table[m_idx][j]) for j in range(n) if j != m_idx and gcd_table[m_idx][j] > 1]
    
    # Sort by number of constraints (descending) - hardest first
    order = sorted(range(n), key=lambda i: -len(get_constraint_info(i)))
    
    assigned = {}  # mod_idx -> residue
    result_pairs = []
    
    def is_compatible(res, m_idx):
        for m_c_idx, r_c in assigned.items():
            g = gcd_table[m_idx][m_c_idx]
            if g > 1 and (res - r_c) % g == 0:
                return False
        return True
    
    # Try multiple orderings to find a good solution
    best_count = 0
    best_result = []
    
    # Ordering 1: most constrained first
    orderings = [
        order,  # most constrained first
        list(range(n)),  # natural order
        sorted(range(n), key=lambda i: moduli[i]),  # by modulus value
        sorted(range(n), key=lambda i: -moduli[i]),  # reverse modulus
    ]
    
    for ord_idx, order in enumerate(orderings):
        if time.time() > deadline:
            break
        
        assigned = {}
        result_pairs = []
        
        for m_idx in order:
            m = moduli[m_idx]
            found = False
            for res in range(m):
                if is_compatible(res, m_idx):
                    assigned[m_idx] = res
                    result_pairs.append((res, m_idx))
                    found = True
                    break
        
        if len(result_pairs) > best_count:
            best_count = len(result_pairs)
            best_result = list(result_pairs)
    
    # Convert to required format
    result = [[res, moduli[m_idx]] for (res, m_idx) in best_result]
    result.sort(key=lambda x: x[1])
    
    return result if result else [[0, 2]]
# EVOLVE-BLOCK-END
