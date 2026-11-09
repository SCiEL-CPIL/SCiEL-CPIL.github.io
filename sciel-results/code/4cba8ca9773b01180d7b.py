import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # All possible moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # We'll use a greedy approach: try to add moduli in order of decreasing 1/m
    # i.e., increasing m is NOT the order; we want smallest m first for largest 1/m
    # But we need to check disjointness constraints
    
    def can_assign(moduli_list, residues_list, new_m):
        """Check if there exists a residue r for new_m such that it's disjoint from all existing."""
        # For each existing (m_j, r_j), the constraint is: r_new != r_j mod gcd(new_m, m_j)
        # We need to find r_new in [0, new_m) that satisfies all constraints.
        # Use CRT-like reasoning: the forbidden residues form a union of residue classes.
        # For each j, the forbidden residues are those r where r ≡ r_j (mod gcd(new_m, m_j))
        # So r mod gcd(new_m, m_j) != r_j mod gcd(new_m, m_j)
        
        # We'll check all possible residues 0 to new_m-1
        for r_new in range(new_m):
            ok = True
            for j in range(len(moduli_list)):
                m_j = moduli_list[j]
                r_j = residues_list[j]
                g = gcd(new_m, m_j)
                if r_new % g == r_j % g:
                    ok = False
                    break
            if ok:
                return True, r_new
        return False, None
    
    def score(moduli_list):
        total = Fraction(0, 1)
        for m in moduli_list:
            total += Fraction(1, m)
        return total
    
    # Greedy: try moduli in order of increasing m (largest 1/m first)
    best_moduli = []
    best_residues = []
    
    for m in sorted(moduli, key=lambda x: x):  # Increasing m = decreasing 1/m
        if time.time() > deadline:
            break
        can, r = can_assign(best_moduli, best_residues, m)
        if can:
            best_moduli.append(m)
            best_residues.append(r)
    
    # Build the result
    result = [[best_moduli[i], best_residues[i]] for i in range(len(best_moduli))]
    
    # Verify validity
    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i+1, n):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    if not is_valid(result):
        # Fallback to the seed solution
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    return result
