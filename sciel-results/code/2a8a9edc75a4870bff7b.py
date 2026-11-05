"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    """Construct a maximal set of pairwise disjoint residue classes with distinct moduli 2..128."""
    deadline = time.time() + 85.0  # 90s wall-clock, 5s safety margin
    
    max_mod = 128
    # Precompute gcd for all pairs of moduli in 2..128
    all_moduli = list(range(2, max_mod + 1))
    
    # For each pair (m_i, m_j), we need gcd(m_i, m_j) to determine compatibility
    # Two classes (r_i, m_i) and (r_j, m_j) are disjoint iff gcd(m_i, m_j) does NOT divide (r_i - r_j)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Greedy approach: try to include as many moduli as possible
    # Key insight: if we can pick residues to make all pairs disjoint, we want to include as many moduli as possible
    
    # Strategy: Process moduli in a smart order. 
    # For a new modulus m, given existing classes (r_i, m_i), we need to find r such that
    # for all i: gcd(m, m_i) does not divide (r - r_i)
    # This is a system of congruence-like constraints.
    # For each existing class i, the forbidden residues mod gcd(m, m_i) are: r ≡ r_i (mod gcd(m, m_i))
    # So r must avoid specific residue classes modulo g_i = gcd(m, m_i)
    
    # We can use inclusion-exclusion or just try all residues 0..m-1 and check
    # Since m <= 128, trying all m residues is cheap
    
    def find_valid_residue(m, existing):
        """Find a residue r in 0..m-1 that is disjoint from all existing classes."""
        for r in range(m):
            valid = True
            for (r_i, m_i) in existing:
                g = gcd(m, m_i)
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    # Try multiple orderings to find a good set
    # Order 1: ascending moduli
    # Order 2: descending moduli
    # Order 3: random orderings
    
    best_set = []
    best_size = 0
    
    # Deterministic orders first
    orders = [
        all_moduli,          # ascending
        list(reversed(all_moduli)),  # descending
    ]
    
    # Add some random orders
    import random
    random.seed(42)
    for _ in range(50):
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        current = []
        for m in order:
            r = find_valid_residue(m, current)
            if r is not None:
                current.append((r, m))
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Try local improvement: try to swap or add moduli
    # Convert to list of [residue, modulus] pairs for return
    result = [[r, m] for (r, m) in best_set]
    
    if not result:
        # Fallback: the original seed program
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result

# EVOLVE-BLOCK-END
