"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_disjoint(r1, m1, r2, m2):
    """Check if residue class r1 mod m1 is disjoint from r2 mod m2."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_addition(r, m, current_set):
    """Check if (r, m) is disjoint from all pairs in current_set."""
    for r2, m2 in current_set:
        if not _is_disjoint(r, m, r2, m2):
            return False
    return True

def _compute_constraint_scores():
    """
    Compute a score for each modulus based on its 'conflict potential' with other moduli.
    Higher score means more conflicts, so we should process these first.
    Score is sum of gcd(m, k) for all k != m in [2, 128].
    """
    scores = {}
    for m in range(2, 129):
        score = 0
        for k in range(2, 129):
            if m != k:
                score += math.gcd(m, k)
        scores[m] = score
    return scores

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses a Deterministic Greedy Algorithm with a Constraint-First Ordering.
    Moduli with higher 'conflict potential' (sum of gcds with others) are processed first.
    """
    # 1. Compute constraint scores
    scores = _compute_constraint_scores()
    
    # 2. Sort moduli by score descending (most constrained first)
    # In case of ties, we can sort by modulus value or prime factors, but score is primary.
    moduli = list(range(2, 129))
    moduli.sort(key=lambda m: (-scores[m], m))
    
    chosen = []
    
    # 3. Greedy selection
    for m in moduli:
        # Try residues in a deterministic order.
        # Does residue order matter? 
        # For a fixed set of existing constraints, if one residue works, all valid residues are equivalent for the count.
        # However, the choice of residue might affect future additions? 
        # No, because the disjointness condition only depends on the modulus m and the residue r relative to others.
        # Wait, the condition is (r - r_j) % gcd(m, m_j) != 0.
        # Different r values satisfy this for different subsets of j.
        # Actually, for a fixed m and a fixed set of existing (r_j, m_j), the set of valid r is fixed.
        # Any valid r works to add the class. The specific r chosen doesn't change the fact that m is now "used".
        # So we just need to find *any* valid r.
        # We can iterate r from 0 to m-1.
        
        found = False
        for r in range(m):
            if _is_valid_addition(r, m, chosen):
                chosen.append((r, m))
                found = True
                break
        
        # If no residue works, we skip this modulus.
    
    return [[r, m] for r, m in chosen]

# EVOLVE-BLOCK-END
