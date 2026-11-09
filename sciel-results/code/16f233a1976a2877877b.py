import time
import math
from fractions import Fraction

def _compatible(residues, m_new, r_new):
    """Check if residue r_new for modulus m_new is compatible with all assigned."""
    for m_old, r_old in residues.items():
        g = math.gcd(m_new, m_old)
        if r_new % g == r_old % g:
            return False
    return True

def _find_residue(residues, m_new):
    """Find a valid residue for m_new, or return None."""
    for r in range(m_new):
        if _compatible(residues, m_new, r):
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Greedy: try moduli in increasing order (smallest first for max 1/m)
    residues = {}
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = _find_residue(residues, m)
        if r is not None:
            residues[m] = r
    
    # Local improvement: try to add skipped moduli by removing conflicting ones
    # Try multiple passes
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Try adding each unplaced modulus
        for m in range(20, 201):
            if time.time() > deadline:
                break
            if m in residues:
                continue
            # Try to place m, possibly removing one conflicting modulus
            # First try direct placement
            r = _find_residue(residues, m)
            if r is not None:
                residues[m] = r
                improved = True
                continue
            # Try removing one modulus to make room
            for m2 in list(residues.keys()):
                if 1.0/m > 1.0/m2:  # only swap if m is better
                    # Try placing m with m2 removed
                    temp = {k: v for k, v in residues.items() if k != m2}
                    r2 = _find_residue(temp, m)
                    if r2 is not None:
                        residues[m] = r2
                        del residues[m2]
                        improved = True
                        break
    
        break
    
    # Convert to required format
    result = [[m, r] for m, r in sorted(residues.items())]
    return result
