"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Greedy approach: select moduli in ascending order, picking the smallest valid residue
    # This tends to capture small moduli which have higher 1/m weights
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
        # Find a residue r in [0, m) such that for all (m_i, r_i) in selected:
        # r % gcd(m, m_i) != r_i % gcd(m, m_i)
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
        if not found:
            continue
    
    # Convert to required format
    result = [[m, r] for m, r in selected]
    
    # If we have time, try local search improvements
    if time.time() < deadline - 10:
        # Try to improve by swapping residues or adding/removing moduli
        # Simple local search: try to find a better configuration
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding a modulus that was skipped
            current_mods = set(m for m, r in selected)
            for m in range(20, 201):
                if m in current_mods:
                    continue
                if time.time() > deadline - 5:
                    break
                for r in range(m):
                    valid = True
                    for (m_i, r_i) in selected:
                        g = gcd(m, m_i)
                        if r % g == r_i % g:
                            valid = False
                            break
                    if valid:
                        # Check if adding this improves the score
                        selected.append((m, r))
                        current_mods.add(m)
                        improved = True
                        break
                if improved:
                    break
    
    return [[m, r] for m, r in selected]
# EVOLVE-BLOCK-END
