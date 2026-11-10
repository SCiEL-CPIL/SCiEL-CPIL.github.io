import math
import random

def construct():
    moduli = list(range(2, 129))
    
    # We want to maximize the count.
    # We'll use a greedy approach with backtracking for the first few, 
    # or just a simple greedy if it works well enough.
    # Given the small size, a simple backtracking to find a large set is feasible.
    
    # Sort moduli. 
    # Heuristic: Try to add moduli that share large GCDs with existing ones first?
    # Or just try all orders?
    
    # Let's try a greedy insertion.
    # Start with a known good set: powers of 2.
    # Then try to insert others.
    
    current_pairs = []
    # Initialize with powers of 2
    for k in range(1, 8):
        m = 1 << k
        r = 1 << (k-1)
        current_pairs.append([r, m])
        
    # Try to add other moduli
    # Sort remaining moduli
    remaining = [m for m in moduli if m not in [p[1] for p in current_pairs]]
    
    # To improve chances, sort remaining by descending value? Or random?
    # Let's try sorting by descending value to get large GCDs with large existing moduli?
    remaining.sort(reverse=True)
    
    for m in remaining:
        # Check if we can find a residue r in [0, m)
        # such that for all [r_j, m_j] in current_pairs:
        # r != r_j mod gcd(m, m_j)
        
        found = False
        # Optimize: Check residues in a smart order?
        # Just iterate 0..m-1
        for r in range(m):
            valid = True
            for r_j, m_j in current_pairs:
                g = math.gcd(m, m_j)
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                current_pairs.append([r, m])
                found = True
                break
        
    return current_pairs
