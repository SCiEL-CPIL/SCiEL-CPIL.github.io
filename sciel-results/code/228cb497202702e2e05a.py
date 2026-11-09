import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    max_mod = 128
    # Sort moduli in descending order of "potential"
    # Heuristic: moduli with more factors or larger size might be harder to place?
    # Or maybe we should try to place large gcds first?
    # Let's try sorting by value descending.
    moduli = list(range(2, max_mod + 1))
    
    # We will try to build a set of (residue, modulus)
    # Start with empty set
    solution = []
    
    # Try adding moduli in a specific order
    # Order: try to add moduli that share large gcds with existing ones
    # A simple greedy: iterate through moduli, try to add if possible.
    # But order matters.
    
    # Let's try multiple orders.
    orders = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: -x.bit_count()), # more bits = more factors?
    ]
    
    best_solution = []
    
    for order in orders:
        current_solution = []
        for m in order:
            # Try to find a residue r for m
            found = False
            for r in range(m):
                valid = True
                for r_j, m_j in current_solution:
                    g = gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        valid = False
                        break
                if valid:
                    current_solution.append((r, m))
                    found = True
                    break
            
            if found:
                continue
            else:
                # If we can't add m, skip it?
                # Or try to rearrange?
                # For now, simple greedy.
                pass
        
        if len(current_solution) > len(best_solution):
            best_solution = current_solution
            
        if time.time() > deadline:
            break
            
    return [[r, m] for r, m in best_solution]
