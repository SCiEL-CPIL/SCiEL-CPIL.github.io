import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Moduli from 2 to 128
    moduli = list(range(128, 1, -1))
    assignments = {}
    
    for m in moduli:
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            valid = True
            # Check against all previously assigned
            for m_prime, r_prime in assignments.items():
                g = math.gcd(m, m_prime)
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                assignments[m] = r
                found = True
                break
        # If not found, we could try to backtrack or just skip.
        # Given the analysis, it's highly likely to be found.
        
    # Convert to list of [residue, modulus]
    result = [[r, m] for m, r in assignments.items()]
    return result
