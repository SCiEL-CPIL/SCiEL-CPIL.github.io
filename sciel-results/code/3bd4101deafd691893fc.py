import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Strategy: Use multiples of a prime p.
    # Max size is min(p, floor(128/p)).
    # Best is p=11, size 11.
    
    best_result = None
    best_size = 0
    
    # Try primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for p in primes:
        if time.time() > deadline:
            break
            
        multiples = [m for m in range(2, 129) if m % p == 0]
        # We can pick at most p residues
        k = min(p, len(multiples))
        
        # Select the first k multiples (smallest moduli often easier? or doesn't matter for existence)
        # Actually, any k multiples work if we assign distinct residues mod p.
        selected_moduli = multiples[:k]
        
        # Assign residues 0, 1, ..., k-1
        # Check if valid
        result = []
        valid = True
        for i, m in enumerate(selected_moduli):
            r = i
            result.append([r, m])
            
        # Verify disjointness
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                r1, m1 = result[i]
                r2, m2 = result[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    # Fallback if no valid set found (should not happen for p=11)
    if best_result is None:
        return [[0, 2]]
        
    return best_result
