import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0

    best_classes = []
    best_count = 0

    # Strategy: For a common divisor g, select k moduli divisible by g
    # and assign distinct residues 0..k-1.
    # This ensures r_i != r_j mod g, and since g divides gcd(m_i, m_j),
    # r_i != r_j mod gcd(m_i, m_j) is satisfied.
    
    for g in range(2, 129):
        moduli = [m for m in range(2, 129) if m % g == 0]
        k = min(g, len(moduli))
        
        if k > best_count:
            # Select k moduli
            selected_moduli = moduli[:k]
            residues = list(range(k))
            
            classes = [[residues[i], selected_moduli[i]] for i in range(k)]
            
            # Verify
            valid = True
            n = len(classes)
            for i in range(n):
                for j in range(i + 1, n):
                    r1, m1 = classes[i]
                    r2, m2 = classes[j]
                    gcd_val = math.gcd(m1, m2)
                    if (r1 - r2) % gcd_val == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_classes = classes
                best_count = k

    # Fallback to seed if no better found (unlikely)
    if best_count == 0:
        best_classes = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return best_classes
