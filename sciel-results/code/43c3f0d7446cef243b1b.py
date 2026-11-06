"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    max_mod = 128
    # Strategy: pick a prime p, use multiples of p in [2, max_mod],
    # assign distinct residues mod p. Count = min(p, floor(max_mod/p)).
    # Find p that maximizes min(p, floor(max_mod/p)).
    best_count = 0
    best_p = None
    for p in range(2, max_mod + 1):
        # Check if p is prime
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for d in range(2, int(math.isqrt(p)) + 1):
                if p % d == 0:
                    is_prime = False
                    break
        if not is_prime:
            continue
        count = min(p, max_mod // p)
        if count > best_count:
            best_count = count
            best_p = p
    
    p = best_p
    moduli = [p * k for k in range(1, max_mod // p + 1) if p * k <= max_mod]
    residues = list(range(len(moduli)))
    
    result = []
    for i, m in enumerate(moduli):
        r = residues[i]
        result.append([r, m])
    
    return result
# EVOLVE-BLOCK-END
