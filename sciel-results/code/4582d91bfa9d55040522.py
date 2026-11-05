"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations, product

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i-1] = L / i
    w = [L // i for i in range(1, 31)]
    
    best_abs_T = None
    best_coeffs = None
    
    # Try support sizes from 2 up to 5
    for size in range(2, 6):
        for subset in combinations(range(30), size):
            # Try all sign assignments
            for signs in product([-1, 1], repeat=size):
                T = 0
                for idx, sign in zip(subset, signs):
                    T += sign * w[idx]
                if T == 0:
                    continue
                abs_T = abs(T)
                if best_abs_T is None or abs_T < best_abs_T:
                    best_abs_T = abs_T
                    coeffs = [0] * 30
                    for idx, sign in zip(subset, signs):
                        coeffs[idx] = sign
                    best_coeffs = coeffs
                    if abs_T == 1:
                        return coeffs
    
    return best_coeffs
# EVOLVE-BLOCK-END
