"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}

def construct():
    # The set of 18 integers with distinct subset products
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Powers of 2: 4, 8, 16, 32
    # Power of 3: 9 (dropped 27 to avoid 3,9,27 collision? No, 3,9 is safe. 3,27 is safe. 9,27 is safe. 3,9,27 is unsafe.)
    # Power of 5: 25
    
    # Let's verify 3, 9, 25, 4, 8, 16, 32, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Wait, if I include 3 and 9, I can't include 27.
    # If I include 3 and 27, I can't include 9.
    # If I include 9 and 27, I can't include 3.
    
    # Strategy: Keep all primes.
    # Add 4, 8, 16, 32.
    # Add 25.
    # Add 9. (3, 9 is safe).
    
    result = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        4, 8, 16, 32,
        25,
        9
    ]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
