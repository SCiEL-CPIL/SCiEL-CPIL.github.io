import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 60  # Safety margin for 90s limit
    
    # Initial best from seed logic: 1/29 - 1/30 = 1/870
    best_sum = Fraction(1, 870)
    best_coeff = [0]*28 + [1, -1]
    
    # Random search
    # We use Fraction to ensure exact arithmetic
    # To speed up, we can use a heuristic: 
    # Randomly assigning -1, 0, 1 is fine, but we can bias towards more non-zeros 
    # to have more terms cancel out.
    
    while time.time() < deadline:
        # Generate a random vector of length 30 with values in {-1, 0, 1}
        # Using a list comprehension for speed
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Calculate sum
        s = Fraction(0, 1)
        for i in range(30):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], i + 1)
                
        if s != 0:
            abs_s = abs(s)
            if abs_s < abs(best_sum):
                best_sum = s
                best_coeff = coeffs
                # Optional: print progress if debugging, but keep silent for production
                # print(f"New best: {abs(best_sum)}")

    return best_coeff

# EVOLVE-BLOCK-END
