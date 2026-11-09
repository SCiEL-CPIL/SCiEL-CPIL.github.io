"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Use the 12 smallest positive zeros of J_{1/2}(x) = sin(sqrt(x))/sqrt(x)
    # These are the squares of the first 12 positive zeros of sin(x), i.e., (k*pi)^2 for k=1..12
    # Wait, J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x)).
    # So the zeros are at sqrt(x) = k*pi, so x = (k*pi)^2.
    
    # Let's verify:
    # J_{1/2}(x) = sqrt(2/(pi*x)) sin(sqrt(x))
    # Zeros when sin(sqrt(x)) = 0 => sqrt(x) = k*pi => x = k^2 * pi^2
    
    # The task says: "Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x)"
    # These are z_i = (i*pi)^2 for i = 1, ..., 12
    
    # Let's compute these values
    z = [float((i * math.pi) ** 2) for i in range(1, 13)]
    
    # They must be finite, > 0 and pairwise distinct
    # (i*pi)^2 for i=1..12 are all positive, finite, and distinct
    
    return z
# EVOLVE-BLOCK-END
