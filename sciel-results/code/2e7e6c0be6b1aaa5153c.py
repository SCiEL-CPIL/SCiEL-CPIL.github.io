"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel J_{1/2} zeros."""
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    def time_remaining():
        return deadline - time.time() - safety_margin

    # The 12 smallest positive zeros of J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # are at sqrt(x) = k*pi => x = k^2 * pi^2 for k = 1, ..., 12
    # This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure.
    # z_i = i^2 * pi^2 for i = 1, ..., 12
    
    z_bessel = [float(i**2 * math.pi**2) for i in range(1, 13)]
    
    # Ensure positive, finite, distinct
    result = []
    for z in z_bessel:
        if not math.isfinite(z) or z <= 0:
            return [4.0 * (i + 1) for i in range(12)]
        result.append(z)
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                return [4.0 * (i + 1) for i in range(12)]
    
    return result
# EVOLVE-BLOCK-END
