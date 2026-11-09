import math

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x),
    which are the roots of sin(sqrt(x))/sqrt(x). These are at x = (k*pi)^2 for k=1..12.
    This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure.
    """
    # The zeros of J_{1/2}(x) are at x = (k*pi)^2 for k = 1, 2, 3, ...
    # J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # So the zeros (excluding x=0) are at sqrt(x) = k*pi, i.e., x = (k*pi)^2
    
    z_list = []
    for k in range(1, 13):
        # Calculate z_k = (k * pi)^2
        val = (k * math.pi) ** 2
        z_list.append(float(val))
    
    # Ensure they are finite, > 0, and pairwise distinct
    # They are all positive and distinct by construction
    # Sort to be safe, though they are generated in order
    z_list.sort()
    
    return z_list
