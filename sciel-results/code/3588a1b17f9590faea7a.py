import math
import time

def construct():
    # Use the 12 smallest positive zeros associated with the asymptotic
    # Bessel behavior of the highest degree Laguerre polynomial L_50^(-1/2)(x).
    # The argument of the Bessel function is 2*sqrt(50*x).
    # Setting this to i*pi gives x = i^2 * pi^2 / 200.
    # This places the roots in the oscillatory region of the Laguerre polynomials
    # rather than the exponential tail, potentially improving the conditioning
    # and confining the quotient q(x)'s largest root B.
    deadline = time.time() + 600.0
    z = [(i * math.pi) ** 2 / 200.0 for i in range(1, 13)]
    result = sorted([float(zi) for zi in z])
    # Safety: verify constraints
    if any(not (math.isfinite(zi) and zi > 0) for zi in result):
        raise ValueError("Invalid nodes")
    if len(set(result)) != 12:
        raise ValueError("Nodes not pairwise distinct")
    return result
