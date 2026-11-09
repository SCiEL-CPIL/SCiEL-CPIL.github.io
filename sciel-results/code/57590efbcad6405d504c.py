<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

import numpy as np
from sympy import (
    symbols, Rational, L, solve, Matrix, real_roots, Poly,
    sqrt, pi, cancel, factor
)

def _construct():
    # Return 12 double-root locations z_1..z_12
    # We want to minimize B/(2*pi), where B is the largest real root of q(x)
    # with a sign change at 1e-198 scale.
    # 
    # The seed returns [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    # Let's try to find better configurations.
    
    # Strategy: Try to make the roots of q(x) as small as possible.
    # One approach is to place the z_i to minimize the largest root of q.
    
    # Let's try a few candidate configurations:
    # 1. Geometric progression
    # 2. Optimized via simple heuristics
    # 3. Symmetric around some center
    
    # For now, let's try a geometric sequence that might give smaller B
    # The key insight is that we want the Laguerre polynomial combination
    # to have zeros at the z_i, and the remaining polynomial q(x) to have
    # small roots.
    
    # Let's try: z_i = a * r^(i-1) for i=1..12
    # We need to choose a and r to minimize B.
    
    # Try several configurations and pick the best one that's computationally feasible.
    
    # Actually, since we can't easily evaluate B without doing the full computation,
    # let's try a configuration that's likely to give small B.
    
    # One idea: place roots close together near the origin to keep q(x) small.
    # But they must be distinct and > 0.
    
    # Let's try: z_i = i * 0.5 for i = 1..12
    # This gives [0.5, 1.0, 1.5, ..., 6.0]
    
    # Or maybe even smaller: z_i = i * 0.1
    # But we need to be careful about numerical issues.
    
    # Let's try a few options and return the one that seems best.
    # Without being able to evaluate B directly, let's use a heuristic.
    
    # The seed uses [4, 8, 12, ..., 48] with sum = 4*(1+2+...+12) = 4*78 = 312
    # The product of (x-z_i)^2 will be large, affecting q(x).
    
    # Let's try: z_i = i * 0.5 for i=1..12
    # Sum = 0.5 * 78 = 39, much smaller.
    
    # Actually, let's think about this more carefully.
    # g(x) = sum_{k=0}^{25} c_k L_{2k}(x)
    # g has double roots at z_1, ..., z_12
    # So g(x) = (x * prod_i (x-z_i)^2) * q(x)
    # where q(x) is a polynomial of degree 50 - 1 - 24 = 25.
    
    # B is the largest real root of q(x) with a sign change.
    # To minimize B, we want q(x) to have its largest root as small as possible.
    
    # The coefficients c_k are determined by the conditions:
    # g(0) = 0, g'(0) = 1, and g(z_i) = g'(z_i) = 0 for all i.
    # That's 2 + 24 = 26 conditions for 26 unknowns.
    
    # The structure of q(x) depends on the z_i.
    
    # Let's try a configuration with smaller z_i values.
    
    # Option 1: z_i = i * 0.5
    # Option 2: z_i = 0.5 * (1.2)^(i-1)  (geometric)
    # Option 3: z_i = i * 0.1
    
    # Let's go with z_i = i * 0.5 as a reasonable improvement.
    
    return [i * 0.5 for i in range(1, 13)]

def construct():
    """Return 12 double-root locations z_1..z_12."""
    return _construct()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
