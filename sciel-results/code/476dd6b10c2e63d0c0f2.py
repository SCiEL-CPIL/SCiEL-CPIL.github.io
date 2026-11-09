"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the 12 positive double-root locations for the Laguerre combination.
    Based on the Cohn-Goncalves construction, the optimal roots for the sign 
    uncertainty problem with m=12 double roots are approximated by the roots 
    of the Laguerre polynomial of degree 2m = 24.
    """
    m = 12
    # The construction involves 2m zeros in the Laguerre space context.
    # We compute the roots of L_{2m}(x).
    degree = 2 * m
    roots, _ = roots_laguerre(degree)
    
    # Ensure roots are positive and sorted (roots_laguerre returns sorted positive roots)
    # Filter out any numerical noise if present, though they should be clean.
    valid_roots = [float(r) for r in roots if r > 0]
    
    # If for some reason we don't have 12, fallback to a heuristic, 
    # but roots_laguerre(24) should always give 24 distinct positive roots.
    # We only need 12? 
    # Wait, the task says "12 positive double-root locations".
    # A function with 12 double roots has degree at least 24.
    # The roots of L_24 are 24 distinct simple roots.
    # Does the construction use ALL 24 roots as the "locations"?
    # "list of 12 positive double-root locations".
    # This implies the output list should have length 12.
    # If the function is f(x) = prod_{i=1}^{12} (x - x_i)^2, then the roots x_i are the locations.
    # In the Cohn-Goncalves construction for the Fourier uncertainty, 
    # the optimal function is often related to the Laguerre polynomial L_{2m}.
    # However, L_{2m} has 2m simple roots.
    # Perhaps the "double roots" refer to the fact that the uncertainty integral 
    # involves a squared term, or the construction pairs roots?
    
    # Let's re-read carefully: "12 positive double-root locations".
    # If I return 24 roots, it violates "list of 12".
    # If the optimal function is L_{12}(x)^2? No, L_{12} has 12 simple roots.
    # If the function is L_{24}(x), it has 24 simple roots.
    # If the construction forces double roots, maybe it's related to the derivative?
    
    # Let's look at the seed: [4.0 * (i + 1) for i in range(12)]. Length 12.
    # So I must return 12 values.
    
    # Hypothesis: The 12 locations are the roots of L_{12}(x)?
    # Or are they half of the roots of L_{24}(x)?
    # Or are they the roots of L_{12}(x) scaled?
    
    # In many uncertainty principles involving Laguerre polynomials, 
    # the extremal function is L_m(x) or similar.
    # If the problem specifies "m=12", it likely refers to the parameter m in the construction.
    # If the seed used 12 roots, and the task asks for 12 roots, 
    # and the seed used linear spacing, 
    # the most likely "natural" 12 roots are the roots of L_{12}(x).
    
    # Let's try roots of L_{12}(x).
    roots_12, _ = roots_laguerre(m)
    return [float(r) for r in roots_12]

# EVOLVE-BLOCK-END
