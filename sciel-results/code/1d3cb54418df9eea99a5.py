"""Compact dark-energy growth approximation: six-coefficient gamma model."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Six coefficients for gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    #
    # We aim to match the constant-gamma seed (0.55) as a baseline but improve
    # by allowing mild dependence on q and u to better capture the evolution
    # of the growth index across different CPL dark-energy models.
    #
    # Strategy: Start near the known good constant value and add small
    # corrections that are physically motivated:
    # - c1: linear correction in q (matter density deficit)
    # - c3: small correction in u (dark energy equation of state)
    # - c2, c4, c5: higher-order terms kept small for stability
    #
    # The constant gamma=0.55 is a well-known approximation for LambdaCDM.
    # For CPL models, the growth index varies mildly with w0 and wa.
    # A reasonable improvement is to let gamma decrease slightly as q increases
    # (more dark energy dominance) and adjust for w0.
    #
    # After analysis, a good set of coefficients that stays within [0.1, 1.2]
    # for typical validation cosmologies (w0 in [-1.5, -0.5], wa in [-1, 1])
    # is:
    c0 = 0.545      # baseline gamma for LambdaCDM (w0=-1, wa=0, u=0)
    c1 = 0.0        # no q dependence to keep LCDM gamma constant
    c2 = 0.0        # no q^2 dependence
    c3 = 0.05       # slope for w0 dependence (gamma increases as w0 increases)
    c4 = 0.0        # no wa*a term, wa effect vanishes at a=1
    c5 = 0.0        # no cross term
    
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
