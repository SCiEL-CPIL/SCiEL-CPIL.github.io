"""Improved growth index approximation using a compact ansatz."""
# EVOLVE-BLOCK-START
def construct():
    # Coefficients for gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # Tuned so that gamma ~ 0.55 for Lambda CDM (u=0) across q in [0.7, 0.95],
    # and gamma shifts appropriately with u = 1+w0+wa*(1-a).
    # For u=0 (Lambda): gamma = c0 + c1*q + c2*q^2
    #   At q=0.7: 0.55 = c0 + 0.7*c1 + 0.49*c2
    #   At q=0.85: 0.55 = c0 + 0.85*c1 + 0.7225*c2
    #   At q=0.95: 0.55 = c0 + 0.95*c1 + 0.9025*c2
    # Solving: c0=0.55, c1=0, c2=0 gives constant 0.55.
    # But gamma does vary slightly; using a small slope helps for non-Lambda.
    # For u != 0, gamma shifts by ~0.11*(u-1) for small w deviations.
    # So gamma ~ 0.55 + 0.11*(u-1) = 0.44 + 0.11*u
    # Combined: gamma = 0.55 + 0.11*(u - 0) but we need to be careful.
    # Let's use: gamma = c0 + c1*q + c3*u
    #   c0 = 0.55, c1 = 0.0, c3 = 0.11
    #   At u=0, q=0.7: 0.55
    #   At u=0, q=0.95: 0.55
    #   At u=1 (w=0), q=0.7: 0.55 + 0.11 = 0.66
    #   At u=0.5, q=0.85: 0.55 + 0.055 = 0.605
    # This seems reasonable. Let's also add a small q-dependence to capture
    # the slight variation in Lambda CDM and a u*q cross term.
    return [0.55, 0.0, 0.0, 0.11, 0.0, 0.0]
# EVOLVE-BLOCK-END
