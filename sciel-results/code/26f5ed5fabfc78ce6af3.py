"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Physically-motivated initial condition:
    # c0 = 0.55 (standard growth index)
    # c3 = -0.5 to cancel leading-order w0 term in gamma expansion
    # This forces remaining coefficients to model subdominant corrections
    return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]
# EVOLVE-BLOCK-END
