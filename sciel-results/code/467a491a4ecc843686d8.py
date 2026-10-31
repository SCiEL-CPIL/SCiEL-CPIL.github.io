"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Computes a pseudo-random priority for the given vector in F_3^n.
    This randomizes the greedy selection order, which helps avoid the
    local clustering and early saturation observed with lexicographic order.
    """
    # Pack the vector into a single integer (base-3 representation)
    val = 0
    for x in vector:
        val = val * 3 + x
        
    # Apply SplitMix64 finalizer to ensure high-quality bit mixing
    # This makes the priority effectively random but deterministic
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb
    val = val ^ (val >> 31)
    
    # Return as a positive float
    return float(val & 0xFFFFFFFFFFFFFFFF)
# EVOLVE-BLOCK-END
