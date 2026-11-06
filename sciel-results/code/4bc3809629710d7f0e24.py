"""Product-formula design for Heisenberg-chain time evolution using static 2nd-order symmetric scheme."""

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula using a static 2nd-order symmetric (Strang) scheme.
    This avoids SciPy dependencies and ensures fast, stable execution.
    The scheme repeats the Strang splitting (A-half, B-full, A-half) 4 times.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12 (though contract says 12)
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # Strang splitting: A(1/2) B(1) A(1/2)
    # We repeat this 4 times to fill 12 stages.
    # Each repetition contributes 1/4 to the total time step for B, and 1/8+1/8=1/4 to A.
    # Total A = 4 * 1/4 = 1. Total B = 4 * 1/4 = 1.
    
    # Structure per Strang step (mapped to B, A pairs):
    # 1. A(1/8)  -> b=0, a=1/8
    # 2. B(1/4)  -> b=1/4, a=0
    # 3. A(1/8)  -> b=0, a=1/8
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    # Step 1
    a[0] = 1.0 / 8.0
    b[0] = 0.0
    a[1] = 0.0
    b[1] = 1.0 / 4.0
    a[2] = 1.0 / 8.0
    b[2] = 0.0
    
    # Step 2
    a[3] = 1.0 / 8.0
    b[3] = 0.0
    a[4] = 0.0
    b[4] = 1.0 / 4.0
    a[5] = 1.0 / 8.0
    b[5] = 0.0
    
    # Step 3
    a[6] = 1.0 / 8.0
    b[6] = 0.0
    a[7] = 0.0
    b[7] = 1.0 / 4.0
    a[8] = 1.0 / 8.0
    b[8] = 0.0
    
    # Step 4
    a[9] = 1.0 / 8.0
    b[9] = 0.0
    a[10] = 0.0
    b[10] = 1.0 / 4.0
    a[11] = 1.0 / 8.0
    b[11] = 0.0
    
    # Verify sums
    # sum(a) = 8 * (1/8) = 1
    # sum(b) = 4 * (1/4) = 1
    
    return a, b
# EVOLVE-BLOCK-END
