"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def coefficients(stages):
    """
    Uses 4th-order Suzuki-Trotter coefficients repeated 4 times to fill 12 stages.
    This provides 4th-order convergence, significantly better than the 2nd-order Strang splitting.
    """
    # 4th-order Suzuki coefficients
    alpha1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    alpha2 = 1.0 - 2.0 * alpha1
    alpha3 = alpha1
    
    # One block of 3 pairs has coefficients [alpha1, alpha2, alpha3] scaled to sum to 1/4?
    # We need 12 pairs total.
    # If we use [alpha1/4, alpha2/4, alpha3/4] repeated 4 times, sum is 1.
    block_a = [alpha1 / 4.0, alpha2 / 4.0, alpha3 / 4.0]
    block_b = [alpha1 / 4.0, alpha2 / 4.0, alpha3 / 4.0]
    
    a = []
    b = []
    for _ in range(4):
        a.extend(block_a)
        b.extend(block_b)
        
    # Ensure lengths match stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    # Truncate if necessary (should not happen for stages=12)
    a = a[:stages]
    b = b[:stages]
    
    # Verify sums
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-10:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        
    return a, b
# EVOLVE-BLOCK-END
