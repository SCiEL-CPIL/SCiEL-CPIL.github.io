"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 580  # safety margin before 630s kill
    
    # Strategy: Use a very tight geometric cluster near the origin.
    # The key insight is that placing all double roots very close together
    # creates a strong "barrier" effect that confines the roots of q(x).
    # 
    # We use a geometric sequence with a small base and moderate ratio.
    # The tightest configurations tend to give the smallest B.
    #
    # From the current best (score ~0.395, B~2.48), we try even tighter clusters.
    
    # Try a very tight geometric cluster
    # base=0.001, ratio=1.5 gives z spanning [0.001, 0.001*1.5^11] ≈ [0.001, 0.0865]
    # This is much tighter than the current base=0.005, ratio=1.8
    
    base = 0.001
    ratio = 1.5
    z = [base * (ratio ** i) for i in range(12)]
    
    # Verify positive and distinct
    assert all(zi > 0 for zi in z)
    assert len(set(z)) == 12
    
    return z
# EVOLVE-BLOCK-END
