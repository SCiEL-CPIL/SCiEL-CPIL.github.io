"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 600  # seconds; leave 30s safety margin
    
    # The seed program returned [4.0 * (i + 1) for i in range(12)] with score 1.0.
    # The Laguerre roots (score 0.604) had smaller min and larger max, suggesting
    # the bound is sensitive to both. Scaling the seed down reduces max root
    # significantly, which may improve the score if the bound is dominated by
    # the largest root or a combination where max is heavy.
    
    # Try a scaled version of the seed roots.
    # Hypothesis: B ~ r_max + 1/r_min. Optimal scale for seed (4..48) is ~0.07.
    # Let's try k=0.1.
    k = 0.1
    roots = [4.0 * (i + 1) * k for i in range(12)]
    
    # Verify constraints
    if all(r > 0 for r in roots) and len(roots) == 12:
        return roots
    else:
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
