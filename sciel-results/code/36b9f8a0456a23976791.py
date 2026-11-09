import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    # Seed roots: equally spaced starting at 4
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Try to optimize the roots using a simple approach
    # The objective likely involves some norm of the polynomial with double roots
    # For now, try a few candidate configurations
    
    candidates = []
    scores = []
    
    # Candidate 1: seed roots
    candidates.append(seed_roots.copy())
    
    # Candidate 2: roots of L_12^(0) scaled
    try:
        from numpy.polynomial.laguerre import laggauss
        roots_l12, _ = laggauss(12)
        # Scale to have similar range as seed
        roots_l12_scaled = roots_l12 * (48.0 / roots_l12.max())
        candidates.append(roots_l12_scaled)
    except:
        pass
    
    # Candidate 3: roots of L_12^(1) scaled  
    try:
        from scipy.special import roots_genlaguerre
        roots_l12_1, _ = roots_genlaguerre(12, 1)
        roots_l12_1_scaled = roots_l12_1 * (48.0 / roots_l12_1.max())
        candidates.append(roots_l12_1_scaled)
    except:
        pass
    
    # Candidate 4: geometric progression
    geo_roots = 4.0 * np.power(2.0, np.arange(12) / 11.0 * 3.0)  # from 4 to 48
    candidates.append(geo_roots)
    
    # Candidate 5: slightly perturbed seed
    np.random.seed(42)
    perturbed = seed_roots + np.random.normal(0, 0.5, 12)
    perturbed = np.sort(perturbed)
    candidates.append(perturbed)
    
    # Simple heuristic: pick the candidate with the "best" property
    # Without knowing the exact objective, use the seed as default
    # But try to return a valid positive root list
    
    best_roots = seed_roots.copy()
    
    # Check time
    if time.time() > deadline - 5:
        return best_roots.tolist()
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
