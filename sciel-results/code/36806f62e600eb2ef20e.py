"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 580  # safety margin before 630s kill
    
    # Strategy: cluster points very tightly near origin with geometric spacing
    # This creates a dense barrier that limits the extent of q(x) roots
    # Start with a tight geometric cluster
    best_score = float('inf')
    best_z = None
    
    # Try several cluster configurations with different base scales and ratios
    configs = []
    # Very tight clusters near origin
    for base in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            z = [base * (ratio ** i) for i in range(12)]
            # Ensure all positive and distinct
            if all(z[i] > 0 and z[i] != z[j] for i in range(12) for j in range(i+1, 12)):
                configs.append(z)
    
    # Also try symmetric-ish arrangements
    for base in [0.01, 0.02, 0.05]:
        z = []
        for i in range(12):
            z.append(base * (1 + 0.1 * i) * (1 + 0.01 * i**2))
        if all(z[i] > 0 and z[i] != z[j] for i in range(12) for j in range(i+1, 12)):
            configs.append(z)
    
    # Sort configs by how tightly clustered they are (smaller spread first)
    configs.sort(key=lambda z: z[-1] - z[0])
    
    # We can only evaluate a few due to time constraints
    # The exact rational evaluation takes seconds to minutes
    # Let's try the most promising (tightest) first
    
    # Since we can't afford many full evaluations, return the tightest cluster
    # The tightest cluster should give the smallest B
    # Use the first config (tightest)
    
    # Actually, let's think more carefully.
    # The tightest cluster (all near 0) means prod(x - z_i)^2 has a very sharp feature near 0
    # This should make q(x) decay quickly, limiting B
    # But if points are too close, the system might be ill-conditioned
    
    # A good compromise: geometric with ratio ~2, base ~0.01
    # z = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48]
    # This spans a wide range but is dense near 0
    
    # Let's go with a very tight geometric cluster
    base = 0.005
    ratio = 1.8
    z = [base * (ratio ** i) for i in range(12)]
    
    # Verify positive and distinct
    assert all(zi > 0 for zi in z)
    assert len(set(z)) == 12
    
    return z
# EVOLVE-BLOCK-END
