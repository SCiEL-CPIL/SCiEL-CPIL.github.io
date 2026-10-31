"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate FCC lattice points
    # FCC condition: i + j + k is even
    # We need 38 atoms. The 38th atom is in the shell with r^2 = 8.
    # Max coordinate magnitude needed is likely small, e.g., up to 3 or 4.
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    points.append((r2, i, j, k))
    
    # Sort by distance squared
    points.sort(key=lambda x: x[0])
    
    # Take the first 38
    selected = points[:N]
    
    # Extract coordinates
    coords = np.array([(p[1], p[2], p[3]) for p in selected], dtype=float)
    
    # Scale to optimal LJ distance
    # Nearest neighbor in integer FCC is sqrt(2).
    # Target nearest neighbor is 2^(1/6).
    # Scale factor = 2^(1/6) / sqrt(2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3)
    scale = 2 ** (-1/3.0)
    coords = coords * scale
    
    # Relax the configuration
    # The evaluator provides science_chemistry.relax, but we must assume it's available 
    # or implement a simple gradient descent if not? 
    # The prompt says: "science_chemistry.relax(positions) ... is available to candidates."
    # So we can call it.
    try:
        import science_chemistry
        # Relax with a time limit or just let it run. 
        # L-BFGS is usually fast.
        relaxed_coords = science_chemistry.relax(coords)
        
        # Check if relaxation finished within time
        if time.time() > deadline:
             # If we are running out of time, return the best we have (pre-relax or partial?)
             # Since relax is a single call, we either get it or not.
             # If it times out, it might raise an exception or return partial.
             # To be safe, if we are close to deadline, we might return the scaled seed.
             # But usually relax is fast for 38 atoms.
             pass
             
        return relaxed_coords
    except Exception:
        # If relax fails or is not available, return the scaled geometric guess
        return coords

# EVOLVE-BLOCK-END
