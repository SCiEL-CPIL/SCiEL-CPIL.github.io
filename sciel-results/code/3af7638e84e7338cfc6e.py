"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import science_chemistry


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for overhead/return
    
    # 1. Construct initial guess: FCC truncated octahedron
    # The global minimum for LJ38 is a truncated octahedron derived from FCC.
    # LJ minimum distance r_min = 2^(1/6)
    r_min = 2 ** (1.0 / 6.0)
    # In FCC, nearest neighbor distance is a / sqrt(2)
    a = r_min * np.sqrt(2)
    
    # Generate FCC points
    # We need enough points to cover the 38th nearest neighbor.
    # 1 (center) + 12 (1st shell) + 6 (2nd shell) + 24 (3rd shell) = 43 points.
    # So we need to generate at least 43 points.
    # A box of size 5x5x5 in lattice units should be plenty.
    max_coord = 5
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    fcc_points.append((i, j, k))
    
    fcc_points = np.array(fcc_points, dtype=float) * a
    
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_points, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the 38 closest
    initial_pos = fcc_points[sorted_indices[:N]]
    
    best_pos = initial_pos
    
    # 2. Relax
    try:
        # Use relax if time permits
        if time.time() < deadline - 5.0:
            relaxed_pos = science_chemistry.relax(initial_pos)
            if relaxed_pos is not None and relaxed_pos.shape == (N, 3):
                # Basic sanity check: are atoms too close?
                # If they are too close, the energy is huge.
                # The relax function should avoid this, but let's check min distance.
                # Calculating all pairs for 38 atoms is O(N^2) = 1444, very fast.
                min_d = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.5:
                    best_pos = relaxed_pos
                else:
                    # If relaxation caused overlap, stick to initial
                    pass
    except Exception:
        pass
        
    # 3. Return best position
    return best_pos
# EVOLVE-BLOCK-END
