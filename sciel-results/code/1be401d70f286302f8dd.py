<<<<<<< SEARCH
def construct():
    t0 = time.time()
    # Build hexagonal lattice
    centers = _hex_lattice_centers()
    inner = [[float(p[0]), float(p[1]), 0.0] for p in centers]
    
    # Center the lattice at origin
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
    
    # Initial outer optimization
    side, ang = _optimize_outer(inner)
    
    # Local refinement
    if time.time() - t0 < DEADLINE - 10:
        budget = min(30.0, DEADLINE - (time.time() - t0) - 5)
        inner, side, ang = _local_refine(inner, budget=budget)
        
    # Final check and safety margin
    # The side calculated by _optimize_outer is the minimal side to enclose the vertices.
    # We should add a tiny epsilon to be safe.
    side *= 1.0001
    
    # Return format: inner, (center_x, center_y), side, angle
    # Center is (0,0) because we centered the lattice.
    return inner, (0.0, 0.0), side, ang
=======
def construct():
    t0 = time.time()
    # Build hexagonal lattice
    centers = _hex_lattice_centers()
    inner = [[float(p[0]), float(p[1]), 0.0] for p in centers]
    
    # Center the lattice at origin
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
    
    # Initial outer optimization
    side, ang = _optimize_outer(inner)
    
    # Local refinement
    if time.time() - t0 < DEADLINE - 10:
        budget = min(60.0, DEADLINE - (time.time() - t0) - 5)
        inner, side, ang = _local_refine(inner, budget=budget)
        
    # Final check and safety margin
    # The side calculated by _optimize_outer is the minimal side to enclose the vertices.
    # We should add a tiny epsilon to be safe.
    side *= 1.0001
    
    # Return format: inner, (center_x, center_y), side, angle
    # Center is (0,0) because we centered the lattice.
    return inner, (0.0, 0.0), side, ang
>>>>>>> REPLACE
