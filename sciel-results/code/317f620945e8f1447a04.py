"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate a spiral on the triangular lattice.
    # This configuration is known to attain the Harborth bound for unit distances.
    points = [(0, 0)]
    if len(points) >= 32:
        return [[p[0], p[1]] for p in points[:32]]
        
    # Directions: (da, db)
    # 0: (1, 0)
    # 1: (0, 1)
    # 2: (-1, 1)
    # 3: (-1, 0)
    # 4: (0, -1)
    # 5: (1, -1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    a, b = 0, 0
    step_count = 1
    dir_idx = 0
    
    while len(points) < 32:
        # Move step_count times in current direction
        for _ in range(step_count):
            a += dirs[dir_idx][0]
            b += dirs[dir_idx][1]
            points.append((a, b))
            if len(points) >= 32:
                break
        if len(points) >= 32:
            break
            
        dir_idx = (dir_idx + 1) % 6
        if dir_idx % 2 == 0:
            step_count += 1

    return [[p[0], p[1]] for p in points[:32]]
# EVOLVE-BLOCK-END
