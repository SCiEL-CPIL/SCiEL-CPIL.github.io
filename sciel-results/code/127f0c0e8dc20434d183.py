"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit
    
    # Try a few candidate geometries that are known to produce photonic band gaps
    # For TM polarization in square lattice, circular holes in dielectric background
    # or dielectric rods in air can produce gaps
    
    # Strategy: Use dielectric rods in air (1=dielectric, 0=air)
    # Try different rod sizes and positions
    
    best_cell = None
    best_score = -1.0
    
    # Generate candidate: central dielectric circle of varying radius
    # For square symmetry, we need the pattern to be symmetric under 90-degree rotation
    # and reflection
    
    # Try a single central circle (has full square symmetry)
    for r in [0.2, 0.25, 0.3, 0.35, 0.4]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < r * r:
                    cell[i][j] = 1
        
        # Verify symmetry
        if _check_symmetry(cell, size):
            if best_cell is None:
                best_cell = cell
                best_score = 0.5  # heuristic: larger gap expected for moderate sizes
    
    # Try four circles at corners (also has square symmetry)
    for r in [0.15, 0.2, 0.25]:
        offset = 0.25
        cell = [[0] * size for _ in range(size)]
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                        cell[i][j] = 1
                        break
        
        if _check_symmetry(cell, size):
            # Heuristic score
            score = 0.6 if 0.18 < r < 0.22 else 0.5
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Try a square hole in dielectric background (1=dielectric, 0=air)
    for hole_size in [0.2, 0.25, 0.3]:
        cell = [[1] * size for _ in range(size)]
        half_hole = hole_size / 2
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_hole and abs(y) < half_hole:
                    cell[i][j] = 0
        
        if _check_symmetry(cell, size):
            score = 0.55
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # If no valid cell found, create a simple symmetric pattern
    if best_cell is None:
        best_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.3 * 0.3:
                    best_cell[i][j] = 1
    
    return best_cell

def _check_symmetry(cell, size):
    """Check if cell has square symmetry: 180-degree rotation, transpose, and reflections."""
    # Check 180-degree rotation symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                return False
    
    # Check transpose symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    
    # Check horizontal reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                return False
    
    # Check vertical reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                return False
    
    return True
# EVOLVE-BLOCK-END
