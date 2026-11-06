"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def design(problem):
    """
    Square-lattice photonic crystal with dielectric rods in air background.
    This is a structural change from the previous hexagonal-hole-in-dielectric design.
    We place dielectric rods (1) at the center and corners of the unit cell.
    This configuration is known to open significant TM band gaps.
    """
    size = problem.get("pixels", 32)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 230s kill
    
    # We will test a few radii for the rods and return the one with the most "standard" good geometry.
    # Since we can't score, we pick a radius known to be effective: ~0.25 of lattice constant.
    # We also try a slightly smaller/larger one to see if we can pick a better one based on fill factor heuristics?
    # No, let's just generate the best candidate: Rods at center and corners.
    
    best_cell = None
    best_fill = -1
    
    # Candidate radii (fraction of size)
    radii = [0.22, 0.24, 0.26]
    
    for r_frac in radii:
        r_px = r_frac * size
        cell = [[0] * size for _ in range(size)]  # Air background
        
        # Rod centers: Center (0.5, 0.5) and Corners (0,0), (1,0), (0,1), (1,1)
        # Note: Corner rods are centered at the corners of the pixel grid.
        # To handle periodicity and symmetry correctly within the 32x32 array:
        # We define the rod centers in pixel coordinates (center of pixel).
        # Center rod: (15.5, 15.5) if 0-indexed 0..31? No, pixel centers are 0.5, 1.5...
        # Let's use fractional coordinates 0..1 and map to pixels.
        
        centers = [
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]
        
        for cx_frac, cy_frac in centers:
            cx_px = cx_frac * size
            cy_px = cy_frac * size
            
            i_min = max(0, int(math.floor(cy_px - r_px)))
            i_max = min(size - 1, int(math.ceil(cy_px + r_px)))
            j_min = max(0, int(math.floor(cx_px - r_px)))
            j_max = min(size - 1, int(math.ceil(cx_px + r_px)))
            
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    # Pixel center
                    px = (j + 0.5) - cx_px
                    py = (i + 0.5) - cy_px
                    if px * px + py * py <= r_px * r_px:
                        cell[i][j] = 1
        
        # Enforce D4 symmetry strictly
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][size-1-i],
                    cell[j][size-1-i],
                    cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                cell[i][j] = 1 if avg >= 0.5 else 0
        
        # Calculate fill fraction
        ones = sum(sum(row) for row in cell)
        frac = ones / (size * size)
        
        # Heuristic: For TM rods in air, a fill fraction around 0.3-0.5 is often good.
        # We prefer the one closest to 0.4.
        if abs(frac - 0.40) < abs(best_fill - 0.40):
            best_fill = frac
            best_cell = cell
            
        if time.time() - start_time > deadline:
            break
            
    if best_cell is None:
        # Fallback: simple checkerboard or just center rod
        best_cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if (i+j) % 2 == 0:
                    best_cell[i][j] = 1
        # Enforce symmetry
        for i in range(size):
            for j in range(size):
                vals = [
                    best_cell[i][j],
                    best_cell[size-1-i][j],
                    best_cell[i][size-1-j],
                    best_cell[size-1-i][size-1-j],
                    best_cell[j][i],
                    best_cell[size-1-j][size-1-i],
                    best_cell[j][size-1-i],
                    best_cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                best_cell[i][j] = 1 if avg >= 0.5 else 0

    return best_cell
# EVOLVE-BLOCK-END
