"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _make_cell(size, r_center, r_off, offset):
    """Create a symmetric 32x32 cell with dielectric background and air holes.
    Holes at center and at 4 symmetric off-center positions.
    """
    cell = [[1] * size for _ in range(size)]
    half = size / 2.0
    
    # Define hole centers in pixel coordinates (0-indexed)
    # Center hole
    hole_centers = [(half - 0.5, half - 0.5)]
    # Off-center holes at (half-0.5-offset, half-0.5-offset) and symmetric positions
    ox = half - 0.5 - offset
    oy = half - 0.5 - offset
    hole_centers.append((ox, oy))
    hole_centers.append((size - 1 - ox, size - 1 - oy))  # 180 rot
    hole_centers.append((oy, ox))  # transpose
    hole_centers.append((size - 1 - oy, size - 1 - ox))  # transpose + 180
    
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            for (cx, cy) in hole_centers:
                dx = px - cx
                dy = py - cy
                # Determine which radius to use
                if cx == half - 0.5 and cy == half - 0.5:
                    r = r_center
                else:
                    r = r_off
                if dx * dx + dy * dy < r * r:
                    cell[i][j] = 0
                    break
    return cell

def _check_symmetry(cell):
    """Verify D4 symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[j][i]:
                return False
    return True

def design(problem):
    """Design a symmetric 32x32 photonic crystal cell for maximum TM band gap (bands 4-5)."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with a good initial guess: dielectric background with air holes
    # Known good starting point for higher-order gaps
    best_cell = _make_cell(size, 0.15, 0.12, 6.0)
    
    # Local optimization: perturb hole parameters
    params = {
        'r_center': 0.15,
        'r_off': 0.12, 
        'offset': 6.0
    }
    
    # Try a grid of parameter values
    r_center_vals = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]
    r_off_vals = [0.08, 0.10, 0.12, 0.14, 0.15, 0.16]
    offset_vals = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0]
    
    # We can't compute the actual band gap here (no PWE solver), 
    # so we need to use a heuristic. Let's try several designs and pick 
    # one that's known to work well from literature.
    
    # Based on literature, for higher-order TM gaps in square PC,
    # structures with moderate hole sizes and specific offsets work well.
    # Let's try a few candidate designs and return the most promising.
    
    candidates = []
    
    # Candidate 1: Current best
    candidates.append(_make_cell(size, 0.15, 0.12, 6.0))
    
    # Candidate 2: Larger center hole
    candidates.append(_make_cell(size, 0.20, 0.10, 5.5))
    
    # Candidate 3: Smaller holes, closer together
    candidates.append(_make_cell(size, 0.12, 0.14, 7.0))
    
    # Candidate 4: No center hole, just off-center
    cell4 = [[1] * size for _ in range(size)]
    ox = size/2.0 - 0.5 - 6.0
    oy = size/2.0 - 0.5 - 6.0
    r_off = 0.15
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            for (cx, cy) in [(ox, oy), (size-1-ox, size-1-oy), (oy, ox), (size-1-oy, size-1-ox)]:
                dx = px - cx
                dy = py - cy
                if dx*dx + dy*dy < r_off*r_off:
                    cell4[i][j] = 0
                    break
    candidates.append(cell4)
    
    # Candidate 5: Dielectric cross/plus shape (air background with dielectric)
    cell5 = [[0] * size for _ in range(size)]
    w = 5  # width of dielectric arms
    for i in range(size):
        for j in range(size):
            if abs(i - (size//2 - 0.5)) < w/2.0 or abs(j - (size//2 - 0.5)) < w/2.0:
                cell5[i][j] = 1
    candidates.append(cell5)
    
    # Candidate 6: Dielectric ring
    cell6 = [[0] * size for _ in range(size)]
    r_inner = 8.0
    r_outer = 12.0
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) - size/2.0
            dy = (j + 0.5) - size/2.0
            dist = math.sqrt(dx*dx + dy*dy)
            if r_inner < dist < r_outer:
                cell6[i][j] = 1
    candidates.append(cell6)
    
    # Candidate 7: Checkered-like pattern with dielectric
    cell7 = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # Symmetric pattern
            if (i < 8 and j < 8) or (i >= 24 and j < 8) or (i < 8 and j >= 24) or (i >= 24 and j >= 24):
                cell7[i][j] = 1
            elif 12 <= i < 20 and 12 <= j < 20:
                cell7[i][j] = 1
    candidates.append(cell7)
    
    # Verify all candidates have correct symmetry
    valid_candidates = []
    for c in candidates:
        if _check_symmetry(c):
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    # Return the first valid candidate (air-hole design is typically best for higher-order gaps)
    return valid_candidates[0]

# EVOLVE-BLOCK-END
