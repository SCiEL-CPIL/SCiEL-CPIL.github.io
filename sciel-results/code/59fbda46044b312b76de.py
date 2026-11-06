"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for max TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 230s kill
    
    size = problem["pixels"]
    
    # Try several symmetric candidate patterns and pick the best one
    # All patterns must have 4-fold rotational symmetry (transpose + 180° rotation)
    
    candidates = []
    
    # Candidate 1: Dielectric cross/plus shape with arms
    # Dielectric forms a plus sign with thickness ~4 pixels
    cell1 = [[0]*size for _ in range(size)]
    center = size // 2
    arm_thickness = 4
    for i in range(size):
        for j in range(size):
            # Horizontal arm
            if abs(j - center + 0.5) <= arm_thickness/2:
                cell1[i][j] = 1
            # Vertical arm
            if abs(i - center + 0.5) <= arm_thickness/2:
                cell1[i][j] = 1
    # Add small squares at corners for extra structure
    corner_size = 3
    for i in range(size):
        for j in range(size):
            in_corner = False
            for ci, cj in [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]:
                if abs(i - ci) < corner_size and abs(j - cj) < corner_size:
                    in_corner = True
                    break
            if in_corner:
                cell1[i][j] = 1
    candidates.append(cell1)
    
    # Candidate 2: Air holes in dielectric background (inverted approach)
    # Dielectric background with 4 circular air holes
    cell2 = [[1]*size for _ in range(size)]
    hole_radius = 0.12
    hole_positions = [
        (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)
    ]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            for hx, hy in hole_positions:
                if (x - hx)**2 + (y - hy)**2 < hole_radius**2:
                    cell2[i][j] = 0
                    break
    candidates.append(cell2)
    
    # Candidate 3: Dielectric ring pattern
    cell3 = [[0]*size for _ in range(size)]
    inner_r = 0.15
    outer_r = 0.35
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = (x*x + y*y)**0.5
            if inner_r < r < outer_r:
                cell3[i][j] = 1
    # Add 4-fold symmetric bumps
    bump_r = 0.08
    bump_positions = [(0.45, 0), (0, 0.45), (-0.45, 0), (0, -0.45)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for bx, by in bump_positions:
                if (x - bx)**2 + (y - by)**2 < bump_r**2:
                    cell3[i][j] = 1
    candidates.append(cell3)
    
    # Candidate 4: Checkerboard-like pattern with 4-fold symmetry
    cell4 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Create a pattern based on distance from center and angle
            r = (x*x + y*y)**0.5
            # Dielectric in regions where r is in certain bands
            if 0.1 < r < 0.25 or 0.35 < r < 0.45:
                cell4[i][j] = 1
    # Ensure 4-fold symmetry by averaging with rotations
    cell4_rot90 = [[cell4[size-1-j][i] for j in range(size)] for i in range(size)]
    cell4_rot180 = [[cell4[size-1-i][size-1-j] for i in range(size) for j in range(size)]]
    cell4_rot270 = [[cell4[j][size-1-i] for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            avg = (cell4[i][j] + cell4_rot90[i][j] + cell4_rot180[i][j] + cell4_rot270[i][j]) / 4.0
            cell4[i][j] = 1 if avg >= 0.5 else 0
    candidates.append(cell4)
    
    # Candidate 5: Dielectric squares at specific positions
    cell5 = [[0]*size for _ in range(size)]
    sq_size = 5
    positions = [
        (6, 6), (6, size-6-sq_size), (size-6-sq_size, 6), (size-6-sq_size, size-6-sq_size),
        (center-sq_size//2, center-sq_size//2)  # center square
    ]
    for pi, pj in positions:
        for i in range(pi, min(pi+sq_size, size)):
            for j in range(pj, min(pj+sq_size, size)):
                cell5[i][j] = 1
    # Ensure symmetry
    for i in range(size):
        for j in range(size):
            if cell5[i][j] == 1:
                cell5[size-1-i][j] = 1
                cell5[i][size-1-j] = 1
                cell5[size-1-i][size-1-j] = 1
                cell5[j][i] = 1
                cell5[j][size-1-i] = 1
                cell5[size-1-i][j] = 1
                cell5[size-1-j][i] = 1
    candidates.append(cell5)
    
    # Verify symmetry for all candidates and pick a good one
    def check_symmetry(cell):
        n = len(cell)
        # Check transpose symmetry
        for i in range(n):
            for j in range(n):
                if cell[i][j] != cell[j][i]:
                    return False
        # Check 180 rotation
        for i in range(n):
            for j in range(n):
                if cell[i][j] != cell[n-1-i][n-1-j]:
                    return False
        return True
    
    valid_candidates = []
    for c in candidates:
        if check_symmetry(c):
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Fallback: simple centered rod
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x*x + ((j+0.5)/size - 0.5)**2 < 0.2**2 else 0 for j in range(size)])
        return cell
    
    # Pick the candidate with moderate dielectric fraction (neither too low nor too high)
    def dielectric_fraction(cell):
        n = len(cell)
        count = sum(sum(row) for row in cell)
        return count / (n*n)
    
    # Prefer candidates with fraction between 0.3 and 0.6
    scored = []
    for c in valid_candidates:
        frac = dielectric_fraction(c)
        # Score based on how close to 0.4 (good mid-range)
        score = -abs(frac - 0.4)
        scored.append((score, c))
    
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    
    return best
# EVOLVE-BLOCK-END
