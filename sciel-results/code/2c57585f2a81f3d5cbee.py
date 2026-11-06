"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import copy

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for max TM band gap (bands 4-5)."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    # Deadline: 120s CPU budget, leave 10s margin
    start_time = time.time()
    deadline = start_time + 110
    
    def is_valid(cell):
        """Check if cell is valid: square symmetry (row-reverse, col-reverse, transpose)."""
        if len(cell) != size:
            return False
        for i in range(size):
            if len(cell[i]) != size:
                return False
            # Row symmetry: cell[i] == cell[size-1-i]
            for j in range(size):
                if cell[i][j] != cell[size - 1 - i][j]:
                    return False
                # Column symmetry: cell[i][j] == cell[j][i] (transpose)
                if cell[i][j] != cell[j][i]:
                    return False
        return True
    
    def make_symmetric(cell):
        """Enforce all symmetries: row-reverse, col-reverse, and transpose."""
        cell = [row[:] for row in cell]
        
        # Iterate until stable
        for _ in range(20):
            changed = False
            
            # Enforce vertical flip: cell[i][j] == cell[size-1-i][j]
            for i in range(size // 2):
                for j in range(size):
                    if cell[i][j] != cell[size - 1 - i][j]:
                        # Use majority or just pick top half
                        cell[size - 1 - i][j] = cell[i][j]
                        changed = True
            
            # Enforce horizontal flip: cell[i][j] == cell[i][size-1-j]
            for i in range(size):
                for j in range(size // 2):
                    if cell[i][j] != cell[i][size - 1 - j]:
                        cell[i][size - 1 - j] = cell[i][j]
                        changed = True
            
            # Enforce transpose: cell[i][j] == cell[j][i]
            for i in range(size):
                for j in range(i + 1, size):
                    if cell[i][j] != cell[j][i]:
                        # Need to be careful here - if we set cell[j][i] = cell[i][j],
                        # we might break previous constraints. Let's set both to 0 if conflict.
                        if cell[i][j] != cell[j][i]:
                            # Check the orbit under all symmetries
                            # For simplicity, force to 0
                            cell[i][j] = 0
                            cell[j][i] = 0
                            changed = True
            
            if not changed:
                break
        
        # Final cleanup: force any remaining inconsistencies to 0
        for i in range(size):
            for j in range(size):
                vals = set()
                vals.add(cell[i][j])
                vals.add(cell[size-1-i][j])  # vertical flip
                vals.add(cell[i][size-1-j])  # horizontal flip
                vals.add(cell[j][i])  # transpose
                if len(vals) > 1:
                    cell[i][j] = 0
                    cell[size-1-i][j] = 0
                    cell[i][size-1-j] = 0
                    cell[j][i] = 0
        
        return cell
    
    def count_dielectric(cell):
        return sum(sum(row) for row in cell)
    
    def estimate_band_gap(cell):
        """
        Quick heuristic estimate of the band gap.
        For a photonic crystal, the band gap is related to the periodicity and contrast.
        A simple proxy: measure the spatial frequency content.
        Higher contrast at the right wavelength -> larger gap potential.
        
        For TM polarization in a square lattice, the fundamental gap is around the
        first Brillouin zone boundary. The gap size depends on the Fourier components
        of epsilon at the reciprocal lattice vectors.
        
        Simple heuristic: compute the magnitude of the first-order Fourier coefficient
        of (epsilon - eps_avg). Higher magnitude -> larger gap potential.
        """
        eps_avg = (count_dielectric(cell) * eps_high + (size*size - count_dielectric(cell)) * eps_low) / (size*size)
        
        # Compute Fourier coefficient at G = (2*pi/size, 0) and (0, 2*pi/size)
        # which correspond to the first Brillouin zone boundary
        fx = 0.0
        fy = 0.0
        n = size * size
        for i in range(size):
            for j in range(size):
                eps = eps_high if cell[i][j] == 1 else eps_low
                delta = eps - eps_avg
                fx += delta * math.cos(2 * math.pi * i / size)
                fy += delta * math.cos(2 * math.pi * j / size)
        
        fx /= n
        fy /= n
        fourier_mag = math.sqrt(fx*fx + fy*fy)
        
        # The gap is roughly proportional to the Fourier component magnitude
        # times a factor depending on polarization
        return fourier_mag
    
    def make_cell_from_params(radius_x, radius_y, offset_x, offset_y, shape="circle"):
        """Create a cell with given dielectric shape parameters."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5 + offset_x
                y = (j + 0.5) / size - 0.5 + offset_y
                if shape == "circle":
                    if x*x/(radius_x*radius_x) + y*y/(radius_y*radius_y) < 1:
                        cell[i][j] = 1
                elif shape == "square":
                    if abs(x) < radius_x and abs(y) < radius_y:
                        cell[i][j] = 1
                elif shape == "cross":
                    if (abs(x) < radius_x and abs(y) < 0.3) or (abs(y) < radius_y and abs(x) < 0.3):
                        cell[i][j] = 1
        return make_symmetric(cell)
    
    # Strategy: Try several known good configurations for TM band gaps
    # The seed uses a centered circle with r=0.2
    # For larger gaps, we might need different geometries
    
    best_cell = None
    best_score = -1
    
    # Candidate 1: Original seed - centered circle r=0.2
    cell1 = make_cell_from_params(0.2, 0.2, 0, 0, "circle")
    score1 = estimate_band_gap(cell1)
    if is_valid(cell1):
        best_cell = cell1
        best_score = score1
    
    # Candidate 2: Larger circles
    for r in [0.25, 0.3, 0.35, 0.4]:
        cell = make_cell_from_params(r, r, 0, 0, "circle")
        if is_valid(cell):
            s = estimate_band_gap(cell)
            if s > best_score:
                best_score = s
                best_cell = cell
    
    # Candidate 3: Elliptical shapes
    for rx in [0.15, 0.2, 0.25, 0.3]:
        for ry in [0.15, 0.2, 0.25, 0.3]:
            if abs(rx - ry) < 0.01:
                continue
            cell = make_cell_from_params(rx, ry, 0, 0, "circle")
            if is_valid(cell):
                s = estimate_band_gap(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
    
    # Candidate 4: Square shapes
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        cell = make_cell_from_params(r, r, 0, 0, "square")
        if is_valid(cell):
            s = estimate_band_gap(cell)
            if s > best_score:
                best_score = s
                best_cell = cell
    
    # Candidate 5: Cross shapes
    for r in [0.15, 0.2, 0.25, 0.3]:
        cell = make_cell_from_params(r, r, 0, 0, "cross")
        if is_valid(cell):
            s = estimate_band_gap(cell)
            if s > best_score:
                best_score = s
                best_cell = cell
    
    # Candidate 6: Ring shapes (dielectric ring, air center)
    for outer_r in [0.3, 0.35, 0.4]:
        for inner_r in [0.1, 0.15, 0.2]:
            cell = [[0]*size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r2 = x*x + y*y
                    if inner_r*inner_r < r2 < outer_r*outer_r:
                        cell[i][j] = 1
            cell = make_symmetric(cell)
            if is_valid(cell):
                s = estimate_band_gap(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
    
    # Candidate 7: Four small circles at quadrant centers
    for r in [0.1, 0.12, 0.15, 0.18]:
        for offset in [0.2, 0.25, 0.3]:
            cell = [[0]*size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    # Distance to nearest quadrant center
                    dx = abs(x) - offset
                    dy = abs(y) - offset
                    if dx*dx + dy*dy < r*r and dx < 0 and dy < 0:
                        cell[i][j] = 1
            cell = make_symmetric(cell)
            if is_valid(cell):
                s = estimate_band_gap(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
    
    # Ensure we always return a valid cell
    if best_cell is None:
        # Fallback to seed
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            best_cell.append([1 if x*x + ((j+0.5)/size - 0.5)**2 < 0.2**2 else 0 for j in range(size)])
    
    # Final validation
    if not is_valid(best_cell):
        # Force validity
        best_cell = make_symmetric(best_cell)
        if not is_valid(best_cell):
            # Last resort: all zeros
            best_cell = [[0]*size for _ in range(size)]
    
    return best_cell
# EVOLVE-BLOCK-END
