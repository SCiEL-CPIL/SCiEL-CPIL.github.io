"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _make_symmetric_cell(size, pattern_func):
    """Create a symmetric cell by applying pattern to the fundamental domain."""
    cell = [[0] * size for _ in range(size)]
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            # Use the minimum distance to any symmetric equivalent to ensure D4 symmetry
            # Actually, we just need to ensure the pattern itself is D4 symmetric
            cell[i][j] = pattern_func(i, j, size)
    # Enforce symmetry explicitly
    for i in range(size):
        for j in range(size):
            # Mirror about horizontal center
            cell[i][j] = max(cell[i][j], cell[size-1-i][j])
            cell[i][j] = max(cell[i][j], cell[i][size-1-j])
            cell[i][j] = max(cell[i][j], cell[size-1-i][size-1-j])
            # 90 degree rotation: (i,j) -> (j, size-1-i)
            cell[i][j] = max(cell[i][j], cell[j][size-1-i])
            cell[i][j] = max(cell[i][j], cell[size-1-j][i])
            cell[i][j] = max(cell[i][j], cell[size-1-j][size-1-i])
            cell[i][j] = max(cell[i][j], cell[j][i])
    # Convert to 0/1
    for i in range(size):
        for j in range(size):
            cell[i][j] = 1 if cell[i][j] > 0.5 else 0
    return cell

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for TM band gap."""
    size = problem["pixels"]
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin
    
    def rod_cell(radius):
        """Dielectric rod in air."""
        def pattern(i, j, sz):
            cx = (i + 0.5) / sz - 0.5
            cy = (j + 0.5) / sz - 0.5
            return 1 if (cx * cx + cy * cy) < radius * radius else 0
        return _make_symmetric_cell(size, pattern)
    
    def hole_cell(radius):
        """Air hole in dielectric background."""
        def pattern(i, j, sz):
            cx = (i + 0.5) / sz - 0.5
            cy = (j + 0.5) / sz - 0.5
            return 1 if (cx * cx + cy * cy) >= radius * radius else 0
        return _make_symmetric_cell(size, pattern)
    
    def ring_cell(r_inner, r_outer):
        """Dielectric ring in air."""
        def pattern(i, j, sz):
            cx = (i + 0.5) / sz - 0.5
            cy = (j + 0.5) / sz - 0.5
            r2 = cx * cx + cy * cy
            return 1 if (r_inner * r_inner <= r2 <= r_outer * r_outer) else 0
        return _make_symmetric_cell(size, pattern)
    
    def square_rod_cell(half_width):
        """Square dielectric rod in air."""
        def pattern(i, j, sz):
            cx = (i + 0.5) / sz - 0.5
            cy = (j + 0.5) / sz - 0.5
            return 1 if (abs(cx) < half_width and abs(cy) < half_width) else 0
        return _make_symmetric_cell(size, pattern)
    
    def cross_cell(arm_width, arm_length):
        """Cross-shaped dielectric structure."""
        def pattern(i, j, sz):
            cx = (i + 0.5) / sz - 0.5
            cy = (j + 0.5) / sz - 0.5
            in_h = abs(cy) < arm_width / 2.0 and abs(cx) < arm_length / 2.0
            in_v = abs(cx) < arm_width / 2.0 and abs(cy) < arm_length / 2.0
            return 1 if (in_h or in_v) else 0
        return _make_symmetric_cell(size, pattern)
    
    candidates = []
    
    # Dielectric rods in air with various radii
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        if time.time() > deadline:
            break
        candidates.append(rod_cell(r))
    
    # Air holes in dielectric
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        if time.time() > deadline:
            break
        candidates.append(hole_cell(r))
    
    # Ring structures
    for ri, ro in [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.1, 0.4), (0.2, 0.5)]:
        if time.time() > deadline:
            break
        candidates.append(ring_cell(ri, ro))
    
    # Square rods
    for hw in [0.15, 0.2, 0.25, 0.3, 0.35]:
        if time.time() > deadline:
            break
        candidates.append(square_rod_cell(hw))
    
    # Cross structures
    for aw, al in [(0.1, 0.4), (0.15, 0.5), (0.2, 0.6), (0.1, 0.6)]:
        if time.time() > deadline:
            break
        candidates.append(cross_cell(aw, al))
    
    # If we have candidates, return the first one (we can't easily score them without the evaluator)
    # Return a reasonable default: a medium-sized dielectric rod
    if candidates:
        return candidates[0]
    
    # Fallback: original rod
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    return cell
# EVOLVE-BLOCK-END
