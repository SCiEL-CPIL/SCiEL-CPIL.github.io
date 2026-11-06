"""
Design a 32x32 D4-symmetric photonic crystal unit cell to maximize the TM 4-5 band gap.

Strategy: Use a physics-informed surrogate based on Fourier coefficients to guide
a local search over D4-symmetric binary patterns. The key insight is that the
4-5 TM band gap in square lattice PCs is strongly influenced by the (1,1) and
(2,0) Fourier components of the dielectric function, and by maintaining an
optimal dielectric fraction (~0.35-0.45 for TM polarization with eps=12).

We parameterize the cell using a small D4-symmetric template (8x8 or 12x12)
and perform a multi-start local search with simulated annealing to escape
local optima. The surrogate evaluates a composite score based on Fourier
content and dielectric fraction, which correlates with the actual band gap.
"""
import math
import time
import random

def _enforce_d4(cell, size):
    """Enforce full D4 symmetry (rotations by 90,180,270 and reflections).
    
    The cell must satisfy:
    - cell[i][j] == cell[size-1-i][j] (vertical reflection)
    - cell[i][j] == cell[i][size-1-j] (horizontal reflection)  
    - cell[i][j] == cell[j][i] (transpose / diagonal reflection)
    
    This implies 8-fold symmetry about the center.
    """
    result = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # Get the 8 symmetric positions
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i),
            ]
            # Majority vote
            val = sum(cell[pi][pj] for pi, pj in positions)
            result[i][j] = 1 if val >= 4 else 0
    return result

def _dielectric_fraction(cell, size=32):
    """Compute fraction of dielectric pixels."""
    total = sum(sum(row) for row in cell)
    return total / (size * size)

def _fourier_magnitude(cell, size, mx, my):
    """Compute magnitude of Fourier coefficient (mx, my) of the binary mask."""
    fx = 0.0
    fy = 0.0
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            val = cell[i][j]
            phase_x = 2 * math.pi * mx * x
            phase_y = 2 * math.pi * my * y
            fx += val * math.cos(phase_x) * math.cos(phase_y) - val * math.sin(phase_x) * math.sin(phase_y)
            fy += val * math.sin(phase_x) * math.cos(phase_y) + val * math.cos(phase_x) * math.sin(phase_y)
    n = size * size
    return math.sqrt((fx/n)**2 + (fy/n)**2)

def _surrogate_score(cell, size=32):
    """
    Physics-informed surrogate for the TM 4-5 band gap.
    
    Key physics:
    - TM band gaps are enhanced by strong 1/epsilon modulation
    - For eps=12, contrast in 1/eps is from 1/12 to 1, i.e., ~0.92
    - The (1,1) Fourier component drives Bragg scattering at X-M zone boundary
    - The (2,0) and (0,2) components drive scattering along zone edges
    - Optimal dielectric fraction is around 0.35-0.50 for TM gaps
    - Higher frequency content generally helps higher-order gaps
    
    Score is a weighted combination that correlates with actual gap.
    """
    f = _dielectric_fraction(cell, size)
    
    # Penalize extreme fractions
    if f < 0.10 or f > 0.80:
        return -1.0
    
    # Sweet spot penalty: optimal is around 0.38-0.45 for TM with eps=12
    frac_penalty = 1.0 - abs(f - 0.42) * 2.5
    
    # Fourier components (normalized)
    f11 = _fourier_magnitude(cell, size, 1, 1)
    f20 = _fourier_magnitude(cell, size, 2, 0)
    f02 = _fourier_magnitude(cell, size, 0, 2)
    f22 = _fourier_magnitude(cell, size, 2, 2)
    f30 = _fourier_magnitude(cell, size, 3, 0)
    
    # The 4-5 gap involves multiple Fourier channels
    # Weight: (1,1) is most important for the primary Bragg condition
    # (2,0) and (0,2) are secondary
    fourier_score = 1.2 * f11 + 0.6 * f20 + 0.6 * f02 + 0.3 * f22 + 0.2 * f30
    
    # Composite score
    score = frac_penalty * fourier_score
    
    # Additional penalty for too-uniform patterns (all same value)
    unique_vals = set(cell[i][j] for i in range(size) for j in range(size))
    if len(unique_vals) <= 1:
        score *= 0.1
    
    return score

def _flip_pixel_d4(cell, size, i, j):
    """Flip pixel (i,j) and all its D4-symmetric counterparts."""
    positions = [
        (i, j),
        (size-1-i, j),
        (i, size-1-j),
        (size-1-i, size-1-j),
        (j, i),
        (size-1-j, i),
        (j, size-1-i),
        (size-1-j, size-1-i),
    ]
    # Deduplicate
    unique_positions = list(set(positions))
    
    # Get current value
    cur_val = cell[i][j]
    new_val = 1 - cur_val
    
    # Apply flip
    for pi, pj in unique_positions:
        cell[pi][pj] = new_val
    
    return unique_positions

def _local_search(cell, size, max_iter=2000, deadline=None):
    """
    Local search with simulated annealing on D4-symmetric binary mask.
    """
    best_cell = [row[:] for row in cell]
    best_score = _surrogate_score(cell, size)
    
    current_cell = [row[:] for row in cell]
    current_score = best_score
    
    temp = 0.05
    cooling = 0.997
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random pixel to flip
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        
        # Flip and evaluate
        positions = _flip_pixel_d4(current_cell, size, i, j)
        new_score = _surrogate_score(current_cell, size)
        
        delta = new_score - current_score
        
        # Accept or reject
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in current_cell]
        else:
            # Undo the flip
            for pi, pj in positions:
                current_cell[pi][pj] = 1 - current_cell[pi][pj]
        
        temp *= cooling
    
    return best_cell, best_score

def _generate_seed_cell(size, seed=None):
    """Generate a random D4-symmetric seed cell."""
    if seed is not None:
        random.seed(seed)
    
    # Start with random values, then enforce D4 symmetry
    cell = [[random.randint(0, 1) for _ in range(size)] for _ in range(size)]
    return _enforce_d4(cell, size)

def _generate_structural_seed(size, pattern_type, params=None):
    """Generate structured seed patterns that are known to produce good gaps."""
    if params is None:
        params = {}
    
    cell = [[0]*size for _ in range(size)]
    
    if pattern_type == "square_pillar":
        # Central square pillar
        sz = params.get("size", 12)
        offset = (size - sz) // 2
        for i in range(offset, offset + sz):
            for j in range(offset, offset + sz):
                cell[i][j] = 1
    
    elif pattern_type == "ring":
        # Square ring
        outer = params.get("outer", 14)
        inner = params.get("inner", 6)
        o_off = (size - outer) // 2
        i_off = (size - inner) // 2
        for i in range(o_off, o_off + outer):
            for j in range(o_off, o_off + outer):
                cell[i][j] = 1
        for i in range(i_off, i_off + inner):
            for j in range(i_off, i_off + inner):
                cell[i][j] = 0
    
    elif pattern_type == "cross":
        # Cross/plus shape
        arm_w = params.get("width", 4)
        length = params.get("length", 24)
        center = size // 2
        half = length // 2
        for k in range(-half, half):
            if abs(k) < arm_w:
                cell[center + k, center + k] = 1
                cell[center + k, center - k] = 1
                cell[center - k, center + k] = 1
                cell[center - k, center - k] = 1
    
    elif pattern_type == "diagonal_stripe":
        # Diagonal stripes with period
        period = params.get("period", 8)
        thickness = params.get("thickness", 3)
        for i in range(size):
            for j in range(size):
                if ((i + j) % period) < thickness:
                    cell[i][j] = 1
    
    elif pattern_type == "checkerboard_mod":
        # Modified checkerboard with varied block size
        block = params.get("block", 4)
        for i in range(size):
            for j in range(size):
                if ((i // block) + (j // block)) % 2 == 0:
                    cell[i][j] = 1
    
    elif pattern_type == "concentric":
        # Concentric square rings
        n_rings = params.get("n_rings", 4)
        ring_w = params.get("ring_w", 3)
        center = size // 2
        for i in range(size):
            for j in range(size):
                # Distance from center in Chebyshev metric
                dist = max(abs(i - center), abs(j - center))
                ring_idx = dist // ring_w
                if ring_idx % 2 == 0:
                    cell[i][j] = 1
    
    elif pattern_type == "hexagonal_approx":
        # Approximate hexagonal packing on square grid
        spacing = params.get("spacing", 6)
        radius = params.get("radius", 2)
        for i in range(0, size, spacing):
            for j in range(0, size, spacing):
                # Place a small cluster
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        pi, pj = i+di, j+dj
                        if 0 <= pi < size and 0 <= pj < size:
                            if di*di + dj*dj <= radius*radius:
                                cell[pi][pj] = 1
    
    # Enforce D4 symmetry
    return _enforce_d4(cell, size)

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap."""
    start_time = time.time()
    deadline = start_time + 215  # Safety margin before 230s kill
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    best_cell = None
    best_score = -1e9
    
    # Phase 1: Try structured seeds with local search
    structural_configs = [
        ("square_pillar", {"size": 10}),
        ("square_pillar", {"size": 12}),
        ("square_pillar", {"size": 14}),
        ("square_pillar", {"size": 8}),
        ("ring", {"outer": 14, "inner": 6}),
        ("ring", {"outer": 16, "inner": 8}),
        ("ring", {"outer": 12, "inner": 4}),
        ("cross", {"width": 4, "length": 24}),
        ("cross", {"width": 6, "length": 20}),
        ("diagonal_stripe", {"period": 8, "thickness": 3}),
        ("diagonal_stripe", {"period": 6, "thickness": 2}),
        ("checkerboard_mod", {"block": 4}),
        ("checkerboard_mod", {"block": 3}),
        ("concentric", {"n_rings": 4, "ring_w": 3}),
        ("concentric", {"n_rings": 5, "ring_w": 2}),
        ("hexagonal_approx", {"spacing": 6, "radius": 2}),
        ("hexagonal_approx", {"spacing": 8, "radius": 2}),
    ]
    
    for pattern_type, params in structural_configs:
        if time.time() > deadline - 30:
            break
        
        seed_cell = _generate_structural_seed(size, pattern_type, params)
        f = _dielectric_fraction(seed_cell, size)
        
        if f < 0.10 or f > 0.80:
            continue
        
        cell, score = _local_search(seed_cell, size, max_iter=1500, deadline=deadline - 25)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Random seeds with local search
    random.seed(42)
    n_random_starts = 15
    for rs in range(n_random_starts):
        if time.time() > deadline - 20:
            break
        
        seed_cell = _generate_seed_cell(size, seed=42 + rs * 100)
        f = _dielectric_fraction(seed_cell, size)
        
        if f < 0.10 or f > 0.80:
            continue
        
        cell, score = _local_search(seed_cell, size, max_iter=1500, deadline=deadline - 20)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 3: If we have time, do a more intensive search on the best cell
    if best_cell is not None and time.time() < deadline - 10:
        # Refine with longer local search
        cell, score = _local_search(best_cell, size, max_iter=3000, deadline=deadline - 5)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Fallback
    if best_cell is None:
        # Simple square pillar fallback
        best_cell = [[0]*size for _ in range(size)]
        pillar_size = 12
        offset = (size - pillar_size) // 2
        for i in range(offset, offset + pillar_size):
            for j in range(offset, offset + pillar_size):
                best_cell[i][j] = 1
        best_cell = _enforce_d4(best_cell, size)
    
    return best_cell
