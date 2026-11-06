"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 220.0  # seconds, with safety margin before 230s hard kill
    margin = 15.0  # seconds before hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin
    
    def make_cell_from_fourier(amps, phases):
        """Create a cell from Fourier superposition, ensuring symmetry.
        
        amps: dict mapping (m,n) to amplitude
        phases: dict mapping (m,n) to phase (radians)
        
        The pattern is: eps(x,y) = base + sum_{(m,n)} amp_{mn} * cos(m*pi*x + n*pi*y + phase_{mn})
        where x,y are in [-0.5, 0.5]
        
        For square symmetry, we need the pattern to be invariant under:
        - 90° rotation: (x,y) -> (-y,x)
        - Reflection across x=y: (x,y) -> (y,x)
        
        This means we should only use modes that are symmetric under these operations.
        For cos(m*pi*x + n*pi*y + phi):
        - Under 90° rotation: cos(m*pi*(-y) + n*pi*x + phi) = cos(n*pi*x - m*pi*y + phi)
          For this to equal cos(m*pi*x + n*pi*y + phi), we need m = -n or special phases.
        
        Actually, let's think more carefully. The dielectric function must satisfy:
        eps(x,y) = eps(-y,x) = eps(y,x) = eps(-x,-y) etc.
        
        For a Fourier mode cos(m*pi*x + n*pi*y + phi):
        - eps(-y,x) = cos(m*pi*(-y) + n*pi*x + phi) = cos(n*pi*x - m*pi*y + phi)
        - For equality with cos(m*pi*x + n*pi*y + phi), we need the mode to be invariant.
        
        The simplest approach: use only modes where the pattern is inherently symmetric.
        For example:
        - cos(2*pi*x) * cos(2*pi*y) is symmetric under all d4 operations
        - cos(4*pi*x) + cos(4*pi*y) is symmetric
        - cos(2*pi*(x+y)) + cos(2*pi*(x-y)) is symmetric (this is 2*cos(2*pi*x)*cos(2*pi*y))
        
        Let me use a basis of d4-symmetric functions:
        1. cos(2*pi*m*x) * cos(2*pi*n*y) for m,n >= 0 (symmetric under reflections)
        2. cos(2*pi*(m*x + n*y)) + cos(2*pi*(m*x - n*y)) = 2*cos(2*pi*m*x)*cos(2*pi*n*y)
        
        So actually, the product form cos(2*pi*m*x)*cos(2*pi*n*y) is already d4-symmetric
        when we allow m,n to be non-negative integers, because:
        - cos(2*pi*m*(-y))*cos(2*pi*n*x) = cos(2*pi*m*y)*cos(2*pi*n*x)
          This equals cos(2*pi*m*x)*cos(2*pi*n*y) only if m=n or we sum over all.
        
        Hmm, let me just enforce symmetry by averaging over the group after generation.
        """
        cell = [[0] * size for _ in range(size)]
        base = 0.3  # base filling fraction
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.0) / size - 0.5  # Wait, let me be consistent
                y = (j + 0.5) / size - 0.5
                
                val = base
                for (m, n), amp in amps.items():
                    phase = phases.get((m, n), 0.0)
                    val += amp * math.cos(m * math.pi * x + n * math.pi * y + phase)
                
                cell[i][j] = 1 if val > 0.5 else 0
        
        # Enforce symmetry by averaging over the d4 group
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # d4 orbit: (i,j), (j,i), (size-1-i, size-1-j), (size-1-j, size-1-i),
                # and reflections: (i, size-1-j), (size-1-j, i), (j, size-1-i), (size-1-i, j)
                vals = [
                    cell[i][j],
                    cell[j][i],
                    cell[size-1-i][size-1-j],
                    cell[size-1-j][size-1-i],
                    cell[i][size-1-j],
                    cell[size-1-j][i],
                    cell[j][size-1-i],
                    cell[size-1-i][j],
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 4 else 0
        
        return sym_cell
    
    def check_symmetry(cell):
        """Check if cell has full square symmetry."""
        size = len(cell)
        # Check transpose: cell[i][j] == cell[j][i]
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[j][i]:
                    return False
        # Check 180° rotation: cell[i][j] == cell[size-1-i][size-1-j]
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[size-1-i][size-1-j]:
                    return False
        # Check reflection across vertical midline: cell[i][j] == cell[i][size-1-j]
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[i][size-1-j]:
                    return False
        return True
    
    def four_rod_pattern(radius, offset):
        """Four rods at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def make_cell_from_pattern(pattern_func):
        """Create a cell from a pattern function, enforcing symmetry."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        
        # Enforce full d4 symmetry
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[j][i],
                    cell[size-1-i][size-1-j],
                    cell[size-1-j][size-1-i],
                    cell[i][size-1-j],
                    cell[size-1-j][i],
                    cell[j][size-1-i],
                    cell[size-1-i][j],
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 4 else 0
        
        return sym_cell
    
    def ring_pattern(inner_r, outer_r):
        """Dielectric ring."""
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def cross_pattern(arm_width, arm_length):
        """Cross/plus shape."""
        def f(x, y):
            if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                return True
            if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                return True
            return False
        return f
    
    def four_rings_pattern(inner_r, outer_r, offset):
        """Four small rings at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx * dx + dy * dy
                    if inner_r * inner_r < r2 < outer_r * outer_r:
                        return True
            return False
        return f
    
    def rod_pattern(radius):
        """Single centered rod."""
        def f(x, y):
            return x * x + y * y < radius * radius
        return f
    
    def estimate_gap_score(cell):
        """Heuristic score for estimating band gap quality.
        
        Based on physics: for higher-order TM gaps (4-5), we want:
        1. Moderate filling fraction (30-50%)
        2. Strong second-order Fourier components (for coupling bands 4 and 5)
        3. Good symmetry
        
        We compute a proxy: the magnitude of the (2,0), (0,2), and (2,2) Fourier coefficients
        of the dielectric distribution, weighted by filling fraction.
        """
        fill = sum(sum(row) for row in cell) / (size * size)
        
        if fill < 0.15 or fill > 0.65:
            return -1.0  # Bad filling fraction
        
        # Compute Fourier coefficients for key modes
        # The dielectric function is eps(x,y) = 1 + (eps_high-1) * cell(x,y)
        # But for gap opening, what matters is the contrast pattern.
        
        # Compute (2,0) mode: integral of cell(x,y) * cos(2*pi*2*x) * cos(2*pi*0*y) dx dy
        # Approximate by sum over pixels
        sum_20 = 0.0
        sum_02 = 0.0
        sum_22 = 0.0
        sum_40 = 0.0
        sum_04 = 0.0
        sum_44 = 0.0
        
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    sum_20 += math.cos(4 * math.pi * x) * math.cos(0)
                    sum_02 += math.cos(0) * math.cos(4 * math.pi * y)
                    sum_22 += math.cos(4 * math.pi * x) * math.cos(4 * math.pi * y)
                    sum_40 += math.cos(8 * math.pi * x)
                    sum_04 += math.cos(8 * math.pi * y)
                    sum_44 += math.cos(8 * math.pi * x) * math.cos(8 * math.pi * y)
        
        # Normalize by number of pixels
        n_pix = size * size
        c20 = sum_20 / n_pix
        c02 = sum_02 / n_pix
        c22 = sum_22 / n_pix
        c40 = sum_40 / n_pix
        c04 = sum_04 / n_pix
        c44 = sum_44 / n_pix
        
        # For the 4-5 TM gap, second-order modes are most important
        # The gap is proportional to the coupling strength, which scales with |c20|, |c02|, |c22|
        second_order_strength = abs(c20) + abs(c02) + abs(c22)
        fourth_order_strength = abs(c40) + abs(c04) + abs(c44)
        
        # Score: prefer strong second-order components with moderate filling
        # The optimal filling is around 0.3-0.4 for dielectric-in-air
        fill_score = 1.0 - abs(fill - 0.35) / 0.35  # peaks at fill=0.35
        
        # Combine: strong second-order + good filling
        score = second_order_strength * 10.0 * fill_score + fourth_order_strength * 2.0 * fill_score
        
        return score
    
    # Candidate patterns to try
    candidates = []
    
    # Four-rod patterns (known to enhance second-order Fourier components)
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(("4rod_r%.2f_o%.2f" % (r, off), four_rod_pattern(r, off)))
    
    # Single rod with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        candidates.append(("rod_r%.2f" % r, rod_pattern(r)))
    
    # Ring patterns
    for ir, orr in [(0.08, 0.18), (0.10, 0.20), (0.12, 0.22), (0.14, 0.25), (0.16, 0.28)]:
        candidates.append(("ring_i%.2f_o%.2f" % (ir, orr), ring_pattern(ir, orr)))
    
    # Cross patterns
    for aw, al in [(0.10, 0.35), (0.12, 0.40), (0.15, 0.45), (0.18, 0.50), (0.20, 0.55)]:
        candidates.append(("cross_w%.2f_l%.2f" % (aw, al), cross_pattern(aw, al)))
    
    # Four small rings
    for ir, orr in [(0.04, 0.10), (0.05, 0.12), (0.06, 0.14), (0.07, 0.16)]:
        for off in [0.22, 0.25, 0.28]:
            candidates.append(("4ring_i%.2f_o%.2f_o%.2f" % (ir, orr, off), four_rings_pattern(ir, orr, off)))
    
    # Fourier-based patterns: superposition of low-order harmonics
    # Use modes that are d4-symmetric: cos(2*pi*m*x)*cos(2*pi*n*y) form
    # Key modes for 4-5 gap: (1,0), (0,1), (1,1), (2,0), (0,2), (2,2), (3,0), (0,3), (3,3)
    # In terms of pixel coordinates, x,y in [-0.5, 0.5], so frequency is m cycles per unit cell
    
    # Define Fourier basis functions (d4-symmetric)
    def fourier_cell(modes):
        """Create cell from Fourier superposition.
        
        modes: list of (amp, m, n) tuples
        The pattern is: f(x,y) = base + sum amp_k * cos(2*pi*m_k*x) * cos(2*pi*n_k*y)
        """
        base = 0.35  # base filling
        
        def f(x, y):
            val = base
            for amp, m, n in modes:
                val += amp * math.cos(2 * math.pi * m * x) * math.cos(2 * math.pi * n * y)
            return val > 0.5
        
        return f
    
    # Try various Fourier mode combinations
    # Focus on modes that enhance (2,0), (0,2), (2,2) components
    fourier_candidates = [
        # Strong (1,0) and (0,1) - creates a square-ish pattern
        [(0.3, 1, 0), (0.3, 0, 1)],
        [(0.4, 1, 0), (0.4, 0, 1)],
        [(0.5, 1, 0), (0.5, 0, 1)],
        
        # Add (1,1) component
        [(0.3, 1, 0), (0.3, 0, 1), (0.2, 1, 1)],
        [(0.4, 1, 0), (0.4, 0, 1), (0.2, 1, 1)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.3, 1, 1)],
        
        # Strong (2,0) and (0,2) - directly enhances second-order
        [(0.3, 2, 0), (0.3, 0, 2)],
        [(0.4, 2, 0), (0.4, 0, 2)],
        [(0.5, 2, 0), (0.5, 0, 2)],
        
        # Mix of (1,0), (0,1), (2,0), (0,2)
        [(0.2, 1, 0), (0.2, 0, 1), (0.3, 2, 0), (0.3, 0, 2)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.2, 2, 0), (0.2, 0, 2)],
        
        # (2,2) component
        [(0.3, 2, 2)],
        [(0.4, 2, 2)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.3, 2, 2)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.2, 2, 0), (0.2, 0, 2), (0.2, 2, 2)],
        
        # Higher order
        [(0.2, 3, 0), (0.2, 0, 3)],
        [(0.3, 3, 0), (0.3, 0, 3)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.2, 3, 0), (0.2, 0, 3)],
        
        # Combined: try to create a pattern with strong 2nd order
        [(0.15, 1, 0), (0.15, 0, 1), (0.25, 2, 0), (0.25, 0, 2), (0.15, 2, 2)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.3, 2, 0), (0.3, 0, 2), (0.2, 2, 2)],
        [(0.1, 1, 0), (0.1, 0, 1), (0.35, 2, 0), (0.35, 0, 2), (0.15, 2, 2)],
    ]
    
    for idx, modes in enumerate(fourier_candidates):
        candidates.append(("fourier_%d" % idx, fourier_cell(modes)))
    
    # Evaluate candidates
    best_cell = None
    best_score = -1
    
    for name, pattern_func in candidates:
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        
        # Basic validation: check symmetry
        if not check_symmetry(cell):
            continue
        
        # Check filling fraction
        fill = sum(sum(row) for row in cell) / (size * size)
        if fill < 0.1 or fill > 0.7:
            continue
        
        # Score using heuristic
        score = estimate_gap_score(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If no good candidate, fall back to a known good design
    if best_cell is None:
        # Four-rod pattern with radius 0.10 at offset 0.25
        cell = make_cell_from_pattern(four_rod_pattern(0.10, 0.25))
        if check_symmetry(cell):
            best_cell = cell
        else:
            # Fallback to single rod
            best_cell = make_cell_from_pattern(rod_pattern(0.20))
    
    return best_cell
# EVOLVE-BLOCK-END
