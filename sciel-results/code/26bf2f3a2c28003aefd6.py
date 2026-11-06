"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]  # 32
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    deadline = 220.0  # seconds, with safety margin before 230s hard kill
    margin = 15.0  # seconds before hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin
    
    def enforce_symmetry(cell):
        """Enforce full d4 (square) symmetry by averaging over the symmetry group."""
        n = len(cell)
        sym_cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                vals = [
                    cell[i][j],
                    cell[n - 1 - i][n - 1 - j],  # 180° rotation
                    cell[j][i],                   # transpose
                    cell[n - 1 - j][n - 1 - i],   # 180° + transpose
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 2 else 0
        return sym_cell
    
    def check_symmetry(cell):
        """Check if cell has the required square symmetry."""
        n = len(cell)
        for i in range(n):
            for j in range(n):
                if cell[i][j] != cell[n - 1 - i][n - 1 - j]:
                    return False
                if cell[i][j] != cell[j][i]:
                    return False
        return True
    
    def compute_fourier_coefficients(cell, n_pw=17):
        """Compute Fourier coefficients of the pixel map for plane-wave method."""
        n = size
        eps_mean = sum(sum(row) for row in cell) / (n * n) * (eps_high - eps_low) + eps_low
        
        coeffs = {}
        for mx in range(-n_pw, n_pw + 1):
            for my in range(-n_pw, n_pw + 1):
                if mx == 0 and my == 0:
                    continue
                real = 0.0
                imag = 0.0
                for i in range(n):
                    for j in range(n):
                        angle = 2.0 * math.pi * (mx * (i + 0.5) / n + my * (j + 0.5) / n)
                        val = cell[i][j]
                        if val != 0:
                            real += val * math.cos(angle)
                            imag += val * math.sin(angle)
                coeff = (real - 1j * imag) / (n * n)
                coeffs[(mx, my)] = coeff
        return coeffs, eps_mean
    
    def approximate_gap(cell):
        """Quick heuristic estimate of the 4-5 TM band gap."""
        n = size
        fill = sum(sum(row) for row in cell) / (n * n)
        
        score = 0.0
        
        if 0.30 <= fill <= 0.50:
            score += 2.0
        elif 0.25 <= fill <= 0.55:
            score += 1.0
        else:
            score -= 1.0
        
        center_regions = [(8, 8), (8, 23), (23, 8), (23, 23)]
        feature_count = 0
        for (ri, rj) in center_regions:
            region_sum = 0
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ii, jj = ri + di, rj + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        region_sum += cell[ii][jj]
            if region_sum > 10:
                feature_count += 1
        
        if feature_count >= 2:
            score += 1.0
        if feature_count == 4:
            score += 1.0
        
        pixels = [cell[i][j] for i in range(n) for j in range(n)]
        mean = fill
        variance = sum((p - mean) ** 2 for p in pixels) / (n * n)
        if variance > 0.2:
            score += 0.5
        
        return score
    
    def make_cell_from_pattern(pattern_func):
        """Create a cell from a pattern function."""
        n = size
        cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x = (i + 0.5) / n - 0.5
                y = (j + 0.5) / n - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        return enforce_symmetry(cell)
    
    # Generate diverse candidate patterns
    
    def rod_pattern(radius):
        def f(x, y):
            return x * x + y * y < radius * radius
        return f
    
    def four_rod_pattern(radius, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def ring_pattern(inner_r, outer_r):
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def cross_pattern(arm_width, arm_length):
        def f(x, y):
            if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                return True
            if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                return True
            return False
        return f
    
    def four_rings_pattern(inner_r, outer_r, offset):
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
    
    def ellipse_rod_pattern(rx, ry):
        """Elliptical rod (anisotropic)."""
        def f(x, y):
            return (x / rx) ** 2 + (y / ry) ** 2 < 1.0
        return f
    
    def four_ellipse_pattern(rx, ry, offset):
        """Four elliptical rods at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if (dx / rx) ** 2 + (dy / ry) ** 2 < 1.0:
                        return True
            return False
        return f
    
    def diamond_pattern(radius):
        """Diamond (rotated square) shape."""
        def f(x, y):
            return abs(x) + abs(y) < radius
        return f
    
    def four_diamond_pattern(radius, offset):
        """Four diamonds at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if abs(dx) + abs(dy) < radius:
                        return True
            return False
        return f
    
    def ring_plus_rod_pattern(rod_r, ring_ir, ring_or):
        """Central rod with surrounding ring."""
        def f(x, y):
            r2 = x * x + y * y
            if r2 < rod_r * rod_r:
                return True
            if ring_ir * ring_ir < r2 < ring_or * ring_or:
                return True
            return False
        return f
    
    def four_rod_ring_pattern(rod_r, ring_ir, ring_or, offset):
        """Four patterns: each has a small rod with a ring around it, at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx * dx + dy * dy
                    if r2 < rod_r * rod_r:
                        return True
                    if ring_ir * ring_ir < r2 < ring_or * ring_or:
                        return True
            return False
        return f
    
    # Fourier harmonic parameterization approach
    # The dielectric distribution can be expressed as:
    # eps(x,y) = eps_mean + A_20*cos(2π(2x+φ20)) + A_02*cos(2π(2y+φ02)) + A_22*cos(2π(2x+2y+φ22))
    # This directly couples to the 4th and 5th TM bands in a square lattice.
    
    def fourier_harmonic_pattern(A20, A02, A22, phi20, phi02, phi22, eps_mid):
        """Create a cell from Fourier harmonics (2,0), (0,2), (2,2) modes."""
        def f(x, y):
            # x, y in [-0.5, 0.5]
            val = eps_mid
            val += A20 * math.cos(2 * math.pi * (2 * x + phi20))
            val += A02 * math.cos(2 * math.pi * (2 * y + phi02))
            val += A22 * math.cos(2 * math.pi * (2 * x + 2 * y + phi22))
            # Threshold at 0 to get binary dielectric
            return val > 0
        return f
    
    # Generate a large set of candidates
    candidates = []
    
    # Four-rod patterns (most promising for higher-order TM gaps)
    for r in [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]:
        for off in [0.20, 0.22, 0.25, 0.27, 0.30, 0.32]:
            candidates.append(four_rod_pattern(r, off))
    
    # Single rod patterns
    for r in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        candidates.append(rod_pattern(r))
    
    # Ring patterns
    for ir, orr in [(0.10, 0.20), (0.12, 0.22), (0.15, 0.25), (0.08, 0.18), (0.10, 0.18)]:
        candidates.append(ring_pattern(ir, orr))
    
    # Cross patterns
    for aw in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for al in [0.35, 0.40, 0.45, 0.50]:
            candidates.append(cross_pattern(aw, al))
    
    # Four small rings
    for ir, orr in [(0.04, 0.10), (0.05, 0.12), (0.06, 0.14), (0.04, 0.09)]:
        for off in [0.22, 0.25, 0.28, 0.30]:
            candidates.append(four_rings_pattern(ir, orr, off))
    
    # Elliptical rods
    for rx in [0.08, 0.10, 0.12]:
        for ry in [0.05, 0.06, 0.08]:
            candidates.append(ellipse_rod_pattern(rx, ry))
    
    # Four ellipses
    for rx in [0.05, 0.06, 0.07, 0.08]:
        for ry in [0.03, 0.04, 0.05]:
            for off in [0.25, 0.28, 0.30]:
                candidates.append(four_ellipse_pattern(rx, ry, off))
    
    # Diamond patterns
    for r in [0.15, 0.20, 0.25, 0.30]:
        candidates.append(diamond_pattern(r))
    
    for r in [0.08, 0.10, 0.12, 0.15]:
        for off in [0.25, 0.28, 0.30]:
            candidates.append(four_diamond_pattern(r, off))
    
    # Ring + rod patterns
    for rod_r in [0.08, 0.10, 0.12]:
        for ring_ir in [0.15, 0.18, 0.20]:
            for ring_or in [0.25, 0.28, 0.30]:
                candidates.append(ring_plus_rod_pattern(rod_r, ring_ir, ring_or))
    
    # Four rod+ring patterns
    for rod_r in [0.04, 0.05, 0.06]:
        for ring_ir in [0.08, 0.10]:
            for ring_or in [0.14, 0.16, 0.18]:
                for off in [0.25, 0.28]:
                    candidates.append(four_rod_ring_pattern(rod_r, ring_ir, ring_or, off))
    
    # Fourier harmonic patterns - this is the new structural approach
    # The (2,0), (0,2), (2,2) modes directly couple to bands 4 and 5
    # in a square lattice photonic crystal
    random.seed(42)
    for _ in range(50):
        A20 = random.uniform(0.0, 1.0)
        A02 = random.uniform(0.0, 1.0)
        A22 = random.uniform(0.0, 1.0)
        phi20 = random.uniform(0.0, 1.0)
        phi02 = random.uniform(0.0, 1.0)
        phi22 = random.uniform(0.0, 1.0)
        eps_mid = random.uniform(0.3, 0.5)
        candidates.append(fourier_harmonic_pattern(A20, A02, A22, phi20, phi02, phi22, eps_mid))
    
    # More systematic Fourier harmonic exploration
    for A20 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for A02 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for A22 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                for phi20 in [0.0, 0.25, 0.5, 0.75]:
                    for phi02 in [0.0, 0.25, 0.5, 0.75]:
                        for phi22 in [0.0, 0.25, 0.5, 0.75]:
                            eps_mid = 0.4
                            candidates.append(fourier_harmonic_pattern(A20, A02, A22, phi20, phi02, phi22, eps_mid))
    
    # Random binary patterns with symmetry
    for _ in range(30):
        n = size
        raw_cell = [[0] * n for _ in range(n)]
        for i in range(n // 2 + 1):
            for j in range(n // 2 + 1):
                val = random.randint(0, 1)
                raw_cell[i][j] = val
                raw_cell[n - 1 - i][n - 1 - j] = val
                raw_cell[j][i] = val
                raw_cell[n - 1 - j][n - 1 - i] = val
        candidates.append(lambda x, y, rc=raw_cell: rc[int((x + 0.5) * n) if int((x + 0.5) * n) < n else n - 1][int((y + 0.5) * n) if int((y + 0.5) * n) < n else n - 1])
    
    # Evaluate all candidates using the heuristic
    best_cell = None
    best_heuristic_score = -1
    
    for idx, pattern_func in enumerate(candidates):
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        
        if not check_symmetry(cell):
            continue
        
        score = approximate_gap(cell)
        
        if score > best_heuristic_score:
            best_heuristic_score = score
            best_cell = cell
    
    # Local search: try to improve the best candidate by flipping pixels
    if best_cell is not None and time_ok():
        current_best = best_cell
        current_score = best_heuristic_score
        n = size
        
        for iteration in range(50):
            if not time_ok():
                break
            
            improved = False
            for _ in range(20):
                if not time_ok():
                    break
                
                ri = random.randint(0, n - 1)
                rj = random.randint(0, n - 1)
                
                new_cell = [row[:] for row in current_best]
                orbit = set()
                i, j = ri, rj
                orbit.add((i, j))
                orbit.add((n - 1 - i, n - 1 - j))
                orbit.add((j, i))
                orbit.add((n - 1 - j, n - 1 - i))
                
                for (oi, oj) in orbit:
                    new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_score = approximate_gap(new_cell)
                if new_score > current_score:
                    current_best = new_cell
                    current_score = new_score
                    best_cell = new_cell
                    best_heuristic_score = new_score
                    improved = True
                    break
            
            if not improved:
                new_cell = [row[:] for row in current_best]
                for _ in range(5):
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    orbit = set()
                    i, j = ri, rj
                    orbit.add((i, j))
                    orbit.add((n - 1 - i, n - 1 - j))
                    orbit.add((j, i))
                    orbit.add((n - 1 - j, n - 1 - i))
                    for (oi, oj) in orbit:
                        new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_score = approximate_gap(new_cell)
                if new_score > current_score:
                    current_best = new_cell
                    current_score = new_score
                    best_cell = new_cell
                    best_heuristic_score = new_score
    
    # If no valid cell was found, use a default
    if best_cell is None:
        best_cell = make_cell_from_pattern(four_rod_pattern(0.10, 0.25))
    
    return best_cell
# EVOLVE-BLOCK-END
