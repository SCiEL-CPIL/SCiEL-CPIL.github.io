"""Program for designing a photonic-crystal unit cell with a large TM band gap."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a D4-symmetric 32x32 unit cell using a cellular automaton-inspired
    approach to create complex microstructures that enhance higher-order band gaps.
    
    The approach:
    1. Start with a simple symmetric seed pattern
    2. Evolve it with a custom 2D cellular automaton rule
    3. Enforce D4 symmetry (invariance under rotations and reflections)
    4. Tune the dielectric fraction for optimal band gap
    
    The CA dynamics create aperiodic-like features with rich high-frequency
    Fourier content, which can enhance Bragg scattering for higher-order bands.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # We'll use a parameterized approach: try different CA evolution steps
    # and dielectric fractions to find a good configuration
    
    def apply_d4_symmetry(cell):
        """Enforce D4 symmetry: cell == cell[::-1] row-wise, column-wise, and transpose."""
        # Start with the cell
        sym_cell = np.array(cell, dtype=np.float64)
        
        # Average over all D4 symmetry operations
        # D4 group has 8 elements: identity, rotations by 90, 180, 270, 
        # and reflections (horizontal, vertical, diagonal, anti-diagonal)
        
        # Identity
        avg = np.zeros_like(sym_cell)
        count = 0
        
        # Identity
        avg += sym_cell
        count += 1
        
        # 90 degree rotation
        avg += np.rot90(sym_cell, 1)
        count += 1
        
        # 180 degree rotation
        avg += np.rot90(sym_cell, 2)
        count += 1
        
        # 270 degree rotation
        avg += np.rot90(sym_cell, 3)
        count += 1
        
        # Horizontal reflection (flip up-down)
        avg += sym_cell[::-1, :]
        count += 1
        
        # Vertical reflection (flip left-right)
        avg += sym_cell[:, ::-1]
        count += 1
        
        # Diagonal reflection (transpose)
        avg += sym_cell.T
        count += 1
        
        # Anti-diagonal reflection
        avg += sym_cell[::-1, ::-1].T
        count += 1
        
        avg /= count
        
        # Threshold to binary
        return (avg > 0.5).astype(int)
    
    def evolve_ca(cell, steps, rule_type="custom"):
        """
        Evolve a 2D cellular automaton for a given number of steps.
        Uses a custom rule that promotes complex pattern formation.
        """
        cell = np.array(cell, dtype=np.int32)
        h, w = cell.shape
        
        for _ in range(steps):
            new_cell = np.zeros_like(cell)
            for i in range(h):
                for j in range(w):
                    # Get 8 neighbors (with periodic boundary)
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % h
                            nj = (j + dj) % w
                            neighbors += cell[ni, nj]
                    
                    center = cell[i, j]
                    
                    # Custom rule: promote diversity and complex patterns
                    # Rule: cell becomes 1 if:
                    # - It's 1 and has 2-5 neighbors (survive)
                    # - It's 0 and has exactly 3 neighbors (birth)
                    # - Or it's 1 and has exactly 6 neighbors (special survival for complexity)
                    if center == 1:
                        if 2 <= neighbors <= 5 or neighbors == 6:
                            new_cell[i, j] = 1
                        else:
                            new_cell[i, j] = 0
                    else:
                        if neighbors == 3:
                            new_cell[i, j] = 1
                        else:
                            new_cell[i, j] = 0
            
            cell = new_cell
        
        return cell
    
    def create_seed_pattern(size, pattern_type=0):
        """Create an initial symmetric seed pattern."""
        cell = np.zeros((size, size), dtype=np.int32)
        cx = size // 2 - 0.5
        cy = size // 2 - 0.5
        
        if pattern_type == 0:
            # Central cross
            mid = size // 2
            cell[mid-1:mid+1, :] = 1
            cell[:, mid-1:mid+1] = 1
        elif pattern_type == 1:
            # Central ring
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - cx)**2 + (j - cx)**2)
                    if 0.25 * size < dist < 0.35 * size:
                        cell[i, j] = 1
        elif pattern_type == 2:
            # Four small squares at quarter positions
            offset = size // 4
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    ci = int(cx + di * offset)
                    cj = int(cx + dj * offset)
                    r = max(1, size // 16)
                    for di2 in range(-r, r+1):
                        for dj2 in range(-r, r+1):
                            ni = ci + di2
                            nj = cj + dj2
                            if 0 <= ni < size and 0 <= nj < size:
                                cell[ni, nj] = 1
        elif pattern_type == 3:
            # Checkerboard-like pattern with symmetry
            for i in range(size):
                for j in range(size):
                    if (i // 4 + j // 4) % 2 == 0:
                        cell[i, j] = 1
        elif pattern_type == 4:
            # Concentric rings
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - cx)**2 + (j - cx)**2)
                    ring = int(dist / (size / 8))
                    if ring % 2 == 0 and dist < 0.45 * size:
                        cell[i, j] = 1
        elif pattern_type == 5:
            # Star pattern
            for i in range(size):
                for j in range(size):
                    dx = i - cx
                    dy = j - cx
                    dist = np.sqrt(dx*dx + dy*dy)
                    angle = np.arctan2(dy, dx)
                    # 8-pointed star
                    star_val = np.abs(np.sin(4 * angle))
                    if dist < 0.4 * size and star_val > 0.5:
                        cell[i, j] = 1
                    elif dist < 0.15 * size:
                        cell[i, j] = 1
        elif pattern_type == 6:
            # Grid of dots
            spacing = size // 6
            for i in range(0, size, spacing):
                for j in range(0, size, spacing):
                    r = max(1, spacing // 3)
                    for di in range(-r, r+1):
                        for dj in range(-r, r+1):
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if di*di + dj*dj <= r*r:
                                    cell[ni, nj] = 1
        elif pattern_type == 7:
            # Diamond pattern
            for i in range(size):
                for j in range(size):
                    dist = abs(i - cx) + abs(j - cx)
                    if dist < 0.3 * size or (0.4 * size < dist < 0.5 * size):
                        cell[i, j] = 1
        
        return cell
    
    # Try multiple configurations and pick the best
    # Since we can't evaluate the band structure here, we'll use heuristics
    # to select a promising configuration
    
    best_cell = None
    best_score = -1
    
    # Try different seed patterns and CA evolution steps
    for pattern_type in range(8):
        seed = create_seed_pattern(size, pattern_type)
        
        # Try different numbers of CA evolution steps
        for steps in [0, 1, 2, 3, 4, 5, 6, 8]:
            evolved = evolve_ca(seed, steps)
            
            # Apply D4 symmetry
            sym_cell = apply_d4_symmetry(evolved)
            
            # Calculate dielectric fraction
            frac = np.sum(sym_cell) / (size * size)
            
            # Heuristic score: prefer moderate dielectric fractions
            # and high structural complexity (measured by edge density)
            
            # Edge density: number of pixels that have both 0 and 1 neighbors
            edge_count = 0
            for i in range(size):
                for j in range(size):
                    center_val = sym_cell[i, j]
                    has_diff_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % size
                            nj = (j + dj) % size
                            if sym_cell[ni, nj] != center_val:
                                has_diff_neighbor = True
                                break
                        if has_diff_neighbor:
                            break
                    if has_diff_neighbor:
                        edge_count += 1
            
            edge_density = edge_count / (size * size)
            
            # Score: balance between dielectric fraction and edge density
            # Optimal fraction is around 0.3-0.5 for good band gaps
            frac_score = 1.0 - abs(frac - 0.4) / 0.4
            edge_score = edge_density
            
            # Combine scores
            score = 0.4 * frac_score + 0.6 * edge_score
            
            if score > best_score:
                best_score = score
                best_cell = sym_cell
    
    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result

# EVOLVE-BLOCK-END
