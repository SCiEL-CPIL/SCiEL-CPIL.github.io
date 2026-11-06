<<<<<<< SEARCH
    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # 2x2 supercell arrangements
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]:
        candidates.append(generate_2x2_supercell(r))
    
    # 2x2 supercell with rings
    for inner, outer in [(0.05, 0.15), (0.08, 0.18), (0.10, 0.20), (0.06, 0.16), 
                         (0.04, 0.12), (0.07, 0.14), (0.09, 0.17), (0.11, 0.19)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Additional 2x2 supercell variants with different offsets
    def generate_2x2_supercell_offset(rod_radius_frac, offset_frac):
        """2x2 supercell with adjustable offset from center."""
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        offset = offset_frac * size
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell
    
    for r, off in [(0.12, 0.25), (0.14, 0.25), (0.16, 0.25), (0.10, 0.30), 
                   (0.12, 0.30), (0.14, 0.30), (0.10, 0.20), (0.12, 0.20)]:
        candidates.append(generate_2x2_supercell_offset(r, off))
    
    # 2x2 supercell with square holes (inverse pattern)
    def generate_2x2_supercell_inverse(rod_radius_frac, hole_radius_frac):
        """2x2 supercell: dielectric background with air holes."""
        cell = [[1] * size for _ in range(size)]
        r_hole = hole_radius_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_hole * r_hole:
                        cell[i][j] = 0
                        break
        return cell
    
    for r_hole in [0.05, 0.08, 0.10, 0.12, 0.15]:
        candidates.append(generate_2x2_supercell_inverse(0.15, r_hole))
    
    # Hexagonal-like patterns
    for r, s in [(0.08, 0.25), (0.10, 0.25), (0.12, 0.30), (0.08, 0.30), (0.10, 0.30)]:
        candidates.append(generate_hexagonal_like(r, s))
    
    # Diamond patterns
    for w, s in [(0.15, 0.40), (0.20, 0.35), (0.15, 0.30), (0.25, 0.40)]:
        candidates.append(generate_diamond_pattern(w, s))
    
    # Spectral parameterization candidates
    # Start with simple harmonic patterns
    spectral_candidates = []
    
    # Fundamental harmonic (m=1, n=1)
    spectral_candidates.append({(1, 1): 1.0, (-1, -1): 1.0, (1, -1): 0.5, (-1, 1): 0.5})
    # Second harmonic
    spectral_candidates.append({(2, 0): 1.0, (0, 2): 1.0, (-2, 0): 1.0, (0, -2): 1.0})
    # Mixed harmonics
    spectral_candidates.append({(1, 1): 1.0, (1, -1): 1.0, (-1, 1): 1.0, (-1, -1): 1.0})
    # Higher order
    spectral_candidates.append({(1, 1): 0.8, (2, 0): 0.5, (0, 2): 0.5, (-1, -1): 0.8, (-2, 0): 0.5, (0, -2): 0.5})
    
    for coeffs in spectral_candidates:
        cell = generate_spectral_cell_fast(coeffs)
        candidates.append(cell)
=======
    # Generate candidate designs
    candidates = []
    
    # === DUAL-STRUCTURE: Dielectric background with air holes in centered-square lattice ===
    def generate_centered_square_holes(hole_radius_frac, offset_frac, secondary_radius_frac=0.0):
        """
        Dielectric background (1) with air holes (0) in a centered-square lattice.
        Primary holes at cell corners/edges, secondary holes offset from center.
        Maintains D4 symmetry.
        """
        cell = [[1] * size for _ in range(size)]
        r_primary = hole_radius_frac * size
        r_secondary = secondary_radius_frac * size
        offset = offset_frac * size
        
        # Primary hole sites: at the 4 corners of the unit cell (and their periodic images)
        # For D4 symmetry about cell center, holes at (0,0), (0,size), (size,0), (size,size)
        # which are equivalent, plus potentially at edge midpoints
        primary_sites = [
            (0.0, 0.0), (size, 0.0), (0.0, size), (size, size),
            (size/2.0, 0.0), (size/2.0, size), (0.0, size/2.0), (size, size/2.0),
        ]
        
        # Secondary hole sites: offset from center in a square pattern
        secondary_sites = []
        if r_secondary > 0:
            secondary_sites = [
                (size/2.0 - offset, size/2.0 - offset),
                (size/2.0 + offset, size/2.0 - offset),
                (size/2.0 - offset, size/2.0 + offset),
                (size/2.0 + offset, size/2.0 + offset),
            ]
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                is_hole = False
                for cx, cy in primary_sites:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_primary * r_primary:
                        is_hole = True
                        break
                if not is_hole:
                    for cx, cy in secondary_sites:
                        dx = x - cx
                        dy = y - cy
                        if dx * dx + dy * dy < r_secondary * r_secondary:
                            is_hole = True
                            break
                if is_hole:
                    cell[i][j] = 0
        return cell
    
    # Explore centered-square lattice with various parameters
    for hp in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        for off in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for rs in [0.0, 0.05, 0.08, 0.10, 0.12]:
                candidates.append(generate_centered_square_holes(hp, off, rs))
    
    # Simpler: just primary holes with varying radius
    for hp in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]:
        candidates.append(generate_centered_square_holes(hp, 0.0, 0.0))
    
    # Secondary holes only (no primary)
    for off in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for rs in [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
            candidates.append(generate_centered_square_holes(0.0, off, rs))
    
    # Mixed: moderate primary + secondary
    for hp in [0.10, 0.12, 0.14, 0.16]:
        for off in [0.15, 0.20, 0.25]:
            for rs in [0.06, 0.08, 0.10, 0.12]:
                candidates.append(generate_centered_square_holes(hp, off, rs))
    
    # Keep some of the original candidates for comparison
    for r in [0.15, 0.20, 0.25, 0.30, 0.35]:
        candidates.append(generate_single_rod(r))
    
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    for r in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]:
        candidates.append(generate_2x2_supercell(r))
    
    for r_hole in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        candidates.append(generate_2x2_supercell_inverse(0.15, r_hole))
    
    # Spectral parameterization candidates
    spectral_candidates = []
    spectral_candidates.append({(1, 1): 1.0, (-1, -1): 1.0, (1, -1): 0.5, (-1, 1): 0.5})
    spectral_candidates.append({(2, 0): 1.0, (0, 2): 1.0, (-2, 0): 1.0, (0, -2): 1.0})
    spectral_candidates.append({(1, 1): 1.0, (1, -1): 1.0, (-1, 1): 1.0, (-1, -1): 1.0})
    spectral_candidates.append({(1, 1): 0.8, (2, 0): 0.5, (0, 2): 0.5, (-1, -1): 0.8, (-2, 0): 0.5, (0, -2): 0.5})
    
    for coeffs in spectral_candidates:
        cell = generate_spectral_cell_fast(coeffs)
        candidates.append(cell)
>>>>>>> REPLACE

