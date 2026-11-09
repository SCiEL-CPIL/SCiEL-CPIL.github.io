"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell with four square dielectric rods located at the 
    quarter-points of the cell. This 'superlattice' configuration with square 
    rods modifies the Fourier coefficients compared to circular rods, which can 
    optimize the band gap between the fourth and fifth TM bands.
    """
    size = problem["pixels"]
    
    # Centers of the four square rods in pixel coordinates
    # Symmetric about the cell center (15.5, 15.5)
    # Placed at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    centers = [
        (8.0, 8.0),
        (24.0, 8.0),
        (8.0, 24.0),
        (24.0, 24.0)
    ]
    
    # Half-width of the square rods
    # Target fill fraction ~0.39 (10x10 pixels per rod, 4 rods -> 400/1024)
    # This is slightly higher fill than the circular seed (0.35) to compensate 
    # for the shape difference and potentially widen the gap.
    half_width = 5.0
    
    cell = []
    for i in range(size):
        y_pix = i + 0.5
        row = []
        for j in range(size):
            x_pix = j + 0.5
            inside = False
            for cx, cy in centers:
                dx = abs(x_pix - cx)
                dy = abs(y_pix - cy)
                if dx < half_width and dy < half_width:
                    inside = True
                    break
            if inside:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
