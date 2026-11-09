"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell with four circular dielectric rods located at the 
    quarter-points of the cell. This 'superlattice' configuration is known to 
    open larger band gaps in square-lattice photonic crystals compared to a 
    single central rod, particularly for TM polarization.
    """
    size = problem["pixels"]
    
    # Centers of the four circular rods in pixel coordinates
    # The cell is 32x32. The symmetry center is (15.5, 15.5).
    # The rods are placed at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    # in normalized coordinates [0, 1).
    # Pixel center (j+0.5)/32.
    # 0.25 * 32 = 8. So we want centers around pixel index 7.5?
    # (j+0.5)/32 = 0.25 => j+0.5 = 8 => j = 7.5.
    # So the geometric center of the rod should be at x=8.0, y=8.0 etc.
    # Let's use float coordinates for the centers.
    centers = [
        (8.0, 8.0),
        (24.0, 8.0),
        (8.0, 24.0),
        (24.0, 24.0)
    ]
    
    # Radius of the rods in pixels
    # Target fill fraction ~0.35
    # Area of 4 circles = 4 * pi * r^2
    # Total area = 32 * 32 = 1024
    # 4 * pi * r^2 = 0.35 * 1024 => r^2 = 0.35 * 1024 / (4 * pi) = 358.4 / 12.56 = 28.5
    # r = sqrt(28.5) = 5.34
    radius = 5.3
    
    cell = []
    for i in range(size):
        # Coordinate in [0, 1) mapped to pixel grid
        # We check if the pixel center is inside any of the circles
        # Pixel center y is (i + 0.5)
        y_pix = i + 0.5
        row = []
        for j in range(size):
            x_pix = j + 0.5
            inside = False
            for cx, cy in centers:
                dx = x_pix - cx
                dy = y_pix - cy
                if dx*dx + dy*dy < radius*radius:
                    inside = True
                    break
            if inside:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
