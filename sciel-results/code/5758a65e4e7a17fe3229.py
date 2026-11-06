"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with a 2x2 basis of dielectric rods.
    This structure enhances the (2,0) and (0,2) Fourier components,
    which are critical for opening higher-order band gaps in square lattices.
    """
    size = problem["pixels"]
    # Use a 2x2 basis of circular rods to maximize higher-order gap components
    # Centers at (8,8), (8,23), (23,8), (23,23) for 32x32 cell
    # Radius 6 pixels
    centers = [(8, 8), (8, 23), (23, 8), (23, 23)]
    radius = 6.0
    r2 = radius * radius
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            is_dielectric = 0
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx * dx + dy * dy <= r2:
                    is_dielectric = 1
                    break
            row.append(is_dielectric)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
