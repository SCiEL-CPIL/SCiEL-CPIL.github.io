# EVOLVE-BLOCK-START
def design(problem):
    """A centered square ring of dielectric."""
    size = problem["pixels"]
    cell = []
    # Center is at (size-1)/2.0
    center = (size - 1) / 2.0
    inner_r = 4.0
    outer_r = 12.0
    
    for i in range(size):
        row = []
        for j in range(size):
            # Manhattan distance? No, Chebyshev distance for square shape
            # Chebyshev distance to center
            dx = abs(i - center)
            dy = abs(j - center)
            r = max(dx, dy)
            
            if inner_r <= r < outer_r:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
