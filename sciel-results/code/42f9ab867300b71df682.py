import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # New Approach: Single Air Hole in Dielectric Background
    # This inverts the contrast of the current program (Dielectric in Air -> Air in Dielectric)
    # and uses a single large feature instead of 4 small ones.
    # This is the standard configuration for maximizing TM band gaps in square lattices.
    
    center = (size - 1) / 2.0
    # Radius optimization: r/a ~ 0.3 is typical for max gap
    r = 9.5
    
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            if x * x + y * y < r * r:
                row.append(0) # Air hole
            else:
                row.append(1) # Dielectric background
        cell.append(row)
    return cell
