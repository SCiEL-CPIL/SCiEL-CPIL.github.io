import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Define 4 circles centered at (+/-0.25, +/-0.25)
    # Radius r = 0.15
    r = 0.15
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            is_dielectric = 0
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    is_dielectric = 1
                    break
            row.append(is_dielectric)
        cell.append(row)
    return cell
