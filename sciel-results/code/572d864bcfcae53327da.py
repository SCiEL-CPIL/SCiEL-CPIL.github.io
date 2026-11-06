"""Rewritten program with a Dielectric Ring design to enhance TM band gap."""

def design(problem):
    """
    Design a 32x32 unit cell with a dielectric ring (annulus) to maximize the 
    TM band gap between the 4th and 5th bands.
    The ring is centered in the cell and invariant under D4 symmetry.
    """
    size = problem["pixels"]
    cell = []
    
    # Define ring parameters
    # Outer radius and inner radius in units of lattice constant (1.0)
    R_out = 0.45
    R_in = 0.20
    
    for i in range(size):
        row = []
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            r_sq = x * x + y * y
            # 1 if inside outer radius AND outside inner radius
            if r_sq < R_out * R_out and r_sq >= R_in * R_in:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
