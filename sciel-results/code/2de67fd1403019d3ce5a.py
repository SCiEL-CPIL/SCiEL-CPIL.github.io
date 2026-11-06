"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses a pattern based on a square lattice with dielectric squares arranged
    to maximize the TM band gap between the 4th and 5th bands.
    
    The design uses a checkerboard-like pattern of dielectric squares in a
    supercell arrangement, which is known to open large band gaps in 2D
    photonic crystals.
    """
    size = problem["pixels"]
    half = size // 2
    
    # Create a pattern that has 4-fold symmetry (square symmetry group)
    # We'll use a design based on a "holey" dielectric pattern
    # where we have dielectric regions forming a specific shape
    
    # Strategy: Use a pattern inspired by the "Yablonovitch" or "Jussila" 
    # type structures. For TM polarization, we want strong electric field
    # confinement.
    
    # Let's try a design with 4 dielectric squares per unit cell arranged
    # in a specific pattern. The key is to have the right balance of
    # dielectric fraction and spatial arrangement.
    
    # Approach: Create a pattern where we have a central dielectric region
    # with holes, or vice versa. Let's try a "cross" pattern or a pattern
    # with 4-fold rotational symmetry.
    
    # Design: 4 small dielectric squares arranged around the center
    # This creates a pattern with good band gap properties
    
    # Actually, let me think about this more carefully. The TM band gap
    # between bands 4 and 5 suggests we need a specific dielectric fraction.
    # For a square lattice photonic crystal, optimal TM gaps often occur
    # with dielectric fraction around 0.3-0.5.
    
    # Let's try a pattern with 4 dielectric blobs arranged at the corners
    # of the unit cell (with 4-fold symmetry about the center)
    
    cell = [[0] * size for _ in range(size)]
    
    # Create 4 dielectric squares arranged with 4-fold symmetry
    # Each square is centered at (±d, ±d) from the cell center
    # with side length s
    
    # For 4-fold symmetry about center (16, 16) in 32x32:
    # We place squares at positions that are symmetric under 90° rotation
    
    # Let's use a pattern where we have a dielectric "ring" or "frame"
    # with specific geometry
    
    # Alternative: Use a pattern based on a mathematical function that
    # has 4-fold symmetry. For example, |sin(πx)| * |sin(πy)| type patterns
    
    # Let's try: dielectric where |x - 0.5| * |y - 0.5| < threshold
    # This creates a cross-like pattern with 4-fold symmetry
    
    # Actually, let me try a more structured approach:
    # Place 4 dielectric squares of side length L at positions
    # (0.5 ± a, 0.5 ± a) where a controls the offset from center
    
    L = 12  # side length of each square in pixels
    a = 5   # offset from center (in pixels)
    
    # The 4 squares are centered at:
    # (16 - a - L/2 + 0.5, 16 - a - L/2 + 0.5) -> top-left
    # (16 + a + L/2 - 0.5, 16 - a - L/2 + 0.5) -> top-right
    # (16 + a + L/2 - 0.5, 16 + a + L/2 - 0.5) -> bottom-right
    # (16 - a - L/2 + 0.5, 16 + a + L/2 - 0.5) -> bottom-left
    
    # Wait, this doesn't have the right symmetry. Let me reconsider.
    
    # For 4-fold rotational symmetry about the center (15.5, 15.5) in 0-indexed:
    # A point (i, j) maps to (j, size-1-i), (size-1-i, size-1-j), (size-1-j, i)
    
    # Let me use a simpler approach: define the pattern in the first quadrant
    # and mirror it
    
    # Design: A dielectric pattern that looks like 4 "petals" or a flower
    # using the function: max(|i - c|, |j - c|) < r where c is center
    
    # Actually, let's try a well-known good pattern: a square with rounded
    # corners, or a pattern based on a superellipse
    
    # Let me try: dielectric where (|i - c|^n + |j - c|^n)^(1/n) < r
    # with n = 4 (superellipse) for better band gap properties
    
    c = (size - 1) / 2.0  # center = 15.5
    n = 4.0  # superellipse exponent
    r = 10.0  # radius
    
    for i in range(size):
        for j in range(size):
            di = abs(i - c)
            dj = abs(j - c)
            val = (di**n + dj**n)**(1.0/n)
            if val < r:
                cell[i][j] = 1
    
    # Verify symmetry
    # Check row reversal: cell[i][j] == cell[size-1-i][j]
    # Check col reversal: cell[i][j] == cell[i][size-1-j]  
    # Check transpose: cell[i][j] == cell[j][i]
    
    # The superellipse pattern is symmetric under all these operations
    # since |i - c| and |j - c| are symmetric about the center
    
    return cell
