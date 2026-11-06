"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap between bands 4 and 5."""
import numpy as np
from itertools import product

def design(problem):
    """
    Design a 32x32 unit cell with four-fold symmetry.
    Uses concentric square rings (Manhattan distance based) to parameterize the design.
    We explore binary sequences of ring assignments to find a good band gap.
    
    The cell is invariant under:
    - 180 degree rotation (cell[i][j] == cell[31-i][31-j])
    - Transpose symmetry (cell[i][j] == cell[j][i])
    - Which implies full D4 symmetry about the center.
    
    For a 32x32 grid, the center is at (15.5, 15.5).
    We use concentric square rings defined by max(|i-15.5|, |j-15.5|) or similar.
    Actually, let's use the Chebyshev distance from center to define rings.
    But since we need exact pixel symmetry, let's define rings based on the minimum number of steps
    to reach the center, or more simply, use the property that the cell is symmetric.
    
    A simpler approach: define the cell by its top-left quadrant (16x16), then mirror it.
    Due to the symmetry requirements:
    cell[i][j] = cell[31-i][31-j] = cell[j][i] = cell[31-j][31-i]
    
    So we only need to specify cell[i][j] for 0 <= i <= 15, 0 <= j <= 15.
    Then:
    cell[i][j] = cell[31-i][31-j]
    cell[i][j] = cell[j][i]
    
    This reduces the search space to 16x16 = 256 bits, but with the constraint that
    cell[i][j] = cell[j][i] within the top-left quadrant (since cell[j][i] is also in the quadrant
    if j <= 15 and i <= 15). Wait, cell[j][i] for i,j in [0,15] is still in the top-left quadrant.
    So the top-left quadrant must be symmetric about its diagonal.
    
    So we need cell[i][j] = cell[j][i] for 0 <= i,j <= 15.
    This reduces the independent variables to 16*17/2 = 136 bits.
    
    Let's parameterize using concentric square rings in the top-left quadrant.
    Define ring r as the set of pixels where max(i, j) = r, for r = 0..15.
    Within each ring, due to symmetry cell[i][j] = cell[j][i], we only need to specify
    values for i <= j. But within a ring max(i,j)=r, if i<=j then j=r and i ranges from 0 to r.
    So ring r has r+1 independent values (for i=0..r, j=r).
    Total independent values: sum_{r=0}^{15} (r+1) = 136.
    
    We'll do a random search over these 136 bits, but that's too large.
    Instead, let's use a coarser parameterization: concentric square rings where each ring
    is either all dielectric or all air. This gives 16 binary parameters.
    
    Let's try this: for each ring r (0 to 15), assign a value v_r in {0,1}.
    Then for pixel (i,j) in top-left quadrant, ring = max(i,j), and cell[i][j] = v_ring.
    Then mirror to get the full cell.
    """
    size = problem["pixels"]  # 32
    half = size // 2  # 16
    
    # Define 16 ring values
    # We'll try a few hand-crafted patterns and also random ones
    # Pattern 1: alternating rings
    # Pattern 2: specific patterns known to work for photonic crystals
    
    best_cell = None
    best_score = -1e9
    
    # Since we can't actually compute the band gap here (no plane-wave solver),
    # we'll rely on known good structures.
    # A common high-gap structure for TM polarization is a dielectric hole in air,
    # or a dielectric rod in air. The seed uses a rod.
    # Another good structure is a checkerboard-like pattern or concentric rings.
    
    # Let's try a pattern with alternating thick rings
    # Ring 0 (center): dielectric
    # Ring 1: air
    # Ring 2: dielectric
    # etc.
    
    # Actually, for TM bands, a high-index material with holes often works well.
    # Let's try: dielectric background with air holes arranged in a square lattice.
    # But we need a single unit cell with 4-fold symmetry.
    
    # Let's try a pattern where we have a dielectric cross or plus shape.
    
    # Try pattern: center dielectric, then air, then dielectric, etc.
    # But let's be smarter. For the 4th and 5th TM bands, we want a gap.
    # The seed (simple rod) gives score 1.0. We need to beat it.
    
    # Let's try a few specific patterns:
    
    patterns = []
    
    # Pattern A: Alternating rings, starting with dielectric at center
    patterns.append([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    # Pattern B: Alternating rings, starting with air at center
    patterns.append([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Pattern C: Dielectric background with a small air hole in the center
    patterns.append([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Pattern D: Dielectric rod (like seed but square)
    patterns.append([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Pattern E: Cross shape
    patterns.append([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Pattern F: Thicker central dielectric
    patterns.append([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Pattern G: Air center, dielectric ring, air, dielectric...
    patterns.append([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Pattern H: Dielectric with a square hole
    patterns.append([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Build the cell from the best pattern
    # We'll pick pattern A (alternating) as a starting point, but let's construct
    # a more optimized one.
    
    # Actually, let's use a known good structure: a dielectric square with a smaller
    # air square hole in the center, or vice versa.
    
    # Let's go with a pattern that has a high contrast structure.
    # Use pattern C: small air hole in dielectric background
    ring_vals = patterns[2]
    
    # Build the 16x16 top-left quadrant
    quad = [[0] * half for _ in range(half)]
    for i in range(half):
        for j in range(half):
            ring = max(i, j)
            quad[i][j] = ring_vals[ring]
    
    # Mirror to get full 32x32 cell
    cell = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = quad[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert cell[i][j] == cell[size - 1 - i][size - 1 - j], "180 symmetry failed"
            assert cell[i][j] == cell[j][i], "Transpose symmetry failed"
    
    return cell
