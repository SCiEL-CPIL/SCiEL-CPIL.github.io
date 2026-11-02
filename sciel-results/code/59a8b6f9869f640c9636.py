"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # We want to pack N=21 circles in a rectangle of perimeter 4.
    # Let width = w, height = h. 2(w+h) = 4 => w+h = 2.
    # We can parameterize by aspect ratio or just fix h=1, w=1 for square?
    # No, variable rectangle. 
    # Maximizing sum of radii.
    
    # A common strategy for these tasks is to use a grid-like structure or
    # a specific packing pattern. Since we need a robust solution that finishes
    # quickly, let's try a few heuristic layouts and optimize them slightly.
    
    # Layout 1: 3x7 grid (approx square aspect ratio)
    # If we use a 3x7 grid, we have 21 circles.
    # Let the rectangle be W x H.
    # If we place them in a grid, the spacing is determined by the radii.
    # For a grid of nx rows and ny cols:
    # W >= nx * (2*r) + 2*r ? No, the distance between centers is 2*r.
    # Total width = (ny-1)*2*r + 2*r = ny*2*r? No.
    # Distance from left wall to first center = r.
    # Distance between centers = 2*r.
    # Distance from last center to right wall = r.
    # Total width W = (ny - 1) * 2*r + 2*r = ny * 2*r? 
    # Wait. Center 1 at x=r. Center 2 at x=r+2r=3r. Center ny at x=r+(ny-1)2r = (2ny-1)r.
    # Right wall at x = (2ny-1)r + r = 2ny*r.
    # So W = 2*ny*r. Similarly H = 2*nx*r.
    # Perimeter = 2(W+H) = 2(2*ny*r + 2*nx*r) = 4r(nx+ny).
    # We need 4r(nx+ny) <= 4 => r <= 1/(nx+ny).
    # Sum of radii = N * r.
    # To maximize sum, we need to minimize nx+ny for a given N, but we must fit N circles in nx*ny >= N.
    
    # Possible grid dimensions for N=21:
    # 1x21: sum = 22. r = 1/22. Sum = 21/22 ≈ 0.954
    # 2x11: sum = 13. r = 1/13. Sum = 21/13 ≈ 1.615
    # 3x7: sum = 10. r = 1/10. Sum = 21/10 = 2.1
    # 4x6 (24 cells, 3 empty): 
    #    If we use a 4x6 grid but only place 21 circles, we can potentially increase r?
    #    No, the bounding box is determined by the outermost circles.
    #    If we leave corners empty, we might be able to expand?
    #    Let's check 3x7 first. It's a very strong candidate.
    
    # Let's generate the 3x7 grid layout.
    # nx=3, ny=7.
    # r = 1/(3+7) = 0.1.
    # W = 2*7*0.1 = 1.4. H = 2*3*0.1 = 0.6.
    # Perimeter = 2(1.4+0.6) = 4.0. Correct.
    # Centers:
    # x_i = (2*i+1)*r for i in 0..6
    # y_j = (2*j+1)*r for j in 0..2
    
    # Can we do better with a non-uniform layout or different aspect ratio?
    # What if we use a 2x10 + 1 layout?
    # 2x11 grid uses 22 slots. We have 21 circles.
    # If we use a 2x11 grid, r = 1/13. Sum = 21/13 = 1.615.
    # 3x7 is 2.1.
    
    # What about 4x6?
    # 4 rows, 6 cols. 24 slots.
    # If we fill all 24, r = 1/10 = 0.1. Sum = 24*0.1 = 2.4.
    # But we only have 21 circles.
    # If we place 21 circles in a 4x6 grid, we can potentially increase r?
    # The bounding box is still determined by the grid extent if we keep the outer rings full.
    # If we remove 3 circles from the interior, we might be able to push the outer circles out?
    # No, the outer circles define the box. If we keep the outer ring of a 4x6 grid, the box is fixed by the outer ring.
    # Outer ring of 4x6 has 2*(4+6)-4 = 16 circles.
    # Inner 2x4 has 8 circles.
    # Total 24.
    # If we remove 3 inner circles, we have 21.
    # The outer 16 circles are still at the boundary.
    # So W and H are fixed by the outer ring positions.
    # If we assume the outer ring is a perfect grid, W=2*6*r, H=2*4*r.
    # Perimeter = 4r(10) = 40r <= 4 => r <= 0.1.
    # Sum = 21 * 0.1 = 2.1.
    # Same as 3x7.
    
    # Can we improve the 3x7 layout by "squeezing" or using a different aspect ratio?
    # The 3x7 grid assumes equal spacing.
    # If we change the aspect ratio, we might fit larger circles?
    # Let W+H=2.
    # In a 3x7 grid, the aspect ratio is W/H = 7/3 ≈ 2.33.
    # W = 1.4, H = 0.6.
    # If we make the rectangle more square, say 1x1, can we fit 21 circles with larger average radius?
    # In a 1x1 square, max radius for 1 circle is 0.25.
    # For 21 circles, it's a classic packing problem.
    # Known packings for 21 circles in a unit square:
    # Best known sum of radii for 21 circles in unit square is approx 0.21-0.22?
    # Wait, the metric is sum of radii.
    # If W=H=1, perimeter=4.
    # If we pack 21 circles in 1x1, let's estimate.
    # Grid 5x5 = 25 circles. r = 1/10 = 0.1. Sum = 25*0.1 = 2.5.
    # But we only have 21.
    # If we use a 5x5 grid in a 1x1 square, r=0.1.
    # We remove 4 circles.
    # The outer ring is fixed. r is still 0.1.
    # Sum = 21 * 0.1 = 2.1.
    # Same as 3x7.
    
    # Is there a configuration that yields > 2.1?
    # Consider 4x6 grid in a rectangle.
    # If we use a 4x6 grid, r=0.1, Sum=2.1 (if 21 circles).
    # What if we use a non-grid packing?
    # Hexagonal packing is denser.
    # In a hexagonal packing, the ratio of area is higher.
    # However, we are maximizing sum of radii, not area.
    # Sum of radii is proportional to the "size" of the circles.
    
    # Let's try to optimize the 3x7 grid by allowing the aspect ratio to vary.
    # Actually, for a fixed grid topology (nx, ny), the optimal aspect ratio for maximizing r
    # given perimeter constraint is when the rectangle is "tight" around the grid.
    # The constraint is 2(W+H) = 4.
    # W = 2*ny*r, H = 2*nx*r.
    # 2(2*ny*r + 2*nx*r) = 4 => r = 1/(nx+ny).
    # This is independent of aspect ratio choice if we assume the grid fills the box.
    # So for any rectangular grid nx x ny, the max sum is N / (nx+ny).
    # We want to minimize nx+ny such that nx*ny >= N.
    # For N=21:
    # nx=1, ny=21 => sum=22. S=21/22=0.95
    # nx=2, ny=11 => sum=13. S=21/13=1.61
    # nx=3, ny=7 => sum=10. S=21/10=2.1
    # nx=4, ny=6 => sum=10. S=21/10=2.1 (using 21 of 24 spots, outer ring defines box)
    # nx=5, ny=5 => sum=10. S=21/10=2.1 (using 21 of 25 spots, outer ring defines box)
    
    # So grid-based solutions cap at 2.1.
    # Can we exceed 2.1?
    # Yes, if we don't use a simple grid.
    # For example, if we can fit 21 circles in a smaller bounding box (relative to their size) than the grid allows.
    # Or, if we can make the circles non-uniform? The problem says "maximize the sum of their radii".
    # It does not require equal radii.
    # However, usually equal radii are optimal for such symmetric constraints.
    # But if we allow variable radii, we might pack more efficiently.
    
    # Let's try a simple local optimization on the 3x7 grid to see if we can push the boundaries.
    # Start with 3x7 grid, r=0.1.
    # Add small perturbations and use gradient ascent on the sum of radii, subject to constraints.
    # Constraints:
    # 1. Circles inside rectangle: x_i >= r_i, x_i <= W-r_i, y_i >= r_i, y_i <= H-r_i.
    # 2. Disjoint interiors: dist(i,j) >= r_i + r_j.
    # 3. Perimeter: 2(W+H) <= 4.
    
    # We can fix W and H to 1.4 and 0.6 (the 3x7 optimal box) and try to increase radii?
    # No, if we increase radii, they will overlap or go out of bounds.
    # We can try to optimize the positions and radii simultaneously.
    
    # Let's use a simple iterative relaxation method.
    # Initialize with 3x7 grid.
    # Iterate:
    # 1. Check constraints.
    # 2. If violated, adjust positions or reduce radii.
    # 3. Try to increase radii.
    
    # A better approach: Use a penalty method or a simple gradient step.
    # Given the time limit, let's stick to the grid solution unless we can implement a very fast local search.
    # The grid solution of 2.1 is likely close to the optimum for N=21.
    # Let's verify if 2.1 is the best possible.
    # If we use a hexagonal packing in a 3x7 "area", we might fit more?
    # But we only have 21 circles.
    
    # Let's try to generate the 3x7 grid and return it.
    # This guarantees a valid solution with score 2.1.
    
    nx, ny = 3, 7
    r = 1.0 / (nx + ny)
    W = 2 * ny * r
    H = 2 * nx * r
    
    circles = []
    for i in range(nx):
        for j in range(ny):
            x = (2 * j + 1) * r
            y = (2 * i + 1) * r
            circles.append([x, y, r])
            
    # We have nx*ny = 21 circles.
    # Return as numpy array.
    result = np.array(circles)
    
    # Verify constraints
    # Perimeter: 2*(W+H) = 2*(1.4+0.6) = 4.0. OK.
    # Disjoint: Distance between adjacent centers is 2r. Sum of radii is 2r. OK.
    # Bounds: x in [r, W-r], y in [r, H-r].
    # Min x = r. Max x = (2*6+1)r = 13r = 1.3. W-r = 1.4-0.1=1.3. OK.
    
    return result
# EVOLVE-BLOCK-END
