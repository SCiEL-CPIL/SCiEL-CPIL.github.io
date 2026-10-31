"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21
DEADLINE = 110.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # Perimeter = 2*(w+h) = 4 => w+h = 2
    # We can choose aspect ratio. Let's try to find a good packing.
    # For 21 circles, a 5x4 grid (20) + 1 extra, or 7x3 grid (21) might work.
    # 7x3 grid: 7 columns, 3 rows.
    # If we use a 7x3 grid, the rectangle aspect ratio should be roughly 7:3.
    # Let width = w, height = h. w + h = 2.
    # For 7 columns and 3 rows, if circles are radius r,
    # w ~ 7*2r + margin, h ~ 3*2r + margin.
    # Actually, for a grid, the centers are at (i + 0.5)*2r, (j + 0.5)*2r.
    # The bounding box width is (cols-1)*2r + 2r = cols*2r.
    # The bounding box height is (rows-1)*2r + 2r = rows*2r.
    # So w = cols * 2r, h = rows * 2r.
    # w + h = (cols + rows) * 2r = 2 => r = 1 / (cols + rows).
    
    # Let's try different grid configurations and pick the one with max sum of radii.
    # Sum of radii = N * r. So we want to maximize r, which means minimizing cols+rows.
    # For N=21, possible grids:
    # 1x21: sum=22, r=1/22
    # 2x11: sum=13, r=1/13
    # 3x7: sum=10, r=1/10
    # 4x6: 24 cells, use 21. sum=10, r=1/10. But we have 3 extra cells.
    # Wait, the grid size determines the rectangle size if we fill it.
    # If we use a 4x6 grid but only place 21 circles, we might be able to make them larger?
    # No, the bounding box is determined by the outermost circles.
    # If we leave some cells empty, we might be able to push the outer circles further apart?
    # No, the bounding box is the minimum rectangle containing all circles.
    # If we remove a corner circle, the bounding box might shrink, allowing larger radii?
    # Or if we remove a circle, the remaining ones can be rearranged.
    
    # Let's try a few simple grid layouts and calculate the sum of radii.
    # Layout 1: 3x7 grid (21 circles).
    # cols=7, rows=3. r = 1 / (7+3) = 0.1. Sum = 21 * 0.1 = 2.1.
    # Layout 2: 4x6 grid (24 cells, use 21).
    # If we use 4 rows and 6 columns, but only 21 circles.
    # If we keep the outer boundary fixed by the outer circles, the radius is determined by the grid.
    # r = 1 / (6+4) = 0.1. Sum = 21 * 0.1 = 2.1.
    # Can we do better?
    # What if we use a non-grid layout?
    
    # Let's try to optimize a 3x7 grid with small perturbations.
    # Or try a different aspect ratio.
    # The perimeter constraint is w+h=2.
    # For a 3x7 grid, the natural aspect ratio is 7:3.
    # w = 7*2r, h = 3*2r. w+h = 20r = 2 => r=0.1.
    
    # Let's try 2x11 grid?
    # w = 11*2r, h = 2*2r. w+h = 26r = 2 => r = 1/13 ~ 0.0769. Sum = 21/13 ~ 1.615. Worse.
    
    # So 3x7 or 4x6 seems best for grids.
    # Is there a better packing?
    # Hexagonal packing is denser.
    # In a hexagonal packing, the vertical distance between rows is sqrt(3)*r.
    # Horizontal distance is 2r.
    # Let's try a hexagonal layout for 21 circles.
    # Rows of 7, 7, 7?
    # Row 0: 7 circles at y=0. x = 0, 2r, 4r, ...
    # Row 1: 7 circles at y=sqrt(3)r. x = r, 3r, 5r, ... (shifted by r)
    # Row 2: 7 circles at y=2*sqrt(3)r. x = 0, 2r, ...
    # Bounding box width:
    # Max x in row 0: 12r (if 7 circles: 0, 2, 4, 6, 8, 10, 12).
    # Max x in row 1: 13r (if shifted: r, 3r, ..., 13r).
    # So width = 13r + r (radius) = 14r? No, the circle extends to x+r.
    # Leftmost x=0, extends to -r. Rightmost x=13r, extends to 14r.
    # So width = 14r.
    # Height:
    # Min y=0, extends to -r. Max y=2*sqrt(3)r, extends to 2*sqrt(3)r + r.
    # Height = 2*sqrt(3)r + 2r = 2r(sqrt(3)+1).
    # Perimeter: w+h = 14r + 2r(sqrt(3)+1) = 2r(7 + sqrt(3) + 1) = 2r(8+sqrt(3)).
    # 2r(8+sqrt(3)) = 2 => r = 1 / (8+sqrt(3)) ~ 1 / 9.732 ~ 0.1027.
    # Sum = 21 * 0.1027 ~ 2.157.
    # This is better than 2.1!
    
    # Let's verify the hexagonal packing dimensions.
    # Row 0: 7 circles. Centers at x_i = 2*i*r for i=0..6. y=0.
    # Row 1: 7 circles. Centers at x_i = (2*i+1)*r for i=0..6. y=sqrt(3)r.
    # Row 2: 7 circles. Centers at x_i = 2*i*r for i=0..6. y=2*sqrt(3)r.
    
    # Check overlaps:
    # Row 0 and Row 1:
    # Circle at (2*i*r, 0) and ((2*j+1)*r, sqrt(3)r).
    # Distance squared: (2*i*r - (2*j+1)*r)^2 + (sqrt(3)r)^2.
    # If j=i, dx = -r, dy = sqrt(3)r. dist^2 = r^2 + 3r^2 = 4r^2. dist=2r. OK.
    # If j=i-1, dx = r, dy = sqrt(3)r. dist^2 = 4r^2. OK.
    # Other neighbors are further.
    # So they are tangent.
    
    # Bounding box:
    # x_min = 0 - r = -r.
    # x_max = max(12r, 13r) + r = 14r.
    # Width = 14r.
    # y_min = 0 - r = -r.
    # y_max = 2*sqrt(3)r + r.
    # Height = 2*sqrt(3)r + 2r.
    # Perimeter = 2*(14r + 2*sqrt(3)r + 2r) = 2*(16r + 2*sqrt(3)r) = 4r*(8+sqrt(3)).
    # Wait, earlier I said 2r(8+sqrt(3)). Let's re-calculate.
    # w = 14r. h = 2r(sqrt(3)+1) = 2sqrt(3)r + 2r.
    # w+h = 16r + 2sqrt(3)r = 2r(8+sqrt(3)).
    # Perimeter = 2*(w+h) = 4r(8+sqrt(3)).
    # Perimeter = 4 => 4r(8+sqrt(3)) = 4 => r(8+sqrt(3)) = 1 => r = 1/(8+sqrt(3)).
    # Yes, correct.
    
    # Sum of radii = 21 * r = 21 / (8+sqrt(3)) ~ 2.1575.
    
    # Can we do even better?
    # What if we use rows of 7, 7, 7 but shift the middle row differently?
    # Or use a different number of rows?
    # 4 rows?
    # 5, 6, 6, 4?
    # Let's stick with the 3x7 hexagonal packing for now.
    # We can try to optimize the positions slightly to gain more space.
    
    # Let's implement the hexagonal packing.
    # Then we can try a local optimization to see if we can improve.
    
    r = 1.0 / (8.0 + np.sqrt(3.0))
    
    circles = []
    # Row 0
    for i in range(7):
        x = 2.0 * i * r
        y = 0.0
        circles.append([x, y, r])
    # Row 1
    for i in range(7):
        x = (2.0 * i + 1.0) * r
        y = np.sqrt(3.0) * r
        circles.append([x, y, r])
    # Row 2
    for i in range(7):
        x = 2.0 * i * r
        y = 2.0 * np.sqrt(3.0) * r
        circles.append([x, y, r])
    
    circles = np.array(circles)
    
    # Now, let's try to optimize the positions and radii.
    # We want to maximize sum(r_i) subject to:
    # 1. All circles inside rectangle w x h, w+h=2.
    # 2. No overlaps.
    # This is a non-convex problem.
    # We can use a simple iterative method:
    # - Start with the hexagonal packing.
    # - Try to increase radii slightly and move circles to avoid overlaps.
    # - Or use a force-based method.
    
    # Given the time limit, let's try a simple local search.
    # We'll use the fact that the perimeter constraint is w+h=2.
    # We can define the bounding box of the circles and scale them to fit w+h=2.
    
    # Let's try to perturb the positions and radii and see if we can improve.
    # We'll use a random restart local search.
    
    best_circles = circles.copy()
    best_score = np.sum(best_circles[:, 2])
    
    # Helper function to check validity and compute score
    def evaluate(c):
        # c is (n, 3)
        # Check overlaps
        n = len(c)
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = c[i, 2] + c[j, 2]
                if dist_sq < r_sum*r_sum - 1e-8:
                    return -1.0
        # Compute bounding box
        x_min = np.min(c[:, 0] - c[:, 2])
        x_max = np.max(c[:, 0] + c[:, 2])
        y_min = np.min(c[:, 1] - c[:, 2])
        y_max = np.max(c[:, 1] + c[:, 2])
        w = x_max - x_min
        h = y_max - y_min
        if w + h > 2.0 + 1e-8:
            return -1.0
        return np.sum(c[:, 2])
    
    # Local search
    # We'll try to move circles slightly and adjust radii.
    # This is complex. Let's try a simpler approach:
    # Keep the structure (3 rows of 7) and optimize the row spacing and horizontal shifts.
    
    # Parameters:
    # r (radius)
    # y1 (y of row 1), y2 (y of row 2)
    # shift1 (x shift of row 1)
    # We can also vary the x spacing.
    
    # Let's define a parameterized family:
    # Row 0: x_i = i * dx, y = 0
    # Row 1: x_i = i * dx + shift1, y = y1
    # Row 2: x_i = i * dx, y = y2
    # r is determined by the perimeter constraint?
    # No, we can scale the whole configuration.
    
    # Let's fix the relative positions and scale to fit w+h=2.
    # Then check for overlaps. If no overlaps, the sum of radii is proportional to the scale.
    
    # Let's try to optimize the relative positions (dx, y1, y2, shift1) to maximize the scale factor.
    
    # This is a 4D optimization.
    # We can use a grid search or random search.
    
    # Given time, let's just return the hexagonal packing.
    # It's a solid improvement over the grid.
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return best_circles
    
    # Let's try a few random perturbations of the hexagonal packing.
    np.random.seed(42)
    for _ in range(100):
        c = best_circles.copy()
        # Perturb positions
        c[:, :2] += np.random.normal(0, 0.01, (N, 2))
        # Perturb radii
        c[:, 2] += np.random.normal(0, 0.005, N)
        c[:, 2] = np.maximum(c[:, 2], 0.01)
        
        score = evaluate(c)
        if score > best_score:
            best_score = score
            best_circles = c.copy()
            
        if time.time() - start_time > DEADLINE:
            break
    
    return best_circles

# EVOLVE-BLOCK-END
