import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    """
    Construct 21 non-overlapping circles in a rectangle with perimeter 4.
    Returns an (N, 3) array where each row is (x, y, r).
    """
    # 1. Generate a good initial configuration using a hexagonal-like pattern.
    # We'll try a few different row distributions to find the best starting point.
    # Common distributions for 21 items in ~6 rows: 4-3-4-3-4-3 or similar.
    
    def make_hex_layout(row_counts):
        coords = []
        y = 0
        for i, count in enumerate(row_counts):
            x_offset = (max(row_counts) - count) * 0.5
            for j in range(count):
                coords.append((x_offset + j, y))
            # Vertical spacing for hex packing
            # If radius is 1, distance between centers is 2.
            # Vertical distance is 2 * sqrt(3)/2 = sqrt(3).
            # However, we just define relative coords now, we will scale later.
            # Let's use standard hex grid spacing: dx=1, dy=sqrt(3)/2 * 2?
            # Let's stick to integer-ish coordinates for simplicity of initial setup,
            # then let optimizer adjust.
            y += 1.0 # Placeholder, will be adjusted by optimizer or we use standard hex spacing
        return np.array(coords)

    # Let's use a proper hexagonal initial guess.
    # Row 0: 4 circles
    # Row 1: 3 circles
    # Row 2: 4 circles
    # Row 3: 3 circles
    # Row 4: 4 circles
    # Row 5: 3 circles
    
    # Standard hex packing coords (radius 1):
    # y_i = i * sqrt(3)
    # x_j = j + (0.5 if i%2==1 else 0)
    
    row_counts = [4, 3, 4, 3, 4, 3]
    init_coords = []
    for i, count in enumerate(row_counts):
        y = i * np.sqrt(3)
        x_start = (max(row_counts) - count) * 0.5
        for j in range(count):
            init_coords.append((x_start + j, y))
    
    init_coords = np.array(init_coords)
    
    # Center the coordinates
    init_coords -= np.median(init_coords, axis=0)
    
    # 2. Define the optimization problem.
    # We want to maximize sum of radii.
    # To make it easier, we can fix the rectangle dimensions (W, H) such that W+H=2.
    # Or we can optimize the positions and the rectangle aspect ratio.
    
    # Let's optimize the circle positions and the aspect ratio of the rectangle.
    # Variables: x_N, y_N, W (width). H = 2 - W.
    # But we also need to optimize the scale/radii?
    # Actually, if we fix the rectangle W x H, we can just find the max min-distance based packing.
    # But the radii can be different.
    #
    # A simpler formulation:
    # Optimize positions (x_i, y_i) and a scale factor s.
    # The rectangle is W x H with W+H=2.
    # Let's fix W and H first? No, that's suboptimal.
    #
    # Let's use the following approach:
    # 1. Find the optimal relative positions and aspect ratio.
    # 2. The "value" is sum of radii.
    #
    # Let's define the objective function to minimize the negative sum of radii,
    # subject to the circles fitting in the rectangle and not overlapping.
    #
    # Variables:
    # - positions: 21 * 2 = 42 variables
    # - radii: 21 variables
    # - aspect ratio: 1 variable (width W, height 2-W)
    #
    # Constraints:
    # - Circle i inside rectangle:
    #   x_i >= r_i
    #   x_i <= W - r_i
    #   y_i >= r_i
    #   y_i <= H - r_i
    # - Non-overlapping:
    #   dist(i, j) >= r_i + r_j
    #
    # This is a Non-Linear Programming (NLP) problem.
    # We can use scipy.optimize.minimize with method 'SLSQP' or 'trust-constr'.
    #
    # To handle the "maximize sum of radii" with variable rectangle size:
    # We can actually just optimize the positions and radii for a FIXED rectangle,
    # and then scale the whole thing? No, the aspect ratio matters.
    #
    # Let's try this:
    # Optimize the configuration in a unit square [0,1]x[0,1]? No, aspect ratio is free.
    #
    # Let's optimize the positions and radii directly, and include the rectangle dimensions
    # as variables in the optimization, but constrain the perimeter.
    
    # Actually, a more robust approach for "maximize sum of radii in fixed perimeter rectangle":
    # 1. Assume a specific aspect ratio (e.g., 1:1, or optimal for hex packing).
    # 2. Solve the packing problem for that rectangle.
    # 3. Try a few aspect ratios.
    
    # Given the time limit, let's try to solve the NLP directly.
    
    def objective_and_constraints(vars_):
        # vars_: [x_0, y_0, ..., x_20, y_20, r_0, ..., r_20, W]
        # H = 2 - W
        # Total vars: 21*2 + 21 + 1 = 64
        
        pos = vars_[:2*N].reshape(N, 2)
        radii = vars_[2*N:4*N]
        W = vars_[4*N]
        H = 2.0 - W
        
        # Ensure W and H are positive
        if W <= 0 or H <= 0:
            return 1e6, []
        
        # Objective: minimize -sum(radii)
        obj_val = -np.sum(radii)
        
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, r, i=i: x - r})
            # x <= W - r
            constraints.append({'type': 'ineq', 'fun': lambda x, W, r, i=i: W - x - r})
            # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda y, r, i=i: y - r})
            # y <= H - r
            constraints.append({'type': 'ineq', 'fun': lambda y, H, r, i=i: H - y - r})
            # r > 0 (implicitly handled by packing, but let's add a small floor)
            constraints.append({'type': 'ineq', 'fun': lambda r, i=i: r - 1e-6})
            
        # Non-overlapping constraints
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2
                sum_r = radii[i] + radii[j]
                # dist >= sum_r  =>  dist^2 - sum_r^2 >= 0
                constraints.append({'type': 'ineq', 'fun': lambda dist_sq, sum_r: dist_sq - sum_r**2})
        
        return obj_val, constraints

    # This direct constraint formulation is too slow/complex for SLSQP with 64 vars and ~400 constraints.
    # Let's use a penalty method or a simpler heuristic.
    
    # Alternative:
    # 1. Fix the aspect ratio to be optimal for hexagonal packing.
    #    For 6 rows of hex packing, the bounding box of the *centers* is roughly 5 wide and 5*sqrt(3) high.
    #    Plus radius 1 on all sides: W = 7, H = 5*sqrt(3) + 2.
    #    Perimeter = 2(W+H) = 2(7 + 8.66 + 2) = 2(17.66) = 35.32.
    #    Scale factor s = 4 / 35.32 = 0.113.
    #    Sum of radii = 21 * 1 * s = 2.37.
    
    # Let's just hardcode a very good hexagonal packing scaled to fit perimeter 4.
    # This avoids the complexity and risk of the optimizer failing or timing out.
    
    # Re-calculate the hexagonal layout precisely.
    # Rows: 4, 3, 4, 3, 4, 3
    # Let r = 1.
    # y-coordinates: 0, sqrt(3), 2sqrt(3), 3sqrt(3), 4sqrt(3), 5sqrt(3)
    # x-coordinates for row i:
    # i=0 (4): 0, 1, 2, 3
    # i=1 (3): 0.5, 1.5, 2.5
    # i=2 (4): 0, 1, 2, 3
    # i=3 (3): 0.5, 1.5, 2.5
    # i=4 (4): 0, 1, 2, 3
    # i=5 (3): 0.5, 1.5, 2.5
    
    # Bounding box of centers:
    # min x = 0, max x = 3
    # min y = 0, max y = 5*sqrt(3)
    
    # Bounding box including radius 1:
    # x_min = -1, x_max = 4  => W_centered = 5
    # y_min = -1, y_max = 5*sqrt(3) + 1 => H_centered = 5*sqrt(3) + 2
    
    # Wait, the width of the center cluster is 3 (from 0 to 3).
    # So the full width is 3 + 2*1 = 5.
    # The full height is 5*sqrt(3) + 2*1 = 5*sqrt(3) + 2.
    
    W_unscaled = 5.0
    H_unscaled = 5.0 * np.sqrt(3) + 2.0
    
    # Perimeter constraint: 2(W + H) = 4  => W + H = 2.
    # We scale the configuration by factor k.
    # W = k * W_unscaled
    # H = k * H_unscaled
    # k * (W_unscaled + H_unscaled) = 2
    # k = 2 / (W_unscaled + H_unscaled)
    
    k = 2.0 / (W_unscaled + H_unscaled)
    
    # Construct the centers
    centers = []
    row_counts = [4, 3, 4, 3, 4, 3]
    for i, count in enumerate(row_counts):
        y = i * np.sqrt(3)
        # Center the row horizontally relative to the max width (4 items => width 3)
        # The total width of 4 items is 3 units.
        # So offset = (3 - (count-1)) / 2
        offset = (3.0 - (count - 1)) / 2.0
        for j in range(count):
            x = offset + j
            centers.append((x, y))
            
    centers = np.array(centers)
    
    # The centers currently span x in [0, 3] and y in [0, 5*sqrt(3)].
    # We need to shift them so that the bounding box (including radius k) fits in [0, W] x [0, H].
    # Current min x of centers is 0. Min y is 0.
    # So the circle at (0,0) has left edge at -k and bottom edge at -k.
    # We need to shift all centers by (k, k) to make the left/bottom edges at 0.
    
    centers += k
    
    # Now min x of circle edges is 0.
    # Max x of circle edges: max center x (3) + k = 3 + k.
    # Wait, W = k * 5.
    # Is 3 + k == 5k?
    # 3 = 4k => k = 0.75.
    # Let's check k from formula:
    # W_unscaled = 5, H_unscaled = 5sqrt(3)+2 ~ 10.66. Sum ~ 15.66.
    # k = 2/15.66 ~ 0.127.
    # 3 + k = 3.127.
    # W = 5k = 0.635.
    # 3.127 > 0.635. Something is wrong.
    
    # Ah, the scaling applies to the coordinates too.
    # The "unscaled" coordinates were for radius 1.
    # So we scale the coordinates by k as well.
    
    centers_scaled = centers * k
    
    # Now re-check bounds.
    # Original centers spanned x in [0, 3], y in [0, 5sqrt(3)].
    # Scaled centers span x in [0, 3k], y in [0, 5sqrt(3)k].
    # We shifted by k initially, but that was for the unshifted logic.
    # Let's restart the positioning logic cleanly.
    
    # 1. Define centers in "unit radius" space.
    centers_unit = []
    row_counts = [4, 3, 4, 3, 4, 3]
    for i, count in enumerate(row_counts):
        y = i * np.sqrt(3)
        offset = (3.0 - (count - 1)) / 2.0
        for j in range(count):
            x = offset + j
            centers_unit.append((x, y))
    centers_unit = np.array(centers_unit)
    
    # 2. Find bounding box of centers_unit.
    min_x_u = np.min(centers_unit[:, 0])
    max_x_u = np.max(centers_unit[:, 0])
    min_y_u = np.min(centers_unit[:, 1])
    max_y_u = np.max(centers_unit[:, 1])
    
    # 3. The full bounding box including radius 1 is:
    # [min_x_u - 1, max_x_u + 1] x [min_y_u - 1, max_y_u + 1]
    # Width W_u = (max_x_u + 1) - (min_x_u - 1) = max_x_u - min_x_u + 2
    # Height H_u = (max_y_u + 1) - (min_y_u - 1) = max_y_u - min_y_u + 2
    
    W_u = max_x_u - min_x_u + 2
    H_u = max_y_u - min_y_u + 2
    
    # 4. Calculate scale factor k.
    # We want to fit this W_u x H_u box into a W x H box with W+H=2.
    # The optimal scaling is k = 2 / (W_u + H_u).
    k = 2.0 / (W_u + H_u)
    
    # 5. Scale the centers.
    # The "origin" of our unit box (min_x_u - 1, min_y_u - 1) should map to (0,0).
    # So, x_final = (x_unit - (min_x_u - 1)) * k
    # y_final = (y_unit - (min_y_u - 1)) * k
    
    centers_final = (centers_unit - np.array([min_x_u - 1, min_y_u - 1])) * k
    
    # 6. Radii are all k.
    radii = np.full(N, k)
    
    # 7. Construct result array.
    result = np.zeros((N, 3))
    result[:, 0] = centers_final[:, 0]
    result[:, 1] = centers_final[:, 1]
    result[:, 2] = radii
    
    return result

# Keep the EVOLVE-BLOCK structure as requested? The prompt says "Keep the function names and signatures".
# It does not explicitly say I must keep the EVOLVE-BLOCK comments, but usually it's safer to follow the structure.
# However, the prompt says "Rewrite the program from scratch... Keep the program's entry-point function name(s)".
# I will provide the full file content.

import numpy as np

def construct():
    N = 21
    row_counts = [4, 3, 4, 3, 4, 3]
    centers_unit = []
    for i, count in enumerate(row_counts):
        y = i * np.sqrt(3)
        offset = (3.0 - (count - 1)) / 2.0
        for j in range(count):
            x = offset + j
            centers_unit.append((x, y))
    centers_unit = np.array(centers_unit)
    
    min_x_u = np.min(centers_unit[:, 0])
    max_x_u = np.max(centers_unit[:, 0])
    min_y_u = np.min(centers_unit[:, 1])
    max_y_u = np.max(centers_unit[:, 1])
    
    W_u = max_x_u - min_x_u + 2
    H_u = max_y_u - min_y_u + 2
    
    k = 2.0 / (W_u + H_u)
    
    centers_final = (centers_unit - np.array([min_x_u - 1, min_y_u - 1])) * k
    radii = np.full(N, k)
    
    result = np.zeros((N, 3))
    result[:, 0] = centers_final[:, 0]
    result[:, 1] = centers_final[:, 1]
    result[:, 2] = radii
    
    return result
