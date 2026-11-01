"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4 (i.e., 2*(w+h)=4 => w+h=2).
    Maximize sum of radii.
    
    Strategy:
    1. Use a grid-like initial layout to provide a good starting point.
    2. Use gradient-based optimization (L-BFGS-B) to maximize the sum of radii
       subject to non-overlap constraints and boundary constraints.
       To handle the non-smooth max radius per circle, we can instead fix the 
       radii to be equal (or optimize them jointly) and maximize the common radius.
       However, the problem asks to maximize the SUM of radii. 
       
       Actually, for identical circles, maximizing the sum is equivalent to maximizing 
       the common radius. But the problem allows different radii.
       
       Let's try a simpler approach first: assume all circles have the same radius r.
       Then we just need to pack 21 equal circles in a rectangle of perimeter 4.
       This is a well-studied problem. The optimal packing for 21 equal circles in a 
       rectangle is likely a hexagonal-like packing.
       
       However, allowing different radii might yield a better sum. But for simplicity 
       and robustness, let's start with equal radii.
       
       Rectangle: w + h = 2.
       Let's parameterize by aspect ratio. Let w = x, h = 2 - x.
       We want to maximize r such that 21 circles of radius r fit in w x h.
       
       We can use a grid layout. For N=21, a 5x5 grid would have 25 cells, so we can 
       use 21 of them. But 21 is not a perfect square. 21 = 3*7.
       A 3x7 grid might work well.
       
       Let's try a few grid configurations and pick the best one.
       
       Configuration 1: 3 rows, 7 columns.
       Configuration 2: 4 rows, 6 columns (with one missing).
       Configuration 3: 5 rows, 5 columns (with 4 missing).
       
       For a grid of n_rows x n_cols, the minimum width and height to fit circles of radius r are:
       w >= 2*r * n_cols
       h >= 2*r * n_rows
       So r <= min(w/(2*n_cols), h/(2*n_rows))
       
       With w + h = 2, we have:
       r <= min((2-h)/(2*n_cols), h/(2*n_rows))
       Let h = 2 - w.
       r <= min(w/(2*n_cols), (2-w)/(2*n_rows))
       
       To maximize r, we set w/(2*n_cols) = (2-w)/(2*n_rows)
       w/n_cols = (2-w)/n_rows
       w * n_rows = (2-w) * n_cols
       w * n_rows = 2*n_cols - w*n_cols
       w * (n_rows + n_cols) = 2 * n_cols
       w = 2 * n_cols / (n_rows + n_cols)
       h = 2 - w = 2 * n_rows / (n_rows + n_cols)
       
       Then r = w / (2 * n_cols) = (2 * n_cols / (n_rows + n_cols)) / (2 * n_cols) = 1 / (n_rows + n_cols)
       
       So for a grid of n_rows x n_cols, the maximum equal radius is r = 1 / (n_rows + n_cols).
       
       Let's check for 3x7: r = 1/(3+7) = 0.1. Sum = 21 * 0.1 = 2.1.
       For 4x6 (approx): If we use 4 rows and 6 cols, but only 21 circles, we can still use the 
       bounding box of 4x6. r = 1/(4+6) = 0.1. Sum = 2.1.
       For 5x5: r = 1/(5+5) = 0.1. Sum = 2.1.
       
       Wait, this suggests that any grid with n_rows + n_cols = 10 gives r=0.1.
       What about 2x11? r = 1/13 ≈ 0.0769. Sum ≈ 1.615. Worse.
       What about 7x3? Same as 3x7.
       
       Is there a better packing? Hexagonal packing might allow a larger radius.
       
       Let's try a hexagonal packing.
       For hexagonal packing in a rectangle, the spacing between centers is 2r in the horizontal 
       direction for adjacent rows, and sqrt(3)*r vertically.
       
       This is more complex. Let's stick with the grid for now and see if we can improve it 
       with a local optimization.
       
       Actually, the problem allows different radii. Maybe we can have some circles larger 
       and some smaller to increase the sum.
       
       Let's try a different approach: use a simple greedy algorithm or a local search 
       to place circles one by one, maximizing the radius of each new circle.
       
       But the deadline is 120s. Let's keep it simple and robust.
       
       I'll implement the grid method for a few configurations and return the best one.
       """
    deadline = time.time() + 110  # Safety margin
    
    # Try several grid configurations
    best_sum = 0
    best_result = None
    
    for n_rows in range(1, 10):
        for n_cols in range(1, 10):
            if n_rows * n_cols < N:
                continue
            
            # Calculate optimal w, h, r for this grid
            w = 2 * n_cols / (n_rows + n_cols)
            h = 2 * n_rows / (n_rows + n_cols)
            r = 1 / (n_rows + n_cols)
            
            # Create the grid positions
            # We need to place N circles in the n_rows x n_cols grid
            # We'll fill the grid row by row, and if we have extra cells, we leave them empty
            positions = []
            for i in range(n_rows):
                for j in range(n_cols):
                    if len(positions) >= N:
                        break
                    x = (j + 0.5) * (w / n_cols)
                    y = (i + 0.5) * (h / n_rows)
                    positions.append((x, y, r))
                
                if len(positions) >= N:
                    break
            
            if len(positions) == N:
                # Check if all circles are within the rectangle and non-overlapping
                # (They should be, by construction, but let's verify)
                valid = True
                for i, (x1, y1, r1) in enumerate(positions):
                    if x1 < r1 or x1 + r1 > w or y1 < r1 or y1 + r1 > h:
                        valid = False
                        break
                    for j, (x2, y2, r2) in enumerate(positions[i+1:], i+1):
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < r1 + r2 - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    total_r = sum(p[2] for p in positions)
                    if total_r > best_sum:
                        best_sum = total_r
                        best_result = np.array(positions)
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple 5x5 grid with 21 circles
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
