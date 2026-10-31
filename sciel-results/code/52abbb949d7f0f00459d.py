"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    best_val = 0.0
    best_config = None

    # We search over aspect ratio alpha = W/H
    # Perimeter P = 2(W+H) = 4 => W+H = 2
    # Let H = h, W = 2-h.
    # Valid h range: (0, 2)
    
    # Try a range of aspect ratios
    # Square is h=1.0 (W=1, H=1)
    # Rectangular can be h < 1 or h > 1
    
    # We will use a hexagonal packing pattern.
    # For a given W, H, we place circles in rows.
    # Row 0: centers at y = r + offset, x = r + 2r*i
    # Row 1: centers at y = r + 2*r*sqrt(3)/2 + offset, x = r + 2r*(i+0.5)
    # ...
    
    # Since we want to maximize sum of radii, and all circles in the pattern have same radius r,
    # Sum = N * r.
    # We need to find the max r such that N circles fit in W x H.
    
    # We can iterate over possible counts of rows and columns to find the best r for a given W, H.
    # Or, simpler: iterate over W, H, and find max r via binary search or direct calculation.
    
    # Let's iterate over H from 0.1 to 1.9 (since W+H=2, W goes 1.9 to 0.1)
    # Step size 0.01 for H
    
    h_vals = np.arange(0.1, 1.9, 0.01)
    
    for h in h_vals:
        if time.time() > deadline:
            break
        w = 2.0 - h
        
        # Try to fit N circles in w x h
        # We assume a hexagonal grid.
        # Let r be the radius.
        # We need to find max r such that we can place N circles.
        
        # Approximation:
        # Area of rectangle = w * h
        # Area of N circles = N * pi * r^2
        # Density of hex packing ~ 0.9069
        # So N * pi * r^2 <= 0.9069 * w * h
        # r <= sqrt(0.9069 * w * h / (N * pi))
        
        # However, boundary effects matter.
        # Let's try a more direct approach:
        # Estimate number of rows and columns.
        
        # Try different numbers of rows m
        # For a given m, the height occupied is m * 2 * r * sqrt(3)/2 + r (approx)
        # Actually, for m rows, height = r + (m-1) * r * sqrt(3) + r = r * (2 + (m-1)*sqrt(3))
        # Wait, distance between row centers is r * sqrt(3).
        # Top of first row is y=r. Bottom of last row is y=H-r.
        # Distance between first and last center: (m-1) * r * sqrt(3).
        # Total height H = r + (m-1)*r*sqrt(3) + r = r * (2 + (m-1)*sqrt(3))
        # So r = H / (2 + (m-1)*sqrt(3))
        
        # For a given m, what is the max width occupied?
        # Rows alternate.
        # Even rows (0, 2, ...): centers at x = r + 2r*k
        # Odd rows (1, 3, ...): centers at x = r + 2r*(k+0.5)
        
        # We need to fit N circles.
        # Let's just iterate m from 1 to N.
        
        for m in range(1, N + 1):
            # Calculate r based on height
            denom_h = 2 + (m - 1) * np.sqrt(3)
            r = h / denom_h
            
            # Check if this r is reasonable (not too small)
            if r < 1e-6:
                continue
            
            # Now check if we can fit N circles in width w with this r and m rows.
            # We distribute N circles among m rows.
            # To minimize width, we should balance the number of circles in each row.
            # However, hex packing requires alternating shifts.
            # Let's just calculate the required width for a "tight" packing.
            
            # Let n_i be the number of circles in row i.
            # Sum n_i = N.
            # To minimize max width, we want n_i to be as equal as possible.
            # Let base = N // m, rem = N % m.
            # rem rows have base+1 circles, m-rem rows have base circles.
            
            base = N // m
            rem = N % m
            
            # Rows with more circles (base+1)
            # Width for a row with k circles:
            # If row is "even" (starts at r): x_max = r + 2*r*(k-1) + r = r * (2k) ? No.
            # Centers: r, r+2r, ..., r+2r(k-1).
            # Rightmost center: r + 2r(k-1).
            # Right edge: rightmost center + r = r + 2r(k-1) + r = 2rk.
            # So width needed is 2*r*k.
            
            # If row is "odd" (starts at r+r): centers at 2r, 4r, ...
            # Rightmost center: 2r(k-1) + 2r? No.
            # Centers: r+2r*0.5 = 2r? No, shift is r.
            # Standard hex:
            # Row 0 (even): x = r + 2r*j. (j=0..k-1). Max x = r + 2r(k-1). Right edge = r + 2r(k-1) + r = 2rk.
            # Row 1 (odd): x = r + 2r*(j+0.5) = 2r + 2r*j. (j=0..k-1). Max x = 2r + 2r(k-1) = 2rk. Right edge = 2rk + r.
            
            # So odd rows need an extra r of width?
            # Actually, we can shift the whole pattern horizontally to center it.
            # The bounding box width is determined by the leftmost and rightmost points.
            
            # Let's assume we can optimize the horizontal shift.
            # The effective width is roughly 2*r * max(k_i) + r.
            
            # Let's be conservative.
            # Max circles in a row = base + (1 if i < rem else 0)
            max_k = base + 1
            min_k = base
            
            # Required width approx
            # If all rows have max_k, width = 2*r*max_k + r (for odd rows)
            # Let's just check if 2*r*max_k + r <= w
            # This is a sufficient condition for a simple aligned grid, but hex is tighter.
            
            # A better check:
            # Can we fit N circles?
            # Let's try to construct the positions.
            
            positions = []
            # Distribute circles
            # We'll put circles in order.
            # This simple check might fail for some optimal configurations,
            # but it's a good heuristic for finding a valid r.
            
            # Let's calculate the exact width needed for the distribution.
            # We want to minimize the bounding box width.
            # This is a complex sub-problem.
            # Heuristic: assume rows are packed as tightly as possible.
            
            # Let's just check if the area fits and use a conservative r.
            # Or, simpler:
            # If 2*r*max_k + r <= w, then it fits.
            
            if 2 * r * max_k + r <= w:
                val = N * r
                if val > best_val:
                    best_val = val
                    # Store config to reconstruct later?
                    # We need to return the array of circles.
                    # We can reconstruct in the final step.
                    best_config = (w, h, r, m, base, rem)
            else:
                # Maybe it fits with a better arrangement?
                # For now, skip.
                pass

    # If no config found, fallback to square grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    w, h, r, m, base, rem = best_config
    
    # Reconstruct circles
    circles = []
    circles_left = N
    for i in range(m):
        # Determine number of circles in this row
        if i < rem:
            k = base + 1
        else:
            k = base
        
        # Adjust k if we run out of circles
        if k > circles_left:
            k = circles_left
            circles_left = 0
        else:
            circles_left -= k
        
        if k <= 0:
            continue
            
        # y position
        y = r + i * r * np.sqrt(3)
        
        # x positions
        # Center the row horizontally in width w
        # Row width = 2*r*(k-1) + 2*r = 2*r*k ?
        # Centers: x_0 ... x_{k-1}
        # Span = 2*r*(k-1)
        # Total width of circles = Span + 2r = 2*r*k
        # Start x = (w - 2*r*k) / 2 + r
        
        start_x = (w - 2 * r * k) / 2 + r
        
        for j in range(k):
            x = start_x + j * 2 * r
            circles.append([x, y, r])
            
    # Convert to array
    res = np.array(circles[:N])
    
    # Verify perimeter
    # Bounding box of circles
    min_x = res[:, 0].min() - res[:, 2].min()
    max_x = res[:, 0].max() + res[:, 2].max()
    min_y = res[:, 1].min() - res[:, 2].min()
    max_y = res[:, 1].max() + res[:, 2].max()
    
    w_bound = max_x - min_x
    h_bound = max_y - min_y
    perim = 2 * (w_bound + h_bound)
    
    if perim > 4.0 + 1e-9:
        # Scale down to fit
        scale = 4.0 / perim
        res[:, 0] *= scale
        res[:, 1] *= scale
        res[:, 2] *= scale
        
    return res
# EVOLVE-BLOCK-END
