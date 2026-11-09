import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_r = 0
    best_points = None
    best_R = 0
    
    # Search for optimal number of rows
    # R should be around sqrt(N)
    for R in range(10, 21):
        q, rem = divmod(N, R)
        n_max = q + (1 if rem > 0 else 0)
        
        # Theoretical radius for this R
        # Width term: 1 / (2 * n_max)
        # Height term: 1 / (sqrt(3) * (R - 1) + 2)
        r_width = 1.0 / (2.0 * n_max)
        r_height = 1.0 / (np.sqrt(3.0) * (R - 1) + 2.0)
        r = min(r_width, r_height)
        
        if r > best_r:
            best_r = r
            best_R = R

    if best_R == 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate points for best_R
    R = best_R
    q, rem = divmod(N, R)
    
    # Row lengths
    # rem rows have q+1, R-rem rows have q
    # To make it look nice, we can alternate or just put longer rows first/last?
    # It doesn't affect the bounding box width if we center each row.
    # But it might affect the "staggering" fit.
    # Let's just assign lengths.
    lengths = [q] * R
    for i in range(rem):
        lengths[i] = q + 1
        
    w = 2.0
    h = np.sqrt(3.0)
    
    points = []
    for i in range(R):
        n_i = lengths[i]
        # Centered x coords
        # Range: -(n_i-1)/2 * w ... (n_i-1)/2 * w
        # Step w
        xs = np.arange(n_i) * w - (n_i - 1) * w / 2.0
        y_i = i * h
        
        # Stagger
        if i % 2 == 1:
            xs = xs + w / 2.0
            
        for x in xs:
            points.append((x, y_i))
            
    points = np.array(points)
    
    # Center the points
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    points = points - [cx, cy]
    
    # Scale to fit in [-0.5, 0.5]
    # Max extent
    x_max = np.max(np.abs(points[:, 0]))
    y_max = np.max(np.abs(points[:, 1]))
    
    # The radius in the generated points is w/2 = 1.
    # We need to scale such that (x_max + 1) * k <= 0.5 ?
    # No.
    # The points are centers. The circles have radius 1 (in this scale).
    # The distance from center to wall must be >= 1.
    # So max |x| + 1 <= 0.5 / k ?
    # Let's say we scale points by k.
    # New centers: k * points.
    # New radius: k * 1 = k.
    # Constraint: max |k*x| + k <= 0.5
    # k * (max |x| + 1) <= 0.5
    # k <= 0.5 / (max |x| + 1)
    
    term_x = x_max + 1.0
    term_y = y_max + 1.0
    k = 0.5 / max(term_x, term_y)
    
    points = points * k
    
    # Verify
    if len(points) != N:
        # Should not happen
        pass
        
    return points
