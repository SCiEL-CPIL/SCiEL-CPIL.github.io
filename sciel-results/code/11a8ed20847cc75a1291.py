import numpy as np

def construct():
    N = 21
    best_score = -1
    best_params = None
    
    # Search for optimal grid dimensions C (cols) and R (rows)
    # We need C * R >= N
    for C in range(1, N + 1):
        for R in range(1, N + 1):
            if C * R < N:
                continue
            
            # For a uniform grid of C x R, to maximize the min dimension of cells
            # given W + H = 2, we set W/C = H/R = k
            # W = C*k, H = R*k
            # C*k + R*k = 2 => k = 2 / (C + R)
            # Cell width = k, Cell height = k
            # Max radius r = k / 2 = 1 / (C + R)
            
            score = N / (C + R)
            
            if score > best_score:
                best_score = score
                best_params = (C, R)
                
    C, R = best_params
    k = 2.0 / (C + R)
    W = C * k
    H = R * k
    # Reduce radius slightly to ensure disjoint interiors despite floating point precision
    r = (k / 2.0) * 0.999
    
    circles = []
    # We need to place N circles.
    # We have C * R cells.
    # Iterate through cells and add circles until we have N.
    count = 0
    for j in range(R):
        for i in range(C):
            if count >= N:
                break
            x = (i + 0.5) * (W / C)
            y = (j + 0.5) * (H / R)
            circles.append([x, y, r])
            count += 1
        if count >= N:
            break
            
    return np.array(circles)
