import numpy as np

def construct():
    # Construct a valid 4-coloring of K_49 using the affine plane AG(2, 7)
    # Vertices are (x, y) in F_7 x F_7
    # Edge color is determined by the slope of the line connecting them
    # Slopes are in P^1(F_7) = {0, 1, 2, 3, 4, 5, 6, infinity}
    # Coloring:
    # 0, infinity -> 0
    # 1, 6 -> 1
    # 2, 5 -> 2
    # 3, 4 -> 3
    
    q = 7
    N = q * q
    
    # Precompute slope to color mapping
    # slope 0 -> 0
    # slope inf -> 0
    # slope 1 -> 1
    # slope 6 -> 1
    # slope 2 -> 2
    # slope 5 -> 2
    # slope 3 -> 3
    # slope 4 -> 3
    
    slope_color = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 2,
        6: 1,
        'inf': 0
    }
    
    def get_color(x1, y1, x2, y2):
        dx = (x2 - x1) % q
        dy = (y2 - y1) % q
        
        if dx == 0:
            # Vertical line
            return slope_color['inf']
        else:
            # Slope m = dy / dx
            # In F_7, inverse of dx
            # 1^-1=1, 2^-1=4, 3^-1=5, 4^-1=2, 5^-1=3, 6^-1=6
            inv_dx = {1:1, 2:4, 3:5, 4:2, 5:3, 6:6}[dx]
            m = (dy * inv_dx) % q
            return slope_color[m]
            
    colors = []
    for i in range(N):
        x1 = i // q
        y1 = i % q
        for j in range(i + 1, N):
            x2 = j // q
            y2 = j % q
            c = get_color(x1, y1, x2, y2)
            colors.append(c)
            
    return {'n': N, 'colors': colors}
