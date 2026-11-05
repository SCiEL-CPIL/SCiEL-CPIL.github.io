import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in axial coordinates
    # 7 central
    # 4 outer: (1,1) [30deg], (-1,2) [90deg], (-2,1) [150deg], (1,-2) [270deg]
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (1, -2)
    ]
    
    centers = []
    for q, r in centers_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    def get_required_L(alpha_deg):
        alpha = np.deg2rad(alpha_deg)
        vertex_angles = [alpha + k * np.pi / 3 for k in range(6)]
        
        max_proj = 0.0
        for cx, cy in centers:
            for theta in vertex_angles:
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                for i in range(6):
                    n_theta = np.deg2rad(30 + i * 60)
                    nx = np.cos(n_theta)
                    ny = np.sin(n_theta)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return max_proj * 2.0 / np.sqrt(3)

    best_L = float('inf')
    best_alpha = 0.0
    
    # Search 0 to 60
    for a in np.linspace(0, 60, 601):
        L = get_required_L(a)
        if L < best_L:
            best_L = L
            best_alpha = a
            
    # Refine
    for a in np.linspace(max(0, best_alpha - 5), min(60, best_alpha + 5), 1001):
        L = get_required_L(a)
        if L < best_L:
            best_L = L
            best_alpha = a
            
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, best_alpha])
        
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
