import math
import numpy as np

N = 11

def construct():
    # Define two candidate configurations
    # Config 1: Flat-top inner hexagons (angle 30), spacing sqrt(3)
    # Config 2: Pointy-right inner hexagons (angle 0), spacing 2.0
    
    configs = []
    
    # Config 1
    s = math.sqrt(3)
    centers1 = [
        (0, 0),
        (s, 0),
        (s/2, 1.5),
        (-s/2, 1.5),
        (-s, 0),
        (-s/2, -1.5),
        (s/2, -1.5),
        (3*s/2, 1.5),
        (-3*s/2, 1.5),
        (-3*s/2, -1.5),
        (3*s/2, -1.5)
    ]
    angle1 = 30.0
    configs.append((centers1, angle1))
    
    # Config 2
    centers2 = [
        (0, 0),
        (2, 0),
        (1, 3/math.sqrt(3)), # Wait, spacing 2 for pointy right?
        # If pointy right, neighbors are at distance 2.
        # Directions: 0, 60, 120...
        # (2, 0)
        # (1, sqrt(3))
        # (-1, sqrt(3))
        # (-2, 0)
        # (-1, -sqrt(3))
        # (1, -sqrt(3))
        # Add 4 more:
        # (3, sqrt(3))
        # (-3, sqrt(3))
        # (-3, -sqrt(3))
        # (3, -sqrt(3))
    ]
    # Let's just use the same lattice structure scaled by 2/sqrt(3)
    scale = 2.0 / math.sqrt(3)
    centers2 = [(x*scale, y*scale) for x, y in centers1]
    angle2 = 0.0
    configs.append((centers2, angle2))
    
    best_S = float('inf')
    best_inner = None
    best_angle = 0.0
    
    for centers, inner_angle_deg in configs:
        # Generate vertices
        vertices = []
        for cx, cy in centers:
            for k in range(6):
                theta = math.radians(inner_angle_deg + k * 60)
                vx = cx + math.cos(theta)
                vy = cy + math.sin(theta)
                vertices.append((vx, vy))
        
        vertices = np.array(vertices)
        
        # Minimize S(alpha)
        # S(alpha) = (1/sqrt(3)) * max( E(alpha + k*pi/3) )
        # E(phi) = max(dot(v, u_phi)) - min(dot(v, u_phi))
        
        def get_extent(phi):
            u = np.array([math.cos(phi), math.sin(phi)])
            dots = vertices @ u
            return np.max(dots) - np.min(dots)
        
        def calc_S(alpha):
            max_e = 0
            for k in range(6):
                phi = alpha + k * math.pi / 3
                e = get_extent(phi)
                if e > max_e:
                    max_e = e
            return max_e / math.sqrt(3)
        
        # Search alpha in [0, 60 deg)
        # Fine grid
        alphas = np.linspace(0, math.pi/3, 1000)
        S_vals = np.array([calc_S(a) for a in alphas])
        idx = np.argmin(S_vals)
        S_min = S_vals[idx]
        alpha_min = alphas[idx]
        
        if S_min < best_S:
            best_S = S_min
            best_inner = [[cx, cy, inner_angle_deg] for cx, cy in centers]
            best_angle = math.degrees(alpha_min)
            
    return best_inner, (0.0, 0.0), best_S, best_angle

