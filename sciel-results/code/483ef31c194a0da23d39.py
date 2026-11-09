"""Program using hexagonal lattice + optimization for minimal outer hexagon side length."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Generate candidate configurations
    configs = []
    
    # Hex lattice basis
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # Config 1: 4x3 rectangular block on hex lattice
    pos1 = []
    for j in range(3):
        for i in range(4):
            pos1.append(i*v1 + j*v2)
    configs.append(pos1)
    
    # Config 2: 3x4 rectangular block
    pos2 = []
    for j in range(4):
        for i in range(3):
            pos2.append(i*v1 + j*v2)
    configs.append(pos2)
    
    # Config 3: Hexagonal cluster (center + ring + partial ring)
    pos3 = [np.array([0.0, 0.0])]
    for k in range(6):
        ang = k * np.pi / 3
        pos3.append(sqrt3 * np.array([np.cos(ang), np.sin(ang)]))
    # 5 more at radius 2*sqrt(3)
    for k in range(5):
        ang = (k + 0.5) * np.pi / 3
        pos3.append(2*sqrt3 * np.array([np.cos(ang), np.sin(ang)]))
    configs.append(pos3)
    
    # Config 4: 2x6 rectangle
    pos4 = []
    for j in range(2):
        for i in range(6):
            pos4.append(i*v1 + j*v2)
    configs.append(pos4)
    
    # Config 5: 6x2 rectangle
    pos5 = []
    for j in range(6):
        for i in range(2):
            pos5.append(i*v1 + j*v2)
    configs.append(pos5)
    
    # Config 6: Staggered rows (like brick pattern)
    pos6 = []
    # Row 0: 4 hexagons
    for i in range(4):
        pos6.append(i*v1)
    # Row 1: 4 hexagons offset by half
    for i in range(4):
        pos6.append((i + 0.5)*v1 + np.array([0, 1.5]))
    # Row 2: 4 hexagons
    for i in range(4):
        pos6.append(i*v1 + np.array([0, 3.0]))
    configs.append(pos6)
    
    # Config 7: Compact diamond
    pos7 = []
    # Center row: 4
    for i in range(-1, 3):
        pos7.append(i*v1)
    # Top row: 4 (staggered)
    for i in range(-1, 3):
        pos7.append((i+0.5)*v1 + np.array([0, 1.5]))
    # Bottom row: 4 (staggered)
    for i in range(-1, 3):
        pos7.append((i+0.5)*v1 - np.array([0, 1.5]))
    configs.append(pos7)
    
    best_L = float('inf')
    best_pos = None
    best_inner_rot = 0.0
    best_theta = 0.0
    
    inner_rotations = [0.0, 30.0]
    outer_rotations = list(range(0, 60, 5))  # 0 to 55 degrees
    
    for pos_list in configs:
        if len(pos_list) != N:
            continue
        
        # Center at origin
        com = np.mean(pos_list, axis=0)
        pos_centered = [p - com for p in pos_list]
        
        for inner_rot in inner_rotations:
            # Compute all vertices
            all_verts = []
            for p in pos_centered:
                for m in range(6):
                    ang = np.radians(inner_rot) + m * np.pi / 3
                    all_verts.append(p + np.array([np.cos(ang), np.sin(ang)]))
            all_verts = np.array(all_verts)
            
            for theta_deg in outer_rotations:
                theta = np.radians(theta_deg)
                normals = []
                for k in range(6):
                    a = theta + k * np.pi / 3
                    normals.append(np.array([np.cos(a), np.sin(a)]))
                
                # Find max |dot(v, n)| over all vertices and normals
                t_min = 0.0
                for v in all_verts:
                    for n in normals:
                        proj = abs(np.dot(v, n))
                        if proj > t_min:
                            t_min = proj
                
                L = t_min * 2.0 / sqrt3
                if L < best_L:
                    best_L = L
                    best_pos = pos_centered
                    best_inner_rot = inner_rot
                    best_theta = theta_deg
    
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), float(best_inner_rot)])
    
    return inner, (0.0, 0.0), float(best_L), float(best_theta)
# EVOLVE-BLOCK-END
