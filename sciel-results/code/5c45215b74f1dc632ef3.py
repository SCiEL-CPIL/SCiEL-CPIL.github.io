import numpy as np

N = 11

def construct():
    sqrt3 = np.sqrt(3)
    
    # Try multiple arrangements
    arrangements = []
    
    # Arrangement 1: 1+6+4
    pos1 = [(0, 0)]
    for i in range(6):
        a = np.radians(60 * i)
        pos1.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
    for i in range(4):
        a = np.radians(60 * i)
        pos1.append((2 * sqrt3 * np.cos(a), 2 * sqrt3 * np.sin(a)))
    arrangements.append(pos1)
    
    # Arrangement 2: 3+4+3+1 or similar
    # Let me try a more rectangular-ish packing
    # Row of 4, row of 4, row of 3
    pos2 = []
    for i in range(4):
        pos2.append((i * sqrt3, 0))
    for i in range(4):
        pos2.append((i * sqrt3 + sqrt3/2, 1.5))
    for i in range(3):
        pos2.append((i * sqrt3 + sqrt3, 3.0))
    arrangements.append(pos2)
    
    # Find best enclosing hexagon for each arrangement
    best_score = float('inf')
    best_result = None
    
    for positions in arrangements:
        apothem_factor = np.sqrt(3) / 2
        
        for rot_deg in np.arange(0, 60, 2):
            rot = np.radians(rot_deg)
            
            min_s = 0
            for k in range(3):
                na = rot + np.radians(30) + np.radians(60) * k
                nx, ny = np.cos(na), np.sin(na)
                dots = [p[0]*nx + p[1]*ny for p in positions]
                rng = max(dots) - min(dots)
                s_k = rng / (2 * apothem_factor)
                if s_k > min_s:
                    min_s = s_k
            
            if min_s < best_score:
                # Compute center
                cx, cy = 0, 0
                for k in range(3):
                    na = rot + np.radians(30) + np.radians(60) * k
                    nx, ny = np.cos(na), np.sin(na)
                    dots = [p[0]*nx + p[1]*ny for p in positions]
                    mid = (max(dots) + min(dots)) / 2
                    cx += mid * nx
                    cy += mid * ny
                cx /= 3
                cy /= 3
                
                cx0 = np.mean([p[0] for p in positions])
                cy0 = np.mean([p[1] for p in positions])
                dx, dy = cx - cx0, cy - cy0
                
                inner = [[p[0]+dx, p[1]+dy, 0.0] for p in positions]
                best_result = (inner, (cx, cy), min_s, rot_deg)
                best_score = min_s
    
    return best_result
