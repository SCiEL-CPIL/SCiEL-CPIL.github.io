import numpy as np

N = 12

def construct():
    # Define a few candidate configurations
    # Each config is a list of (x, y, angle)
    
    configs = []
    
    # Config 1: 3x3 block + 3 attached
    # 3x3 block centers
    centers_3x3 = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = np.sqrt(3) * (i + j / 2.0)
            y = 1.5 * j
            centers_3x3.append((x, y, 0.0))
            
    # Add 3 more to the "sides" of the 3x3 block
    # The 3x3 block has flat sides of length 3 hexagons.
    # Attaching to the middle of a side might be better?
    # Let's attach to the 3 hexagons on the "right" side (i=1)
    # Their centers are at (sqrt(3)*1.5, -1.5), (sqrt(3)*1, 0), (sqrt(3)*0.5, 1.5)
    # Wait, the neighbors of i=1, j=-1 is i=2, j=-1?
    # Lattice: (i, j) -> (i+1, j) is right.
    # Let's add (2, -1), (2, 0), (2, 1)
    extra = []
    for j in range(-1, 2):
        i = 2
        x = np.sqrt(3) * (i + j / 2.0)
        y = 1.5 * j
        extra.append((x, y, 0.0))
        
    config_1 = centers_3x3 + extra
    configs.append(config_1)
    
    # Config 2: Star (1+6+5)
    center = [(0.0, 0.0, 0.0)]
    ring1 = []
    ring2 = []
    for k in range(6):
        angle = k * np.pi / 3
        x1 = np.sqrt(3) * np.cos(angle)
        y1 = np.sqrt(3) * np.sin(angle)
        ring1.append((x1, y1, 0.0))
        
        # 5 in ring 2, skip one arm (e.g., k=5)
        if k < 5:
            x2 = 2 * np.sqrt(3) * np.cos(angle)
            y2 = 2 * np.sqrt(3) * np.sin(angle)
            ring2.append((x2, y2, 0.0))
            
    config_2 = center + ring1 + ring2
    configs.append(config_2)
    
    # Config 3: 2x2x2 block + 4?
    # Let's try a 4x3 block
    centers_4x3 = []
    for i in range(-1, 3): # 4 wide
        for j in range(-1, 2): # 3 high
            x = np.sqrt(3) * (i + j / 2.0)
            y = 1.5 * j
            centers_4x3.append((x, y, 0.0))
    # This is 12 hexagons
    configs.append(centers_4x3)
    
    # Calculate S for each config
    best_config = None
    best_S = float('inf')
    
    for config in configs:
        S = calc_min_side(config)
        if S < best_S:
            best_S = S
            best_config = config
            
    # Return the best config
    # Ensure center of outer hexagon is (0,0) and rotation 0
    return best_config, (0.0, 0.0), best_S, 0.0

def calc_min_side(config):
    # config: list of (x, y, angle_deg)
    # We want to find the minimal side length S of a regular hexagon
    # centered at (0,0) that contains all these hexagons.
    # The outer hexagon can be rotated. We assume optimal alignment is 0, 30, etc.
    # But for a regular hexagon, the support function is periodic.
    # We check 6 directions (normals).
    
    max_proj = 0.0
    # Check 6 normal directions
    for k in range(6):
        angle = k * np.pi / 3
        u = np.array([np.cos(angle), np.sin(angle)])
        
        max_support = 0.0
        for (x, y, a_deg) in config:
            a_rad = np.deg2rad(a_deg)
            # Support of a unit hexagon at (x,y) with rotation a_rad in direction u
            # h = (x,y) dot u + h_unit(u - a_rad)
            
            # h_unit for a unit hexagon (circumradius 1)
            # The distance from center to side is sqrt(3)/2.
            # The normal to the side closest to direction theta is at theta rounded to nearest 30 deg?
            # Actually, h_unit(theta) = (sqrt(3)/2) / cos(theta - theta_normal)
            # where theta_normal is the normal of the closest side.
            # For a hexagon with vertices at 0, 60, ..., the sides are normal to 30, 90, ...
            # So the normals are at 30 + 60*k.
            
            theta = angle - a_rad
            # Find the closest normal angle
            # Normals are at pi/6 + k*pi/3
            # theta_mod = theta % (pi/3)
            # If theta_mod is close to pi/6, use that.
            # Actually, the support function of a regular hexagon with inradius r is:
            # h(theta) = r / cos((theta - theta_0) mod (pi/3) - pi/6)
            # Let's simplify:
            # The max projection of a unit hexagon (inradius r=sqrt(3)/2) in any direction is r / cos(delta)
            # where delta is the angle between the direction and the closest normal.
            # The max delta is pi/6 (30 deg).
            
            # Calculate delta
            # The normals are at pi/6, pi/2, ...
            # Let's find the angle to the closest normal.
            # theta is the direction in local frame.
            # The closest normal is at round(theta / (pi/3)) * pi/3 + pi/6 ?
            # No, the normals are at pi/6 + k*pi/3.
            
            # Let's just use the formula:
            # h = r / cos( (theta - pi/6) % (pi/3) - pi/6 )
            # This assumes the hexagon is aligned such that a normal is at pi/6.
            # Our unit hexagon has vertices at 0, 60... so normals at 30, 90...
            # So yes, normal at pi/6.
            
            delta = (theta - np.pi/6) % (np.pi/3) - np.pi/6
            # delta is in [-pi/6, pi/6]
            h_unit = (np.sqrt(3)/2.0) / np.cos(delta)
            
            proj = x * u[0] + y * u[1] + h_unit
            if proj > max_support:
                max_support = proj
                
        # The outer hexagon has side S.
        # Its inradius is S * sqrt(3) / 2.
        # The support in direction angle is S * sqrt(3) / 2.
        # So S * sqrt(3) / 2 >= max_support
        # S >= max_support * 2 / sqrt(3)
        
        S_req = max_support * 2.0 / np.sqrt(3)
        if S_req > max_proj:
            max_proj = S_req
            
    return max_proj
