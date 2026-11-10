# EVOLVE-BLOCK-START
import numpy as np

def moving_sofa_3d_path() -> np.ndarray:
    """
    3D moving sofa: find a path through an L-shaped 3D corridor that maximizes
    the swept volume of a rigid block.
    
    The corridor is: {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    
    Returns a (k,6) array of poses (x,y,z,yaw,pitch,roll) in degrees.
    """
    # Create a smooth path through the corridor
    # The path goes:
    # 1. Along the left arm (x decreasing, y~0.5, z~0.5)
    # 2. Turns in XY plane (yaw 0->90)
    # 3. Along the middle (y increasing)
    # 4. Turns in YZ plane (pitch 0->90)  
    # 5. Along the right arm (z increasing)
    
    path = []
    
    # Phase 1: Move in -x direction
    n1 = 15
    for i in range(n1):
        t = i / (n1 - 1) if n1 > 1 else 0
        x = 0.4 - t * 0.8  # from 0.4 to -0.4
        y = 0.5
        z = 0.5
        path.append([x, y, z, 0.0, 0.0, 0.0])
    
    # Phase 2: Turn in XY plane (yaw 0->90)
    n2 = 30
    for i in range(n2):
        t = i / (n2 - 1) if n2 > 1 else 0
        # Use smoothstep for better rotation
        t_smooth = t * t * (3 - 2 * t)
        yaw = 90.0 * t_smooth
        x = -0.4 + t * 0.8  # from -0.4 to 0.4
        y = 0.5 + t * 1.0  # from 0.5 to 1.5
        z = 0.5
        path.append([x, y, z, yaw, 0.0, 0.0])
    
    # Phase 3: Move in +y direction
    n3 = 25
    for i in range(n3):
        t = i / (n3 - 1) if n3 > 1 else 0
        x = 0.4
        y = 1.5 + t * 3.0  # from 1.5 to 4.5
        z = 0.5
        path.append([x, y, z, 90.0, 0.0, 0.0])
    
    # Phase 4: Turn in YZ plane (pitch 0->90)
    n4 = 30
    for i in range(n4):
        t = i / (n4 - 1) if n4 > 1 else 0
        t_smooth = t * t * (3 - 2 * t)
        pitch = 90.0 * t_smooth
        x = 0.4
        y = 5.0
        z = 0.5 + t * 1.0  # from 0.5 to 1.5
        path.append([x, y, z, 90.0, pitch, 0.0])
    
    # Phase 5: Move in +z direction
    n5 = 15
    for i in range(n5):
        t = i / (n5 - 1) if n5 > 1 else 0
        x = 0.4
        y = 5.0
        z = 1.5 + t * 1.0  # from 1.5 to 2.5
        path.append([x, y, z, 90.0, 90.0, 0.0])
    
    return np.array(path, dtype=np.float64)

# EVOLVE-BLOCK-END
