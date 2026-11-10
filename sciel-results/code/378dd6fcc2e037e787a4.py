# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_3d_path() -> np.ndarray:
    """
    Compute an optimized path for the 3D moving sofa problem.
    
    The corridor is:
    - {x<=0, 0<=y<=1, 0<=z<=1} (left arm)
    - {0<=x<=1, 0<=y<=5, 0<=z<=1} (bottom arm)  
    - {0<=x<=1, 5<=y<=6, z>=0} (vertical extension)
    
    The sofa must go from left (x<0, y~0) through the bottom corner to the
    vertical extension (y~6, z>0). This requires:
    1. Entering from the left
    2. Turning 90° in the X-Y plane (corner at origin)
    3. Traveling along y
    4. Turning 90° in the Y-Z plane (at y=5-6)
    5. Exiting upward
    
    We construct the path with careful attention to the turning radii.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Key parameters for the path
    # The sofa block is 4x4x4 grid points with 0.05 spacing = 0.15 units per side
    
    # Phase 1: Enter from left, approach the corner
    # Start: x<0, y in [0,1], z in [0,1], yaw=0 (facing +x)
    # Need to navigate around the corner at (0,0,0)
    
    # Phase 2: Turn 90° in XY plane - this is the hard part
    # The sofa needs to rotate from facing +x to facing +y
    
    # Phase 3: Travel along y from 0 to ~5
    
    # Phase 4: Turn 90° in YZ plane
    # Rotate from facing +y to facing +z
    
    # Phase 5: Exit upward
    
    # Build path segments
    path_points = []
    
    # Helper to add interpolated points
    def add_segment(p1, p2, n_steps, max_dist=0.05):
        """Add interpolated points between two poses"""
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        
        # Compute required number of steps based on distance
        dist = np.linalg.norm(p2[:3] - p1[:3])
        angle_diff = np.sqrt((p2[3]-p1[3])**2 + (p2[4]-p1[4])**2 + (p2[5]-p1[5])**2)
        
        n_needed = max(1, int(np.ceil(max(dist, angle_diff*0.01) / max_dist)))
        n = max(n_steps, n_needed)
        
        for i in range(n + 1):
            t = i / n
            p = p1 * (1-t) + p2 * t
            path_points.append(p)
    
    # Define key waypoints for the path
    # The critical constraint is that during rotation, the bounding box of the
    # rotation arc must stay in the corridor.
    
    # For the XY turn: rotating about z-axis
    # The sofa center should be positioned so that as it rotates from yaw=0 to yaw=90,
    # all points stay in the corridor.
    
    # Optimal position for XY turn: center at approximately (0.5, 0.5, 0.5)
    # but we need to be more careful.
    
    # Let me use a parametric approach based on the known good solution structure.
    # The seed solution has ~99 points. Let me create a similar structure but
    # with optimized turning points.
    
    # Phase 1: Approach from left (x < 0)
    # y ~ 0.5, z ~ 0.5, yaw ~ 0
    x1, y1, z1 = -0.5, 0.5, 0.5
    yaw1, pitch1, roll1 = 0, 0, 0
    
    # Move to just before the corner
    x2, y2, z2 = -0.1, 0.5, 0.5
    yaw2, pitch2, roll2 = 0, 0, 0
    
    add_segment([x1,y1,z1,yaw1,pitch1,roll1], [x2,y2,z2,yaw2,pitch2,roll2], 10)
    
    # Phase 2: The XY turn - rotate from facing +x to facing +y
    # This is the critical part. The sofa needs to pivot around the corner.
    # Optimal: move center to (0.5, 0.5, 0.5) while rotating
    
    # Step 2a: Move to turning position
    x3, y3, z3 = 0.3, 0.3, 0.5
    yaw3, pitch3, roll3 = 15, 0, 0
    
    add_segment([x2,y2,z2,yaw2,pitch2,roll2], [x3,y3,z3,yaw3,pitch3,roll3], 15)
    
    # Step 2b: Continue rotation
    x4, y4, z4 = 0.5, 0.5, 0.5
    yaw4, pitch4, roll4 = 45, 0, 0
    
    add_segment([x3,y3,z3,yaw3,pitch3,roll3], [x4,y4,z4,yaw4,pitch4,roll4], 15)
    
    # Step 2c: Complete the turn
    x5, y5, z5 = 0.5, 0.7, 0.5
    yaw5, pitch5, roll5 = 75, 0, 0
    
    add_segment([x4,y4,z4,yaw4,pitch4,roll4], [x5,y5,z5,yaw5,pitch5,roll5], 15)
    
    # Step 2d: Finish XY turn
    x6, y6, z6 = 0.5, 1.0, 0.5
    yaw6, pitch6, roll6 = 90, 0, 0
    
    add_segment([x5,y5,z5,yaw5,pitch5,roll5], [x6,y6,z6,yaw6,pitch6,roll6], 10)
    
    # Phase 3: Travel along y from 1 to 5
    x7, y7, z7 = 0.5, 5.0, 0.5
    yaw7, pitch7, roll7 = 90, 0, 0
    
    add_segment([x6,y6,z6,yaw6,pitch6,roll6], [x7,y7,z7,yaw7,pitch7,roll7], 30)
    
    # Phase 4: The YZ turn - rotate from facing +y to facing +z
    # This happens at y ~ 5-6
    # The sofa needs to pivot up
    
    # Step 4a: Begin the YZ turn
    x8, y8, z8 = 0.5, 5.5, 0.5
    yaw8, pitch8, roll8 = 90, 15, 0
    
    add_segment([x7,y7,z7,yaw7,pitch7,roll7], [x8,y8,z8,yaw8,pitch8,roll8], 15)
    
    # Step 4b: Continue rotation
    x9, y9, z9 = 0.5, 5.8, 0.7
    yaw9, pitch9, roll9 = 90, 45, 0
    
    add_segment([x8,y8,z8,yaw8,pitch8,roll8], [x9,y9,z9,yaw9,pitch9,roll9], 15)
    
    # Step 4c: More rotation
    x10, y10, z10 = 0.5, 6.0, 1.0
    yaw10, pitch10, roll10 = 90, 75, 0
    
    add_segment([x9,y9,z9,yaw9,pitch9,roll9], [x10,y10,z10,yaw10,pitch10,roll10], 15)
    
    # Step 4d: Complete YZ turn
    x11, y11, z11 = 0.5, 6.0, 1.5
    yaw11, pitch11, roll11 = 90, 90, 0
    
    add_segment([x10,y10,z10,yaw10,pitch10,roll10], [x11,y11,z11,yaw11,pitch11,roll11], 10)
    
    # Phase 5: Exit upward
    x12, y12, z12 = 0.5, 6.0, 3.0
    yaw12, pitch12, roll12 = 90, 90, 0
    
    add_segment([x11,y11,z11,yaw11,pitch11,roll11], [x12,y12,z12,yaw12,pitch12,roll12], 20)
    
    # Convert to array
    path = np.array(path_points, dtype=np.float64)
    
    # Remove duplicate consecutive points
    if len(path) > 1:
        mask = np.ones(len(path), dtype=bool)
        for i in range(1, len(path)):
            if np.allclose(path[i], path[i-1], atol=1e-10):
                mask[i] = False
        path = path[mask]
    
    # Check deadline
    if time.time() > deadline - 30:
        return path
    
    # Ensure minimum length
    if len(path) < 1:
        return np.array([[0.5, 0.5, 0.5, 0, 0, 0]], dtype=np.float64)
    
    return path

# EVOLVE-BLOCK-END
