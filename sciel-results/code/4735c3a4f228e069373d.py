import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Generate hex lattice points
    # Center (0,0)
    # Shell 1 (6 points)
    # Shell 2 (12 points: 6 corners dist 2, 6 mids dist sqrt(3))
    
    # Hex coords to Cartesian
    def hex_to_cart(q, r):
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        return x, y

    points = [(0,0)]
    # Shell 1
    for q, r in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
        points.append((q,r))
        
    # Shell 2
    # Corners (dist 2)
    corners = [(2,0), (2,-2), (0,-2), (-2,0), (-2,2), (0,2)]
    # Mids (dist sqrt(3))
    mids = [(2,-1), (1,-2), (0,-1) # wait, (0,-1) is shell 1? No.
            # Shell 2 mids are (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2), (1,2)??
            # Let's list all shell 2
            ]
            
    # Correct Shell 2 generation
    shell2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                shell2.append((q,r))
                
    # Identify corners and mids in shell2
    # Corners have dist 2 from origin
    # Mids have dist sqrt(3)
    
    hex_pts = []
    for q, r in shell2:
        x, y = hex_to_cart(q, r)
        d_sq = x*x + y*y
        if abs(d_sq - 4.0) < 1e-6:
            hex_pts.append((q, r, 'corner'))
        elif abs(d_sq - 3.0) < 1e-6:
            hex_pts.append((q, r, 'mid'))
            
    # We need 16 points.
    # Base: Center (1) + Shell 1 (6) = 7.
    # Need 9 more from Shell 2.
    # Strategy: Take all 6 Mids. Take 3 Corners.
    # To minimize diameter, remove corners that are "far" from each other?
    # Actually, we want to keep the set "compact".
    # Removing 3 corners from the 6 creates a set with 3 remaining corners.
    # To minimize max distance, we should remove corners such that the remaining ones are close?
    # No, we want to minimize the diameter of the whole set.
    # If we keep 3 corners, the max distance between them is determined by their angular spread.
    # If we keep 3 corners at 120 deg, max dist is 2*2*sin(60) = 2*sqrt(3) = 3.46.
    # If we keep 3 corners at 60 deg (adjacent), max dist is 2.
    # But we must include the mids.
    # The mids are at dist sqrt(3).
    # If we keep adjacent corners, say at 0, 60, 120.
    # The mids are at 30, 90, 150, 210, 270, 330.
    # The point at 270 (mid) is far from corner at 0?
    # Angle diff 90. Dist squared 7.
    # Point at 210 (mid) is far from corner at 0?
    # Angle diff 150. Dist squared 13.
    # So we want to avoid large angle differences between kept corners and kept mids.
    # Since we keep all 6 mids, we have mids at all 6 positions.
    # So any corner we keep will be at angle diff 150 from some mid.
    # Thus, any set of 3 corners will have a corner-mid pair with dist squared 13.
    # So $R^2 = 13$ is fixed for this topology (all mids + 3 corners).
    
    # Can we drop a mid and keep a corner?
    # If we drop 1 mid, we have 5 mids.
    # We can keep 4 corners?
    # Total 1 + 6 + 5 + 4 = 16.
    # If we drop the mid at 210 (which was causing the dist 13 with corner at 0),
    # and keep corner at 0, we might reduce max dist.
    # Let's try: Keep all 6 corners? No, 1+6+6+3=16.
    # If we keep all 6 corners, we must drop 3 mids.
    # Max dist corner-corner is 4. $R^2=16$.
    
    # What if we keep 4 corners and 5 mids?
    # Drop 1 mid and 1 corner?
    # 1 + 6 + 5 + 4 = 16.
    # If we drop corner at 0 and mid at 210?
    # Remaining corners: 60, 120, 180, 240, 300? (5 corners).
    # Wait, 1+6+5+4=16 means 4 corners, 5 mids.
    # Let's try to optimize the selection of 9 points from shell 2.
    
    # There are 12 shell 2 points. We choose 9.
    # We can iterate over combinations? 12 choose 9 = 220.
    # For each combination, calculate R^2.
    # This is fast.
    
    best_r_sq = float('inf')
    best_points = None
    
    # Get list of shell 2 points with Cartesian coords
    s2_pts = []
    for q, r, typ in hex_pts:
        x, y = hex_to_cart(q, r)
        s2_pts.append((x, y))
        
    # Base points: Center + Shell 1
    base_pts = []
    x, y = hex_to_cart(0,0)
    base_pts.append((x,y))
    for q, r in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
        x, y = hex_to_cart(q, r)
        base_pts.append((x,y))
        
    # Iterate over combinations of 9 from 12
    import itertools
    for combo in itertools.combinations(range(12), 9):
        pts = base_pts[:]
        for i in combo:
            pts.append(s2_pts[i])
            
        # Calculate R^2
        pts_arr = np.array(pts)
        # Distances
        # Use scipy distance? Or manual
        # N=16, so 120 pairs. Fast.
        d_sq = np.zeros((16,16))
        for i in range(16):
            for j in range(i+1, 16):
                d = np.linalg.norm(pts_arr[i] - pts_arr[j])
                d_sq[i,j] = d*d
                d_sq[j,i] = d*d
                
        min_d = np.min(d_sq[np.triu_indices(16, k=1)])
        max_d = np.max(d_sq[np.triu_indices(16, k=1)])
        
        if min_d < 1e-6: continue
        
        r_sq = max_d / min_d
        
        if r_sq < best_r_sq:
            best_r_sq = r_sq
            best_points = pts_arr.copy()
            
    if time.time() > deadline:
        # Return best found or fallback
        if best_points is not None:
            return best_points
        else:
            return np.array(base_pts + s2_pts[:9])
            
    # Optional: Local optimization on best_points
    # Skip for now to ensure safety, as combinatorial search is exhaustive for this topology.
    
    return best_points
# EVOLVE-BLOCK-END
