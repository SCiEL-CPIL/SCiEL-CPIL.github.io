"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import itertools
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # 1. Generate candidate point sets
    # We want compact sets of 21 points.
    # Hexagonal packing is densest.
    # A "hexagonal number" cluster is 1, 3, 5, 7, 5, 3, 1 (25 pts).
    # Removing the 4 corner points (distance 2 from center in hex coords) leaves 21 pts.
    # This shape is roughly a hexagon with corners cut off.
    
    points_hex = []
    # Hex coordinates (q, r)
    # Distance from center in hex metric: max(|q|, |r|, |q+r|) ? No, standard axial.
    # Let's use axial coordinates (q, r) where distance is (|q|+|r|+|q+r|)/2
    # Or just generate a grid and filter by Euclidean distance from center.
    
    # Generate a large grid of hex points
    # Spacing 1
    for q in range(-5, 6):
        for r in range(-5, 6):
            # Convert axial to pixel
            x = 1.0 * (q + 0.5 * r)
            y = 1.0 * (math.sqrt(3) / 2.0 * r)
            points_hex.append((x, y))
            
    # Sort by distance from center
    points_hex.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 21
    pts = np.array(points_hex[:21])
    
    # Center the points at (0,0)
    center = np.mean(pts, axis=0)
    pts = pts - center
    
    # Compute bounding box of the point set
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    w_pts = max_x - min_x
    h_pts = max_y - min_y
    
    # The "aspect ratio" of the point cloud is w_pts / h_pts
    # To maximize the minimum distance between points and walls relative to the radius,
    # we should match the rectangle's aspect ratio to the point cloud's aspect ratio.
    # Rectangle perimeter is 4, so w + h = 2.
    # Let w = 2 * ratio / (1 + ratio) and h = 2 / (1 + ratio) where ratio = w/h.
    # Ideally ratio = w_pts / h_pts.
    
    # Let's try a range of aspect ratios around the ideal one, plus the square case.
    ideal_ratio = w_pts / h_pts if h_pts > 0 else 1.0
    
    # Candidate aspect ratios
    # We want to test ratios that allow the cloud to fit well.
    # If we force the cloud into a square, we might have wasted space in one dimension.
    # Matching the aspect ratio maximizes the minimum clearance.
    
    aspect_ratios = [
        ideal_ratio,
        1.0,
        ideal_ratio * 0.9,
        ideal_ratio * 1.1,
        ideal_ratio * 0.8,
        ideal_ratio * 1.2
    ]
    
    # Filter valid aspect ratios (w, h > 0)
    valid_aspects = []
    for ratio in aspect_ratios:
        if ratio <= 0: continue
        w = 2.0 * ratio / (1.0 + ratio)
        h = 2.0 / (1.0 + ratio)
        if w > 0 and h > 0:
            valid_aspects.append((w, h))
    
    # Deduplicate
    valid_aspects = list(set(valid_aspects))
    
    for w, h in valid_aspects:
        if time.time() > deadline:
            break
            
        # 2. Scale and translate points to fit the rectangle
        # We want to map the point cloud's bounding box [min_x, max_x] x [min_y, max_y]
        # to [0, w] x [0, h]? 
        # No, that would stretch the points anisotropically, changing the "circle" packing properties.
        # Instead, we should scale the points uniformly by a factor `s` such that they fit,
        # then center them.
        # However, the "radius" of the circles is not fixed by the point spacing alone if we allow variable radii.
        # But for a first pass, assume all circles have the same radius `r`.
        # Then the constraint is:
        # 1. Min distance between any two points * s >= 2r
        # 2. Distance from any point to nearest wall >= r
        
        # Let's find the optimal uniform scale `s` and center `(cx, cy)` for a given `r`.
        # Actually, let's just scale the points to fit the rectangle with some padding,
        # then compute the maximum possible `r`.
        
        # Strategy:
        # 1. Scale points so that their width is <= w and height <= h.
        #    s_x = w / w_pts
        #    s_y = h / h_pts
        #    s = min(s_x, s_y)
        # 2. Apply scale s to points.
        # 3. Center them in the rectangle.
        # 4. Compute min distance between points (d_min_pts).
        # 5. Compute min distance to walls (d_min_wall).
        # 6. The max radius r = min(d_min_pts / 2, d_min_wall).
        
        # This assumes uniform radius. Is this optimal?
        # Likely close to optimal for symmetric packings.
        # We can then try to "push" points to increase r?
        # For now, this geometric construction is fast and robust.
        
        s = min(w / w_pts, h / h_pts)
        
        # Scaled points
        scaled_pts = pts * s
        
        # Current bounds of scaled points
        s_min_x = np.min(scaled_pts[:, 0])
        s_max_x = np.max(scaled_pts[:, 0])
        s_min_y = np.min(scaled_pts[:, 1])
        s_max_y = np.max(scaled_pts[:, 1])
        
        # Center in rectangle
        # We want to center the bounding box of the points in the rectangle.
        offset_x = (w - (s_max_x - s_min_x)) / 2.0 - s_min_x
        offset_y = (h - (s_max_y - s_min_y)) / 2.0 - s_min_y
        
        centers = scaled_pts + np.array([offset_x, offset_y])
        
        # Compute min distance between points
        d_min_pts = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if d < d_min_pts:
                    d_min_pts = d
        
        # Compute min distance to walls
        d_min_wall = float('inf')
        for i in range(N):
            d_min_wall = min(d_min_wall, 
                             centers[i,0], 
                             w - centers[i,0], 
                             centers[i,1], 
                             h - centers[i,1])
        
        r = min(d_min_pts / 2.0, d_min_wall)
        
        # This gives a valid packing with uniform radius r.
        # Score = N * r
        score = N * r
        
        # We can try to improve this by allowing non-uniform radii?
        # Or by optimizing the positions slightly?
        # Let's try a quick local optimization to see if we can squeeze more out.
        # But first, let's record this as a candidate.
        
        if score > best_score:
            best_score = score
            # Create result array
            res = np.zeros((N, 3))
            res[:, 0] = centers[:, 0]
            res[:, 1] = centers[:, 1]
            res[:, 2] = r
            best_result = res
            
        # Optional: Try to improve with a few steps of gradient ascent on radii?
        # Or just try a few variations of the aspect ratio more finely?
        # The geometric construction is O(1) per aspect ratio.
        # We can afford to try more aspect ratios.
        
    # Let's do a finer search around the best aspect ratio found so far?
    # Or just a grid of aspect ratios.
    # The "ideal" ratio is a good starting point.
    # Let's try 10 more points around the ideal ratio.
    
    if best_result is not None:
        # Find the aspect ratio that gave the best score?
        # We didn't store it. Let's just re-run a fine grid.
        # The ideal ratio is w_pts / h_pts.
        # Let's try ratios from ideal*0.8 to ideal*1.2 with step 0.05.
        if time.time() < deadline - 10:
            for i in range(-20, 21):
                if time.time() > deadline:
                    break
                ratio = ideal_ratio * (1.0 + i * 0.01)
                if ratio <= 0: continue
                w = 2.0 * ratio / (1.0 + ratio)
                h = 2.0 / (1.0 + ratio)
                if w <= 0 or h <= 0: continue
                
                s = min(w / w_pts, h / h_pts)
                scaled_pts = pts * s
                s_min_x = np.min(scaled_pts[:, 0])
                s_max_x = np.max(scaled_pts[:, 0])
                s_min_y = np.min(scaled_pts[:, 1])
                s_max_y = np.max(scaled_pts[:, 1])
                
                offset_x = (w - (s_max_x - s_min_x)) / 2.0 - s_min_x
                offset_y = (h - (s_max_y - s_min_y)) / 2.0 - s_min_y
                centers = scaled_pts + np.array([offset_x, offset_y])
                
                d_min_pts = float('inf')
                for i_idx in range(N):
                    for j_idx in range(i_idx+1, N):
                        d = math.hypot(centers[i_idx,0] - centers[j_idx,0], centers[i_idx,1] - centers[j_idx,1])
                        if d < d_min_pts:
                            d_min_pts = d
                
                d_min_wall = float('inf')
                for i_idx in range(N):
                    d_min_wall = min(d_min_wall, 
                                     centers[i_idx,0], 
                                     w - centers[i_idx,0], 
                                     centers[i_idx,1], 
                                     h - centers[i_idx,1])
                
                r = min(d_min_pts / 2.0, d_min_wall)
                score = N * r
                
                if score > best_score:
                    best_score = score
                    res = np.zeros((N, 3))
                    res[:, 0] = centers[:, 0]
                    res[:, 1] = centers[:, 1]
                    res[:, 2] = r
                    best_result = res

    # Fallback
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
