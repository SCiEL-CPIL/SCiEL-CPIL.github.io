"""3D Container Loading with Extreme Point Representation (EPR) and multi-strategy search."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    def check_overlap(x, y, z, w, h, d, placed):
        """Check if a box at (x,y,z) with size (w,h,d) overlaps with any placed box."""
        for px, py, pz, pw, ph, pd, _ in placed:
            # Overlap if intervals overlap in all 3 dimensions
            if x < px + pw and px < x + w and \
               y < py + ph and py < y + h and \
               z < pz + pd and pz < z + d:
                return True
        return False

    def pack_with_epr(ordered_indices):
        """
        Pack boxes using Extreme Point Representation.
        Returns (placements_list, total_volume)
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        placed = []  # List of (x, y, z, w, h, d, index)
        extreme_points = [(0, 0, 0)]
        total_volume = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations
            for ow, oh, od in orients:
                # Try all extreme points
                for ex, ey, ez in extreme_points:
                    # Bounds check
                    if ex + ow > W or ey + oh > H or ez + od > D:
                        continue
                    
                    # Overlap check
                    if check_overlap(ex, ey, ez, ow, oh, od, placed):
                        continue
                    
                    # Score: Prefer lower Y (stacking), then lower Z, then lower X
                    # This tends to fill from the bottom up and front to back
                    score = ey * 1000000 + ez * 1000 + ex
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (ex, ey, ez, ow, oh, od)
            
            if best_pos:
                ex, ey, ez, ow, oh, od = best_pos
                placements[idx] = (0, ex, ey, ez, ow, oh, od)
                placed.append((ex, ey, ez, ow, oh, od, idx))
                total_volume += ow * oh * od
                
                # Generate new extreme points
                # The new corners created by this box
                new_pts = [
                    (ex + ow, ey, ez),
                    (ex, ey + oh, ez),
                    (ex, ey, ez + od)
                ]
                
                # Combine old and new points
                all_pts = list(extreme_points) + new_pts
                
                # Prune points that are covered by the NEW box
                filtered_pts = []
                for p in all_pts:
                    px, py, pz = p
                    # Check if p is inside the new box
                    if ex <= px < ex + ow and ey <= py < ey + oh and ez <= pz < ez + od:
                        continue
                    # Also check if p is out of bounds? No, points can be on the boundary.
                    # But points > W, H, D are useless.
                    if px > W or py > H or pz > D:
                        continue
                    filtered_pts.append(p)
                
                # Remove duplicates
                extreme_points = list(set(filtered_pts))
        
        return placements, total_volume

    # Strategies for sorting
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    # Run each static sorting strategy
    for sort_key in strategies:
        if time.time() > deadline - 10:
            break
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = pack_with_epr(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v

    # If we have time, try some random restarts
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            random.shuffle(indexed_boxes)
            ordered = indexed_boxes[:]
            p, v = pack_with_epr(ordered)
            if v > best_volume:
                best_placements = p
                best_volume = v

    return best_placements
# EVOLVE-BLOCK-END
