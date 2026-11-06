import time
import itertools
import heapq
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # 1. Preprocess boxes: Sort by volume descending, then by max dimension
    # We keep track of original indices
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    # Sort by volume descending
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Function to get all unique orientations for a box
    def get_orientations(w, h, d):
        # Return all 6 permutations, but remove duplicates
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)

    # Extreme Points / Skyline Heuristic
    # We maintain a list of "extreme points" (x, y, z) where a new box could be placed.
    # Initially, the only extreme point is (0, 0, 0).
    # When a box is placed at (x, y, z) with dimensions (w, h, d),
    # we generate new candidate points.
    
    # To be efficient, we can use a priority queue or just a list.
    # Given 60 boxes, a list is fine.
    
    def pack_boxes(order):
        """
        Packs boxes in the given order using the Extreme Points heuristic.
        Returns a list of placements (bin, x, y, z, w, h, d) for all boxes.
        """
        placements = [None] * n
        # extreme_points is a list of (x, y, z)
        extreme_points = [(0, 0, 0)]
        
        # We need to track which boxes are placed where
        # box_placements[i] = (x, y, z, w, h, d) or None if not placed
        # Actually, we just need to check overlap with already placed boxes.
        # Since n is small (60), checking against all placed boxes is O(N^2 * 6) which is fine.
        
        placed_boxes = []  # List of (x, y, z, w, h, d)
        
        for idx, orig_idx, w, h, d in order:
            if time.time() > deadline:
                break
                
            best_pos = None
            best_orient = None
            best_score = -1  # We want to minimize (x + y + z) or similar? 
            # Actually, a common heuristic is to place as close to (0,0,0) as possible.
            # Let's try minimizing x + y + z.
            
            # Get orientations
            orients = get_orientations(w, h, d)
            
            # Filter orientations that fit in container
            valid_orients = [(ow, oh, od) for ow, oh, od in orients if ow <= W and oh <= H and od <= D]
            if not valid_orients:
                continue
                
            # Try all extreme points and all valid orientations
            for px, py, pz in extreme_points:
                for ow, oh, od in valid_orients:
                    # Check bounds
                    if px + ow > W or py + oh > H or pz + od > D:
                        continue
                    
                    # Check overlap with placed boxes
                    overlap = False
                    for bx, by, bz, bw, bh, bd in placed_boxes:
                        # Check if boxes overlap
                        # [px, px+ow] vs [bx, bx+bw]
                        if px < bx + bw and px + ow > bx and \
                           py < by + bh and py + oh > by and \
                           pz < bz + bd and pz + od > bz:
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    # Valid placement found
                    # Heuristic: minimize x+y+z (pack towards corner)
                    # Or minimize x + 2*y + 3*z? Let's stick to x+y+z for now.
                    score = px + py + pz
                    if best_pos is None or score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placements[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                placed_boxes.append((bx, by, bz, bw, bh, bd))
                
                # Generate new extreme points
                # The new box creates new "corner" points.
                # Points to consider:
                # (bx + bw, by, bz)
                # (bx, by + bh, bz)
                # (bx, by, bz + bd)
                # (bx + bw, by + bh, bz)
                # (bx + bw, by, bz + bd)
                # (bx, by + bh, bz + bd)
                # (bx + bw, by + bh, bz + bd)
                
                new_points = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd)
                ]
                
                # Add to extreme_points if they are within bounds and not already there
                # We can use a set for deduplication
                ep_set = set(extreme_points)
                for np in new_points:
                    if np[0] <= W and np[1] <= H and np[2] <= D:
                        if np not in ep_set:
                            ep_set.add(np)
                            extreme_points.append(np)
                
                # Optional: Clean up extreme points that are "dominated"?
                # If point A is at (x1, y1, z1) and point B is at (x2, y2, z2)
                # and x1<=x2, y1<=y2, z1<=z2, then B is dominated by A?
                # Not necessarily, because a box might fit at B but not A if A is blocked.
                # However, if A is free and B is free, and A is "closer" to origin,
                # placing at A is generally better.
                # But if A is blocked by other boxes, B might be the only option.
                # So we shouldn't prune too aggressively.
                # For 60 boxes, the number of extreme points stays manageable.
                
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any remaining None placements (if we broke early due to deadline)
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return placements

    # 2. Define the order
    # FFD: First Fit Decreasing (by volume)
    # The order is the sorted list of (index, w, h, d)
    # pack_boxes expects (idx, orig_idx, w, h, d) ? 
    # My pack_boxes signature: for idx, orig_idx, w, h, d in order
    # So order should be a list of tuples (something, orig_idx, w, h, d).
    # The 'idx' in the loop isn't used in my current pack_boxes implementation, 
    # except for iteration. I can just pass 0.
    
    order = [(i, i, w, h, d) for i, (w, h, d) in enumerate(sorted([(w,h,d) for w,h,d in boxes], key=lambda x: x[0]*x[1]*x[2], reverse=True))]
    # Wait, I need to preserve original indices correctly.
    # Let's re-construct the order properly.
    
    # Create a list of (volume, orig_idx, w, h, d)
    vol_idx = []
    for i, (w, h, d) in enumerate(boxes):
        vol_idx.append((w*h*d, i, w, h, d))
    vol_idx.sort(key=lambda x: x[0], reverse=True)
    
    # The order for pack_boxes
    # pack_boxes iterates: for idx, orig_idx, w, h, d in order
    # So we need to map vol_idx to this format.
    final_order = []
    for rank, (vol, orig_idx, w, h, d) in enumerate(vol_idx):
        final_order.append((rank, orig_idx, w, h, d))
        
    # 3. Run the packing
    result = pack_boxes(final_order)
    
    return result
# EVOLVE-BLOCK-END
