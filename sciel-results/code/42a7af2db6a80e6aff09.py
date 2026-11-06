"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Generate all 6 orientations for each box
    def get_orientations(w, h, d):
        dims = [w, h, d]
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        return orients
    
    # Precompute orientations for all boxes
    box_orients = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending (larger boxes first for better packing)
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Greedy placement using a list of empty rectangular prisms
    # Start with the full container as one empty prism
    # Each placement splits the prism into up to 3 new prisms
    
    def try_place(boxes, container, time_limit):
        W, H, D = container
        n = len(boxes)
        
        # Empty prisms: list of (x, y, z, w, h, d)
        empty = [(0, 0, 0, W, H, D)]
        
        # Result: for each box index, (bin, x, y, z, w, h, d)
        # bin=0 if placed, -1 if not
        result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
        # Sort by volume descending
        order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = -1
            best_empty_idx = -1
            best_split_info = None
            
            # Try each empty prism
            for ei, (ex, ey, ez, ew, eh, ed) in enumerate(empty):
                if time.time() > time_limit:
                    break
                for (ow, oh, od) in orients:
                    # Check if orientation fits in this prism
                    if ow > ew or oh > eh or od > ed:
                        continue
                    
                    # Try placing at the corner (ex, ey, ez) of this prism
                    # This is a simple heuristic: place at the minimum corner
                    px, py, pz = ex, ey, ez
                    
                    # Calculate a score: prefer placements that leave more regular space
                    # Score based on how much "wasted" space is created
                    # After placing (ow, oh, od) at (px, py, pz) in prism (ex, ey, ez, ew, eh, ed):
                    # New prisms:
                    # 1. Right: (px+ow, py, pz, ew-ow, eh, ed) if ew-ow > 0
                    # 2. Above: (px, py+oh, pz, ow, eh-oh, ed) if eh-oh > 0
                    # 3. Behind: (px, py, pz+od, ow, oh, ed-od) if ed-od > 0
                    # Note: these overlap in corners, but that's okay for a heuristic.
                    # Actually, we need to be careful. The standard way is:
                    # - Split along x: left part (ex, ey, ez, ow, eh, ed) and right part (ex+ow, ey, ez, ew-ow, eh, ed)
                    # - Then in the right part, split along y: front (ex+ow, ey, ez, ew-ow, oh, ed) and back (ex+ow, ey+oh, ez, ew-ow, eh-oh, ed)
                    # - Then in the back part, split along z: ... 
                    # This gets complicated. Let's use a simpler approach.
                    
                    # For scoring, use the sum of dimensions of the "gaps"
                    gap_x = ew - ow
                    gap_y = eh - oh
                    gap_z = ed - od
                    # Prefer smaller gaps (tighter fit)
                    score = -(gap_x + gap_y + gap_z)
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_orient = (ow, oh, od)
                        best_empty_idx = ei
            
            if best_pos is not None and time.time() <= time_limit:
                px, py, pz = best_pos
                ow, oh, od = best_orient
                ei = best_empty_idx
                ex, ey, ez, ew, eh, ed = empty[ei]
                
                # Remove the old empty prism
                empty.pop(ei)
                
                # Add new empty prisms from the split
                # The placed box occupies (px, py, pz) to (px+ow, py+oh, pz+od)
                # Since we placed at the corner (ex, ey, ez), px=ex, py=ey, pz=ez
                # New prisms:
                if ew > ow:
                    empty.append((ex+ow, ey, ez, ew-ow, eh, ed))
                if eh > oh:
                    empty.append((ex, ey+oh, ez, ow, eh-oh, ed))
                if ed > od:
                    empty.append((ex, ey, ez+od, ow, oh, ed-od))
                
                result[idx] = (0, px, py, pz, ow, oh, od)
        
        return result
    
    # Try the greedy approach
    result = try_place(boxes, container, deadline)
    
    return result
# EVOLVE-BLOCK-END
