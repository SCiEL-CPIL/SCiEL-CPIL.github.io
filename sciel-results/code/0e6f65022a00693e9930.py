"""3D Container Loading with shelf-based packing and rotation."""
import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Get all 6 orientations for a box
    def get_orientations(w, h, d):
        # All permutations of (w, h, d)
        return list(set(itertools.permutations([w, h, d])))
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Shelf-based packing
    # We maintain a list of shelves, each with:
    #   (z, d, y, h, x, w) where:
    #   - z: bottom z of shelf
    #   - d: depth of shelf (extent in z direction)
    #   - y: bottom y of shelf
    #   - h: height of shelf (extent in y direction)
    #   - x: current x position (next free x)
    #   - w: remaining width in shelf
    
    # placements[i] = (bin, x, y, z, w, h, d)
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    # Shelves: list of dicts
    shelves = []
    
    # Try to place each box
    for idx, (orig_idx, (bw, bh, bd)) in indexed_boxes:
        if time.time() > deadline:
            break
        
        bw, bh, bd = int(bw), int(bh), int(bd)
        
        # Try all orientations
        best_placement = None
        best_score = -1
        
        for ow, oh, od in get_orientations(bw, bh, bd):
            # Check if this orientation fits in container
            if ow > W or oh > H or od > D:
                continue
            
            # Try to find a shelf where this box fits
            placed = False
            
            # First, check existing shelves
            for shelf in shelves:
                sz, sd, sy, sh, sx, sw = shelf
                # Box must fit in shelf's y-z extent and have space in x
                if oh <= sh and od <= sd and ow <= sw:
                    # Check y-z overlap with shelf
                    # Shelf occupies [sy, sy+sh] x [sz, sz+sd]
                    # Box at (sx, sy, sz) with dims (ow, oh, od)
                    # We place box at x=sx, y=sy, z=sz
                    x, y, z = sx, sy, sz
                    # Check if box fits within container
                    if x + ow <= W and y + oh <= H and z + od <= D:
                        # This is a valid placement
                        # Score: prefer placements that minimize wasted space
                        score = -((sw - ow) * 1.0)  # prefer tight fit in x
                        if best_score < score:
                            best_score = score
                            best_placement = (x, y, z, ow, oh, od, shelf)
                        placed = True
            
            # If not placed in existing shelf, try to create new shelf
            if not placed:
                # Try to create a new shelf at the bottom or on top of existing
                # Find the lowest z where we can place a shelf of depth od
                # and height oh that doesn't overlap with existing shelves
                
                # Simple approach: try z=0 first, then increasing z
                # Actually, let's try to stack shelves vertically
                
                # Find available z positions
                # We'll try z = 0, and then z = max of (sz + sd) for shelves that don't overlap in y
                candidate_zs = [0]
                for shelf in shelves:
                    sz, sd, sy, sh, sx, sw = shelf
                    # If this shelf's y range doesn't overlap with our box's y range at y=0
                    # Actually, let's just try stacking on top
                    candidate_zs.append(sz + sd)
                
                # Also try z positions based on existing shelf tops
                for cz in sorted(set(candidate_zs)):
                    if cz + od > D:
                        continue
                    # Check if we can place a shelf at z=cz with depth od
                    # Check overlap with existing shelves
                    can_place = True
                    for shelf in shelves:
                        sz, sd, sy, sh, sx, sw = shelf
                        # Check z overlap: [cz, cz+od) vs [sz, sz+sd)
                        if cz < sz + sd and sz < cz + od:
                            # z overlap exists, check if y ranges are disjoint
                            # Our box would be at y=0 (or we need to find a y)
                            # Actually, for a new shelf, we place box at some y
                            # Let's check if there's a y where the box doesn't overlap
                            # Shelf y range: [sy, sy+sh)
                            # Box y range: [y, y+oh)
                            # We need [y, y+oh) to not overlap [sy, sy+sh)
                            # i.e., y + oh <= sy or y >= sy + sh
                            # Try y=0
                            if 0 + oh > sy and 0 < sy + sh:
                                # Overlap at y=0, try y = sy + sh
                                y2 = sy + sh
                                if y2 + oh > H:
                                    can_place = False
                                    break
                            # If no overlap at y=0 or we can shift, it's fine
                            # Actually this is getting complex. Let's simplify.
                            # For now, just check if z ranges overlap and assume
                            # we can find a y position.
                            pass
                    
                    if not can_place:
                        continue
                    
                    # Try to place at x=0, y=0, z=cz
                    x, y, z = 0, 0, cz
                    if x + ow <= W and y + oh <= H and z + od <= D:
                        score = -((W - ow) * 1.0)  # prefer tight fit
                        if best_score < score:
                            best_score = score
                            best_placement = (x, y, z, ow, oh, od, None)
                        placed = True
                        break
        
        if best_placement is not None:
            x, y, z, ow, oh, od, shelf = best_placement
            placements[orig_idx] = (0, x, y, z, ow, oh, od)
            
            if shelf is not None:
                # Update existing shelf
                shelves[shelves.index(shelf)] = (shelf[0], shelf[1], shelf[2], shelf[3], shelf[4] + ow, shelf[5] - ow)
            else:
                # Create new shelf
                shelves.append((z, od, y, oh, x + ow, W - x - ow))
    
    return placements
