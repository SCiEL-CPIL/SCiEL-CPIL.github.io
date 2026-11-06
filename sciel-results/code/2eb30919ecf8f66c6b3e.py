"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        # All 6 unique rotations (permutations of dimensions)
        orients = set()
        for perm in itertools.permutations((w, h, d)):
            orients.add(perm)
        # Return as list of tuples
        return list(orients)
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Placement list: each entry is (x, y, z, w, h, d)
    placements = []
    
    # For each box in sorted order, try to place it
    # We use a simple skyline-like approach but in 3D:
    # Try all orientations, and for each orientation, try to find a position
    # that doesn't overlap with existing placements and fits in container.
    
    # To speed up, we'll use a grid-based occupancy check or just brute force
    # with early termination. Given 60 boxes and 120s limit, brute force with
    # smart pruning should work.
    
    # Actually, let's use a more efficient approach: maintain a list of placed
    # boxes and for each new box, try candidate positions derived from existing
    # placements (like in 2D shelf packing but extended to 3D).
    
    # Candidate positions: for each existing placement, we can try placing the
    # new box adjacent to it in 6 directions (left, right, front, back, up, down)
    # plus the origin (0,0,0).
    
    # But this can be slow. Let's use a simpler heuristic:
    # Try to place each box at the "lowest" available position.
    
    # Define a function to check if a box at (x,y,z) with dims (w,h,d) overlaps
    # with any existing placement or goes out of bounds.
    def fits(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placements:
            # Check overlap: two boxes overlap if they overlap in all 3 axes
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return False
        return True
    
    # For each box, try orientations and find a valid position
    # To find positions, we'll use a grid of candidate positions derived from
    # the corners of existing placements.
    
    # Build candidate x, y, z positions
    def get_candidate_positions():
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in placements:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        # Also add W, H, D as boundaries (but we check bounds in fits)
        xs.add(W)
        ys.add(H)
        zs.add(D)
        return sorted(xs), sorted(ys), sorted(zs)
    
    # For each box in sorted order
    result = [None] * n  # Will store placement info for each original index
    
    for (idx, w, h, d, vol) in indexed_boxes:
        if time.time() > deadline:
            # Return whatever we have; unplaced boxes get -1
            break
        
        orients = get_orientations(w, h, d)
        placed = False
        
        # Get candidate positions
        xs, ys, zs = get_candidate_positions()
        
        # Sort orientations by some heuristic (e.g., align longest dim with longest container dim)
        # For now, try all orientations
        for ow, oh, od in orients:
            if placed:
                break
            # Try all candidate positions
            for cx in xs:
                if placed:
                    break
                if cx + ow > W:
                    continue
                for cy in ys:
                    if placed:
                        break
                    if cy + oh > H:
                        continue
                    for cz in zs:
                        if placed:
                            break
                        if cz + od > D:
                            continue
                        if fits(cx, cy, cz, ow, oh, od):
                            placements.append((cx, cy, cz, ow, oh, od))
                            result[idx] = (0, cx, cy, cz, ow, oh, od)
                            placed = True
                            break
        
        if not placed:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unprocessed boxes (if we hit deadline)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
