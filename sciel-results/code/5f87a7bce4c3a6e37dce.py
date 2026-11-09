import time
import random
import itertools
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s finish)
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Helper to check time
    def check_time():
        return time.time() > deadline

    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    # Use a 3D grid for occupancy if dimensions are reasonable
    # This allows O(1) or O(volume) checks instead of O(N) list checks
    # If dimensions are too large, we fall back to list-based
    MAX_DIM = 200
    use_grid = (W <= MAX_DIM and H <= MAX_DIM and D <= MAX_DIM)
    
    if use_grid:
        grid = np.zeros((W, H, D), dtype=bool)
    else:
        grid = None

    def can_place_grid(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any cell in the box volume is occupied
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True

    def place_grid(x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = True

    def can_place_list(placed, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # Overlap check
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def generate_candidates(placed, W, H, D):
        """
        Generate candidate positions.
        We use a set of points derived from the corners of placed boxes
        plus the origin. We prioritize points that are "lower" (z), then "left" (x), then "back" (y).
        """
        cands = set()
        cands.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Right face
            cands.add((px + pw, py, pz))
            # Front face
            cands.add((px, py + ph, pz))
            # Top face
            cands.add((px, py, pz + pd))
            
            # Edges (intersections of faces)
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            
            # Corner
            cands.add((px + pw, py + ph, pz + pd))
            
            # Align to axes (0 plane)
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))
            
            # Also try aligning to the "far" side of the container if close?
            # No, let's stick to local corners for efficiency.

        valid_cands = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid_cands.append((x, y, z))
        
        # Sort by priority: Lowest Z, then Lowest X, then Lowest Y
        # This implements a "Bottom-Left-Far" (or similar) heuristic
        valid_cands.sort(key=lambda p: (p[2], p[0], p[1]))
        return valid_cands

    def pack_pass(order, boxes, W, H, D, grid):
        """
        Attempt to pack boxes in the given order.
        Returns list of placements and total volume.
        """
        placed = []
        result = [None] * len(boxes)
        total_vol = 0
        
        for idx in order:
            if check_time():
                break
                
            w, h, d = boxes[idx]
            rots = get_rotations(w, h, d)
            
            # Filter rotations that fit in container
            valid_rots = [(rw, rh, rd) for rw, rh, rd in rots if rw <= W and rh <= H and rd <= D]
            if not valid_rots:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                continue
                
            # Generate candidates based on current placement
            # To save time, we can cache candidates if no box was placed in the last step?
            # But placement changes the space, so we must regenerate or update.
            # Regenerating is O(N_placed). For 60 boxes, this is small.
            cands = generate_candidates(placed, W, H, D)
            
            best_pos = None
            best_rot = None
            
            # We iterate candidates in priority order.
            # We iterate rotations for each candidate.
            # To improve packing, we might want to try the "best" rotation for a position?
            # Or just first fit. First fit is standard for greedy.
            
            for cx, cy, cz in cands:
                if check_time():
                    break
                    
                for rw, rh, rd in valid_rots:
                    if use_grid:
                        if can_place_grid(cx, cy, cz, rw, rh, rd):
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
                            break
                    else:
                        if can_place_list(placed, cx, cy, cz, rw, rh, rd):
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
                            break
                            
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                if use_grid:
                    place_grid(bx, by, bz, bw, bh, bd)
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                total_vol += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        # Fill any remaining Nones
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
                
        return result, total_vol

    # Sorting strategies
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    def sort_vol(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_max(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_surf(lst):
        return sorted(lst, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_base(lst):
        # Sort by base area (w*h)
        return sorted(lst, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    def sort_aspect(lst):
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))

    strategies = [sort_vol, sort_max, sort_surf, sort_base, sort_aspect]

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_time():
            break
        order = [t[0] for t in sf(indexed)]
        
        # Reset grid
        if use_grid:
            grid.fill(False)
        
        res, vol = pack_pass(order, boxes, W, H, D, grid)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    # We use a fixed seed for reproducibility but random orders
    rng = random.Random(42)
    max_restarts = 500 # We will stop when time runs out
    
    # Pre-compute a few random orders to avoid overhead? No, generation is fast.
    for _ in range(max_restarts):
        if check_time():
            break
            
        # Create a random order with some bias towards larger boxes
        # Shuffle the indexed boxes, but keep large ones somewhat prioritized?
        # Simple random shuffle of indices
        order = [t[0] for t in indexed]
        rng.shuffle(order)
        
        # Alternatively, sort by volume + noise
        # Let's try: sort by volume, then apply random swaps to top K
        # Or just pure random for diversity.
        
        if use_grid:
            grid.fill(False)
            
        res, vol = pack_pass(order, boxes, W, H, D, grid)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
