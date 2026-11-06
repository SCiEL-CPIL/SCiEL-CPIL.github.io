"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute all 6 rotations for each box
    def get_rotations(w, h, d):
        # All 6 unique permutations of (w, h, d)
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Sort boxes: largest volume first, with tie-breaking on max dimension
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # Placed boxes: list of (x, y, z, w, h, d)
    placed = []
    results = [None] * n  # Will store (bin, x, y, z, w, h, d) for each original box index
    
    def fits(x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check overlap with all placed boxes
        for px, py, pz, pw, ph, pd in placed:
            # Overlap if intervals overlap on all 3 axes
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True
    
    # For efficiency, collect candidate x, y, z values
    def get_candidate_axes():
        xs = {0}
        ys = {0}
        zs = {0}
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        return sorted(xs), sorted(ys), sorted(zs)
    
    # Skyline-based placement: try to find a position that minimizes z, then x, then y
    # but also considers "fit" quality to reduce fragmentation
    def find_best_position(rw, rh, rd):
        """Find the best position for a box with dimensions (rw, rh, rd).
        Returns (x, y, z) or None if no position found."""
        xs, ys, zs = get_candidate_axes()
        best_pos = None
        best_key = None
        
        for cx in xs:
            if cx + rw > W:
                continue
            for cy in ys:
                if cy + rh > H:
                    continue
                for cz in zs:
                    if cz + rd > D:
                        continue
                    if not fits(cx, cy, cz, rw, rh, rd):
                        continue
                    # Best-Fit Decreasing Height: minimize z, then x, then y
                    key = (cz, cx, cy)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (cx, cy, cz)
        
        return best_pos
    
    for orig_idx, (bw_orig, bh_orig, bd_orig) in indexed_boxes:
        if time.time() > deadline:
            # Mark remaining as not placed
            for i in range(n):
                if results[i] is None:
                    results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            return results
        
        rots = get_rotations(bw_orig, bh_orig, bd_orig)
        best_pos = None
        best_key = None
        
        for (rw, rh, rd) in rots:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                cx, cy, cz = pos
                key = (cz, cx, cy)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz, rw, rh, rd)
        
        if best_pos is not None:
            cx, cy, cz, rw, rh, rd = best_pos
            placed.append((cx, cy, cz, rw, rh, rd))
            results[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
        else:
            results[orig_idx] = (-1, 0, 0, 0, bw_orig, bh_orig, bd_orig)
    
    # Fill in any remaining None (shouldn't happen if we process all boxes)
    for i in range(n):
        if results[i] is None:
            results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return results
# EVOLVE-BLOCK-END
