"""3D Container Loading - Skyline-based approach with multiple orientations."""
import time
from itertools import product

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 115  # seconds, safety margin
    
    n = len(boxes)
    
    # Generate all 6 orientations for each box (w, h, d permutations)
    def get_orientations(w, h, d):
        orients = set()
        for perm in product([w, h, d], repeat=3):
            if len(set(perm)) == 3 or (len(set(perm)) == 2 and perm.count(perm[0]) != 3):
                orients.add(perm)
            elif len(set(perm)) == 1:
                orients.add(perm)
            else:
                # handle case with exactly 2 distinct values
                if perm.count(perm[0]) == 2 or perm.count(perm[1]) == 2:
                    orients.add(perm)
        # Actually, let's just generate all 6 permutations properly
        orients = set()
        dims = [w, h, d]
        for i in range(3):
            for j in range(3):
                if j != i:
                    for k in range(3):
                        if k != i and k != j:
                            orients.add((dims[i], dims[j], dims[k]))
        return list(orients)
    
    # Skyline-based 3D packing
    # A skyline point represents the highest z-coordinate at a given (x, y) position
    # We maintain a list of skyline points (x, y, z) where z is the height of the "sky"
    
    def can_place(skyline, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) can be placed at (x, y, z)."""
        # Check if the box fits within container
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Check against existing placements (stored in skyline)
        # For each skyline point, check if it interferes with the new box
        for sx, sy, sz in skyline:
            # The skyline point (sx, sy, sz) means that at position (sx, sy), 
            # the minimum z-coordinate for a new box to not overlap is sz
            # But actually, skyline represents the top surface
            # Let me reconsider: skyline points represent corners where the height changes
            
            # Check if the new box overlaps with any existing box
            # Two boxes overlap if their projections on all three axes overlap
            pass
        return True
    
    # Better approach: maintain list of placed boxes and check for overlap
    def boxes_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True
    
    def find_position(placed, bw, bh, bd):
        """Find a valid position for a box of size (bw, bh, bd)."""
        # Try to place at origin first, then try other positions
        # Use a simple approach: try all possible positions
        # For efficiency, we can use the fact that optimal positions are at corners
        
        # Candidate x positions: 0 and right edges of placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
        
        # Filter candidates that fit
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        # Try all combinations, prefer positions closer to origin (bottom-left-front)
        best_pos = None
        best_score = float('inf')
        
        for x in sorted(x_candidates):
            for y in sorted(y_candidates):
                for z in sorted(z_candidates):
                    # Check if this position is valid (no overlap)
                    valid = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if boxes_overlap(x, y, z, bw, bh, bd, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if valid:
                        # Score: prefer lower z, then lower y, then lower x
                        score = z * 1000000 + y * 1000 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Sort boxes by volume (largest first) for better packing
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # For each box, precompute orientations
    box_orientations = []
    for i in range(n):
        w, h, d = boxes[i]
        box_orientations.append(get_orientations(w, h, d))
    
    # Greedy placement
    placed = []  # list of (x, y, z, w, h, d)
    placement_results = [None] * n  # will store (bin, x, y, z, w, h, d) for each box
    
    for idx in box_indices:
        if time.time() - t0 > DEADLINE:
            break
        
        w, h, d = boxes[idx]
        orients = box_orientations[idx]
        
        # Try each orientation
        best_pos = None
        best_orient = None
        best_score = float('inf')
        
        for bw, bh, bd in orients:
            pos = find_position(placed, bw, bh, bd)
            if pos is not None:
                x, y, z = pos
                # Score: prefer lower z, then lower y, then lower x
                score = z * 1000000 + y * 1000 + x
                if score < best_score:
                    best_score = score
                    best_pos = pos
                    best_orient = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            placement_results[idx] = (0, x, y, z, bw, bh, bd)
        else:
            placement_results[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any remaining unplaced boxes
    for i in range(n):
        if placement_results[i] is None:
            w, h, d = boxes[i]
            placement_results[i] = (-1, 0, 0, 0, w, h, d)
    
    return placement_results
