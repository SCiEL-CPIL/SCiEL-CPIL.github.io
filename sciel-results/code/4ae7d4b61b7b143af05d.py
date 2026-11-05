"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Preprocess: compute volume and sort by volume descending
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # State: list of placed boxes as (x, y, z, w, h, d)
    placed = []
    # For each original index, store (bin, x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Grid resolution: use a reasonable resolution that balances accuracy and speed
    # We'll use a grid where each cell is ~1 unit if container is small, otherwise larger
    max_grid_size = 50
    grid_w = min(max_grid_size, max(1, int(W)))
    grid_h = min(max_grid_size, max(1, int(H)))
    grid_d = min(max_grid_size, max(1, int(D)))
    
    # Occupancy grid: 3D array where each cell is 0 (empty) or 1 (occupied)
    # We'll use a coarser grid for performance
    occ = np.zeros((grid_w, grid_h, grid_d), dtype=np.uint8)
    
    def update_occupancy(x, y, z, w, h, d, val):
        """Update occupancy grid for a box placement."""
        x0 = int(x / W * grid_w)
        x1 = int((x + w) / W * grid_w)
        y0 = int(y / H * grid_h)
        y1 = int((y + h) / H * grid_h)
        z0 = int(z / D * grid_d)
        z1 = int((z + d) / D * grid_d)
        
        x0 = max(0, min(grid_w - 1, x0))
        x1 = max(0, min(grid_w, x1))
        y0 = max(0, min(grid_h - 1, y0))
        y1 = max(0, min(grid_h, y1))
        z0 = max(0, min(grid_d - 1, z0))
        z1 = max(0, min(grid_d, z1))
        
        if x1 > x0 and y1 > y0 and z1 > z0:
            occ[x0:x1, y0:y1, z0:z1] = val
    
    def get_orientations(w, h, d):
        """Generate all 6 unique orientations as tuples."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def fits(x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in container without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def candidate_positions():
        """Generate candidate positions by combining x/y/z coordinates from placed boxes."""
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter coordinates to be within container bounds
        xs = [x for x in xs if 0 <= x <= W]
        ys = [y for y in ys if 0 <= y <= H]
        zs = [z for z in zs if 0 <= z <= D]
        
        # If too many combinations, limit to a subset for speed
        max_candidates = 2000
        if len(xs) * len(ys) * len(zs) > max_candidates:
            # Prioritize smaller coordinates (closer to origin)
            xs.sort()
            ys.sort()
            zs.sort()
            # Take first N from each dimension
            limit = max(1, int((max_candidates ** (1/3)) + 1))
            xs = xs[:limit]
            ys = ys[:limit]
            zs = zs[:limit]
        
        positions = []
        for x in xs:
            for y in ys:
                for z in zs:
                    positions.append((x, y, z))
        
        # Sort by proximity to origin for deterministic ordering
        positions.sort(key=lambda p: (p[0], p[1], p[2]))
        
        return positions
    
    def largest_empty_space_score(x, y, z, w, h, d):
        """
        Score a position by the largest contiguous empty space in the box's footprint.
        Higher score means more room for future boxes (creates larger voids).
        We approximate this by looking at the empty cells in a region around the box.
        """
        # Convert to grid coordinates
        x0 = int(x / W * grid_w)
        x1 = int((x + w) / W * grid_w)
        y0 = int(y / H * grid_h)
        y1 = int((y + h) / H * grid_h)
        z0 = int(z / D * grid_d)
        z1 = int((z + d) / D * grid_d)
        
        x0 = max(0, min(grid_w - 1, x0))
        x1 = max(0, min(grid_w, x1))
        y0 = max(0, min(grid_h - 1, y0))
        y1 = max(0, min(grid_h, y1))
        z0 = max(0, min(grid_d - 1, z0))
        z1 = max(0, min(grid_d, z1))
        
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return 0.0
        
        # Look at a region slightly larger than the box to estimate available space
        margin = max(1, min(3, grid_w // 10, grid_h // 10, grid_d // 10))
        rx0 = max(0, x0 - margin)
        rx1 = min(grid_w, x1 + margin)
        ry0 = max(0, y0 - margin)
        ry1 = min(grid_h, y1 + margin)
        rz0 = max(0, z0 - margin)
        rz1 = min(grid_d, z1 + margin)
        
        if rx1 <= rx0 or ry1 <= ry0 or rz1 <= rz0:
            return 0.0
        
        region = occ[rx0:rx1, ry0:ry1, rz0:rz1]
        empty_fraction = 1.0 - (np.sum(region) / region.size if region.size > 0 else 1.0)
        
        return empty_fraction
    
    def best_fit_position(w, h, d):
        """Find the best position for a box with given dimensions."""
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidate_positions():
            if not fits(x, y, z, w, h, d):
                continue
            
            # Primary: maximize largest empty space (create voids for future boxes)
            # Secondary: minimize x+y+z (compactness)
            empty_score = largest_empty_space_score(x, y, z, w, h, d)
            compactness = x + y + z
            
            # Combine: prefer positions with more empty space, break ties by compactness
            # Use a weighted combination
            score = -empty_score * 100.0 + compactness * 0.01
            
            if score < best_score:
                best_score = score
                best_pos = (x, y, z)
        
        return best_pos
    
    # Main placement loop
    for (w, h, d, orig_idx) in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = get_orientations(w, h, d)
        
        best_placement = None
        best_score = float('inf')
        
        for (ow, oh, od) in orients:
            pos = best_fit_position(ow, oh, od)
            if pos is not None:
                x, y, z = pos
                # Score: prefer placements that maximize empty space, then minimize coordinates
                empty_score = largest_empty_space_score(x, y, z, ow, oh, od)
                compactness = x + y + z
                score = -empty_score * 100.0 + compactness * 0.01
                if score < best_score:
                    best_score = score
                    best_placement = (x, y, z, ow, oh, od)
        
        if best_placement is not None:
            x, y, z, ow, oh, od = best_placement
            placed.append((x, y, z, ow, oh, od))
            result[orig_idx] = (0, x, y, z, ow, oh, od)
            update_occupancy(x, y, z, ow, oh, od, 1)
    
    return result
# EVOLVE-BLOCK-END
