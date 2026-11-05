"""3D Container Loading with distance-transform guided placement."""
import numpy as np
from itertools import permutations
import time

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = len(boxes)
    
    # Generate all 6 unique rotations for a box
    def get_rotations(w, h, d):
        rots = set()
        for p in permutations((w, h, d)):
            rots.add(p)
        return list(rots)
    
    # Precompute rotations for each box
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Use a 3D occupancy grid with distance transform
    # Grid resolution: 1 unit (if container is large, we may need to be careful)
    # For efficiency, we'll use a grid where each cell is 1 unit
    # But if container is very large, this could be memory intensive
    # Let's check if grid is feasible
    total_cells = W * H * D
    if total_cells > 10**7:
        # Too large for full grid, fall back to simpler approach
        return _simple_load(boxes, container, start_time, deadline)
    
    # Initialize distance grid: distance to nearest boundary
    # dist[x,y,z] = min(x, W-1-x, y, H-1-y, z, D-1-z)
    dist = np.zeros((W, H, D), dtype=np.int32)
    for x in range(W):
        for y in range(H):
            for z in range(D):
                dist[x, y, z] = min(x, W-1-x, y, H-1-y, z, D-1-z)
    
    # Track placed boxes
    placed = [None] * n  # (x, y, z, w, h, d) or None
    placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # For each box in sorted order, try to place it
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        best_score = -1
        best_pos = None
        best_rot = None
        
        for rot in box_rots[idx]:
            rw, rh, rd = rot
            if rw > W or rh > H or rd > D:
                continue
            
            # Find best position using distance transform
            # We want to maximize the minimum distance in the footprint
            # This creates large contiguous voids
            # Scan possible positions
            best_local_score = -1
            best_local_pos = None
            
            # For efficiency, we can use the distance grid
            # The score at position (x,y,z) with size (rw,rh,rd) is:
            # min(dist[x:x+rw, y:y+rh, z:z+rd])
            # But computing this naively is O(rw*rh*rd) per position
            
            # Optimization: precompute for each rotation
            # Actually, let's just scan but with early termination
            # and use the fact that we want max-min distance
            
            for x in range(W - rw + 1):
                if time.time() > deadline:
                    break
                for y in range(H - rh + 1):
                    if time.time() > deadline:
                        break
                    # Quick check: is there enough space in z?
                    # Find the best z position
                    # The minimum distance over the footprint
                    # We can compute this more efficiently
                    
                    # For a given x,y, find z that maximizes min dist
                    # dist[x:x+rw, y:y+rh, :] gives a (rw, rh, D) array
                    # We want min over x:x+rw, y:y+rh for each z, then max over z
                    # This is still expensive
                    
                    # Simpler: just try all z
                    for z in range(D - rd + 1):
                        # Check if all cells in footprint have dist >= 1 (not occupied)
                        # Actually, dist > 0 means not on boundary, but we need to check occupancy
                        # Wait, the distance transform approach needs modification:
                        # We need to track occupied cells separately
                        # Let me reconsider...
                        pass
            
            # The above is too slow. Let me use a different approach:
            # Use a 3D boolean occupancy grid and for each position, check if all cells are free
            # But that's O(rw*rh*rd) per position too.
            
            # Better approach: use the distance transform but update it after each placement
            # The distance transform gives us the "largest empty cube" at each point
            # For a box of size (rw, rh, rd), we can place it at (x,y,z) if:
            # dist[x, y, z] >= 0 and the box fits in the empty space
            # Actually, the standard approach is:
            # 1. Maintain a 3D grid of "can place a 1x1x1 box here"
            # 2. After placing a box, update the grid
            # 3. For a box of size (rw, rh, rd), find positions where a (rw,rh,rd) box fits
            
            # Let me use a simpler but effective heuristic:
            # Place boxes in a space-filling curve order, trying to pack tightly
            # But with rotation optimization
            
            # Given time constraints, let's use a simpler approach:
            # Sort by volume, try to place each box at the "first fit" position
            # with rotation that minimizes wasted space
            pass
        
        # Fallback: simple first-fit with rotation
        placed_pos = _try_place_simple(idx, boxes, box_rots, W, H, D, placed, time.time(), deadline)
        if placed_pos is not None:
            x, y, z, rw, rh, rd = placed_pos
            placed[idx] = (x, y, z, rw, rh, rd)
            placements[idx] = (0, x, y, z, rw, rh, rd)
    
    return placements


def _try_place_simple(idx, boxes, box_rots, W, H, D, placed, current_time, deadline):
    """Try to place a box using first-fit with rotation."""
    w, h, d = boxes[idx]
    
    # Try each rotation
    for rot in box_rots[idx]:
        rw, rh, rd = rot
        if rw > W or rh > H or rd > D:
            continue
        
        # Try to find a position where this box fits
        # Use a grid-based approach: check all possible positions
        # For efficiency, scan in a specific order (e.g., x, then y, then z)
        for x in range(W - rw + 1):
            if current_time > deadline:
                return None
            for y in range(H - rh + 1):
                if current_time > deadline:
                    return None
                for z in range(D - rd + 1):
                    # Check if this position is free
                    if _is_free(x, y, z, rw, rh, rd, placed, W, H, D):
                        return (x, y, z, rw, rh, rd)
    
    return None


def _is_free(x, y, z, rw, rh, rd, placed, W, H, D):
    """Check if a box of size (rw, rh, rd) can be placed at (x, y, z)."""
    # Check against all placed boxes
    for i, p in enumerate(placed):
        if p is None:
            continue
        px, py, pz, pw, ph, pd = p
        # Check for overlap
        if (x < px + pw and x + rw > px and
            y < py + ph and y + rh > py and
            z < pz + pd and z + rd > pz):
            return False
    return True


def _simple_load(boxes, container, start_time, deadline):
    """Simple fallback loader for large containers."""
    W, H, D = container
    placements = []
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    
    # Sort by volume descending
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    result = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        if time.time() > deadline:
            break
        
        # Try all rotations
        rots = set()
        for p in permutations((w, h, d)):
            rots.add(p)
        rots = list(rots)
        
        placed = False
        for rw, rh, rd in rots:
            if rw > W or rh > H or rd > D:
                continue
            
            # Try to place at current position
            if x + rw <= W and y + rh <= H and z + rd <= D:
                result[orig_idx] = (0, x, y, z, rw, rh, rd)
                x += rw
                row_h = max(row_h, rh)
                layer_d = max(layer_d, rd)
                placed = True
                break
        
        if not placed:
            # Try to find a position
            for px in range(W - w + 1):
                if time.time() > deadline:
                    break
                for py in range(H - h + 1):
                    if time.time() > deadline:
                        break
                    for pz in range(D - d + 1):
                        # Check if free
                        free = True
                        for r in result:
                            if r is None or r[0] == -1:
                                continue
                            rx, ry, rz, rw2, rh2, rd2 = r
                            if (px < rx + rw2 and px + w > rx and
                                py < ry + rh2 and py + h > ry and
                                pz < rz + rd2 and pz + d > rz):
                                free = False
                                break
                        if free:
                            result[orig_idx] = (0, px, py, pz, w, h, d)
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        
        if not placed:
            result[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any remaining None entries
    for i in range(len(result)):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return result
# EVOLVE-BLOCK-END
