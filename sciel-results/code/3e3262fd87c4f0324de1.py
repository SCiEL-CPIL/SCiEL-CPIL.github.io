"""3D bin packing: best-fit decreasing with grid-accelerated placement and consolidation."""
import time
import numpy as np

DEADLINE = 130.0  # wall-clock seconds; evaluator kills at 150s, keep margin


def _build_grid(W, H, D):
    """Create a boolean occupancy grid for the container."""
    return np.zeros((W, H, D), dtype=bool)


def _can_place(grid, x, y, z, w, h, d):
    """Check if a box of size (w,h,d) can be placed at (x,y,z) in the grid."""
    if x + w > grid.shape[0] or y + h > grid.shape[1] or z + d > grid.shape[2]:
        return False
    return not np.any(grid[x:x+w, y:y+h, z:z+d])


def _place(grid, x, y, z, w, h, d):
    """Place a box of size (w,h,d) at (x,y,z) in the grid."""
    grid[x:x+w, y:y+h, z:z+d] = True


def _remove(grid, x, y, z, w, h, d):
    """Remove a box of size (w,h,d) at (x,y,z) from the grid."""
    grid[x:x+w, y:y+h, z:z+d] = False


def _find_best_position(grid, W, H, D, w, h, d):
    """Find the position that minimizes wasted space (closest to origin corner).
    
    Uses a scoring function: prefer positions with smaller x, y, z (towards origin)
    and smaller gaps to walls. Returns (x, y, z) or None.
    """
    best_pos = None
    best_score = None
    
    # Candidate x positions: 0 and (existing box ends)
    # For efficiency, we scan all valid x, y, z but use the grid for O(1) checks
    # Actually, let's be smarter: generate candidate coordinates from occupied cells
    
    # Get all unique x, y, z boundaries from occupied cells
    # This is expensive to compute each time, so let's just scan a subset
    # For small containers, full scan is fine. For larger, use candidate coords.
    
    # Strategy: try positions near existing boxes and at origin
    # First try (0,0,0)
    candidates = [(0, 0, 0)]
    
    # Add positions adjacent to existing occupied regions
    # Find min/max of occupied cells for bounds
    if grid.any():
        occ = np.argwhere(grid)
        if len(occ) > 0:
            # Sample some boundary positions
            # For each occupied cell, the cell to the right, above, and in front
            # is a candidate start position
            seen = set()
            for cx, cy, cz in occ:
                for dx, dy, dz in [(w, 0, 0), (0, h, 0), (0, 0, d), 
                                   (w, h, 0), (w, 0, d), (0, h, d), (w, h, d)]:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if nx < W and ny < H and nz < D and (nx, ny, nz) not in seen:
                        seen.add((nx, ny, nz))
                        candidates.append((nx, ny, nz))
            # Also try positions just past each dimension
            for cx, cy, cz in occ:
                for nx, ny, nz in [(cx + w, cy, cz), (cx, cy + h, cz), (cx, cy, cz + d)]:
                    if nx < W and ny < H and nz < D and (nx, ny, nz) not in seen:
                        seen.add((nx, ny, nz))
                        candidates.append((nx, ny, nz))
    
    # Deduplicate and filter
    unique_candidates = []
    seen = set()
    for x, y, z in candidates:
        if x + w <= W and y + h <= H and z + d <= D and (x, y, z) not in seen:
            seen.add((x, y, z))
            unique_candidates.append((x, y, z))
    
    if not unique_candidates:
        return None
    
    # Score each candidate: prefer positions closer to origin (minimize x+y+z)
    # and minimize wasted volume in the corner
    for x, y, z in unique_candidates:
        if _can_place(grid, x, y, z, w, h, d):
            # Score: lower is better. Use x+y+z as primary, waste as secondary
            waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
            score = (x + y + z, waste)
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (x, y, z)
    
    return best_pos


def _find_any_position(grid, W, H, D, w, h, d):
    """Find any valid position for the box."""
    # Try (0,0,0) first
    if _can_place(grid, 0, 0, 0, w, h, d):
        return (0, 0, 0)
    
    # Systematic scan (limited for performance)
    for x in range(0, W - w + 1):
        for y in range(0, H - h + 1):
            for z in range(0, D - d + 1):
                if _can_place(grid, x, y, z, w, h, d):
                    return (x, y, z)
    return None


def _occupied_volume(items):
    return sum(w * h * d for (_, _, _, w, h, d) in items)


def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    start = time.time()
    n = len(boxes)
    
    # Sort boxes by volume descending
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # bins: list of (grid, items) where items = list of (x, y, z, w, h, d)
    bins = []  # each element: [grid, list_of_items]
    placement = [None] * n  # placement[i] = (bin_idx, x, y, z)
    
    for idx in order:
        w, h, d = boxes[idx]
        best_bin = -1
        best_pos = None
        best_score = None
        
        # Try existing bins
        for bi, (grid, items) in enumerate(bins):
            pos = _find_best_position(grid, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
                score = (x + y + z, waste)
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
        
        if best_bin >= 0:
            grid, items = bins[best_bin]
            _place(grid, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            items.append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placement[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
        else:
            # Open a new bin
            grid = _build_grid(W, H, D)
            pos = _find_best_position(grid, W, H, D, w, h, d)
            if pos is None:
                pos = (0, 0, 0)
            _place(grid, pos[0], pos[1], pos[2], w, h, d)
            bins.append([grid, [(pos[0], pos[1], pos[2], w, h, d)]])
            placement[idx] = (len(bins) - 1, pos[0], pos[1], pos[2])
    
    def build_result():
        res = []
        for i in range(n):
            w, h, d = boxes[i]
            bi, x, y, z = placement[i]
            res.append((bi, x, y, z, w, h, d))
        return res
    
    def score_of(placement):
        occ = set()
        last = -1
        for i in range(n):
            bi, x, y, z = placement[i]
            occ.add(bi)
            if bi > last:
                last = bi
        B = len(occ)
        last_vol = 0
        for i in range(n):
            bi, x, y, z = placement[i]
            if bi == last:
                last_vol += boxes[i][0] * boxes[i][1] * boxes[i][2]
        frac = last_vol / container_vol if container_vol else 0.0
        return B + 0.1 * (1 - frac)
    
    # Consolidation: try to move boxes from higher-indexed bins to lower-indexed bins
    cur_score = score_of(placement)
    
    while time.time() < DEADLINE - 3:
        improved = False
        # Process boxes from highest bin index down
        max_bin = max(bi for bi, _, _ in placement)
        for bi in range(max_bin, 0, -1):
            if time.time() >= DEADLINE - 3:
                break
            # Get all boxes in this bin
            boxes_in_bin = [i for i in range(n) if placement[i][0] == bi]
            if not boxes_in_bin:
                continue
            
            # Try to move each box to a lower bin
            for i in boxes_in_bin:
                if time.time() >= DEADLINE - 3:
                    break
                w, h, d = boxes[i]
                _, x, y, z = placement[i]
                
                # Remove from current bin
                grid_cur, items_cur = bins[bi]
                _remove(grid_cur, x, y, z, w, h, d)
                items_cur = [item for item in items_cur if item != (x, y, z, w, h, d)]
                bins[bi] = [grid_cur, items_cur]
                
                # Try to place in a lower bin
                found = False
                for target_bi in range(bi):
                    grid_t, items_t = bins[target_bi]
                    pos = _find_best_position(grid_t, W, H, D, w, h, d)
                    if pos is not None:
                        _place(grid_t, pos[0], pos[1], pos[2], w, h, d)
                        items_t.append((pos[0], pos[1], pos[2], w, h, d))
                        bins[target_bi] = [grid_t, items_t]
                        placement[i] = (target_bi, pos[0], pos[1], pos[2])
                        found = True
                        improved = True
                        break
                
                if not found:
                    # Put back in original bin
                    _place(grid_cur, x, y, z, w, h, d)
                    items_cur.append((x, y, z, w, h, d))
                    bins[bi] = [grid_cur, items_cur]
                    placement[i] = (bi, x, y, z)
        
        # Remove empty bins (renumber)
        new_bins = []
        bin_map = {}
        for bi in range(len(bins)):
            if bins[bi][1]:  # non-empty
                bin_map[bi] = len(new_bins)
                new_bins.append(bins[bi])
        bins = new_bins
        for i in range(n):
            old_bi, x, y, z = placement[i]
            if old_bi in bin_map:
                placement[i] = (bin_map[old_bi], x, y, z)
        
        new_score = score_of(placement)
        if new_score >= cur_score - 1e-9:
            break
        cur_score = new_score
        if not improved:
            break
    
    return build_result()
