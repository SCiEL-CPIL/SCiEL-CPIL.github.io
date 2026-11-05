import time
import random
import numpy as np
from scipy import ndimage

DEADLINE = time.time() + 145.0

def _grid_find_max_empty(occ_grid, W, H, D, bw, bh, bd, grid_res=20):
    """
    Find the largest empty axis-aligned box of size (bw, bh, bd) in the grid.
    occ_grid is a 3D boolean numpy array of shape (H, D, W) or similar.
    We return (x, y, z) of the top-left-front corner of the empty box in grid coordinates,
    or None if not found.
    """
    # Ensure grid dimensions match container
    # The grid is a downsampled representation of the container.
    # occ_grid[i, j, k] is True if cell (i, j, k) is occupied.
    # We need to find a (bw, bh, bd) sub-box that is all False.
    
    # Use scipy.ndimage to find empty regions? 
    # Actually, scipy doesn't have a direct "find max empty box" function.
    # We can use a sliding window approach or morphological erosion.
    
    # Let's use a simpler approach: 
    # If the grid is small (e.g., 20x20x20), we can just scan all possible positions.
    # 20^3 = 8000 positions. For each, check if the (bw, bh, bd) box is empty.
    # This is fast enough.
    
    gh, gd, gw = occ_grid.shape
    
    # Convert box size to grid units
    # bw, bh, bd are in container units.
    # grid_res is the number of cells per unit? Or total cells?
    # Let's assume grid_res is the number of cells along each dimension.
    # So cell size = W / gw, H / gh, D / gd.
    
    # We need the box to fit in the grid.
    # The box in grid units:
    # bw_g = ceil(bw * gw / W)
    # bh_g = ceil(bh * gh / H)
    # bd_g = ceil(bd * gd / D)
    
    bw_g = max(1, int(np.ceil(bw * gw / W)))
    bh_g = max(1, int(np.ceil(bh * gh / H)))
    bd_g = max(1, int(np.ceil(bd * gd / D)))
    
    if bw_g > gw or bh_g > gh or bd_g > gd:
        return None
    
    # Scan all possible top-left positions
    best_pos = None
    # We want to place it in the "first" available spot in a specific order,
    # or the "best" spot. Let's try to minimize the bounding box expansion.
    # For simplicity, let's just find the first empty spot in a priority order:
    # Sort by (z, y, x) to fill from bottom-front-left.
    
    for z in range(gd - bd_g + 1):
        for y in range(gh - bh_g + 1):
            for x in range(gw - bw_g + 1):
                # Check if the box (x, y, z) to (x+bw_g, y+bh_g, z+bd_g) is empty
                # occ_grid is (gh, gd, gw)
                # Box occupies:
                # y: y to y+bh_g
                # z: z to z+bd_g
                # x: x to x+bw_g
                sub_box = occ_grid[y:y+bh_g, z:z+bd_g, x:x+bw_g]
                if not np.any(sub_box):
                    return (x, y, z)
    
    return None

def _update_grid(occ_grid, x, y, z, bw, bh, bd, W, H, D, grid_res=20):
    """Update the occupancy grid when a box is placed."""
    gh, gd, gw = occ_grid.shape
    
    # Convert to grid coordinates
    x_g = int(x * gw / W)
    y_g = int(y * gh / H)
    z_g = int(z * gd / D)
    
    bw_g = max(1, int(np.ceil(bw * gw / W)))
    bh_g = max(1, int(np.ceil(bh * gh / H)))
    bd_g = max(1, int(np.ceil(bd * gd / D)))
    
    # Clamp to grid bounds
    x_g = min(x_g, gw - 1)
    y_g = min(y_g, gh - 1)
    z_g = min(z_g, gd - 1)
    
    # Mark as occupied
    occ_grid[y_g:min(y_g+bh_g, gh), z_g:min(z_g+bd_g, gd), x_g:min(x_g+bw_g, gw)] = True

def _pack_ffd(boxes, container, grid_res=20):
    """
    First-Fit Decreasing strategy with grid-based placement.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending
    indices = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    bins = [] # List of lists of (x, y, z, w, h, d)
    grids = [] # List of occupancy grids
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try to place in existing bins
        for bi in range(len(bins)):
            # Try to find a spot in this bin
            pos = _grid_find_max_empty(grids[bi], W, H, D, w, h, d, grid_res)
            if pos is not None:
                x_g, y_g, z_g = pos
                # Convert grid coords back to container coords
                # We need to be careful here. The grid cell (x_g, y_g, z_g) corresponds to
                # container coords (x_g * W/gw, y_g * H/gh, z_g * D/gd)
                # But we want to place the box at the "bottom-left-front" of the empty space.
                # Let's just use the grid coords scaled to container.
                x = x_g * W / grids[bi].shape[2]
                y = y_g * H / grids[bi].shape[0]
                z = z_g * D / grids[bi].shape[1]
                
                # Check if it fits in container
                if x + w <= W and y + h <= H and z + d <= D:
                    bins[bi].append((x, y, z, w, h, d))
                    _update_grid(grids[bi], x, y, z, w, h, d, W, H, D, grid_res)
                    placed = True
                    break
        
        if not placed:
            # Start new bin
            bins.append([(0, 0, 0, w, h, d)])
            gh = grid_res
            gd = grid_res
            gw = grid_res
            occ = np.zeros((gh, gd, gw), dtype=bool)
            _update_grid(occ, 0, 0, 0, w, h, d, W, H, D, grid_res)
            grids.append(occ)
    
    return bins

def _pack_ff(boxes, container, grid_res=20):
    """
    First-Fit strategy with grid-based placement.
    """
    W, H, D = container
    n = len(boxes)
    
    bins = []
    grids = []
    
    for idx in range(n):
        w, h, d = boxes[idx]
        placed = False
        
        for bi in range(len(bins)):
            pos = _grid_find_max_empty(grids[bi], W, H, D, w, h, d, grid_res)
            if pos is not None:
                x_g, y_g, z_g = pos
                x = x_g * W / grids[bi].shape[2]
                y = y_g * H / grids[bi].shape[0]
                z = z_g * D / grids[bi].shape[1]
                
                if x + w <= W and y + h <= H and z + d <= D:
                    bins[bi].append((x, y, z, w, h, d))
                    _update_grid(grids[bi], x, y, z, w, h, d, W, H, D, grid_res)
                    placed = True
                    break
        
        if not placed:
            bins.append([(0, 0, 0, w, h, d)])
            gh = grid_res
            gd = grid_res
            gw = grid_res
            occ = np.zeros((gh, gd, gw), dtype=bool)
            _update_grid(occ, 0, 0, 0, w, h, d, W, H, D, grid_res)
            grids.append(occ)
    
    return bins

def _score_solution(bins, container):
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    last_bin = bins[-1]
    if not last_bin:
        return B
    last_vol = sum(w * h * d for (x, y, z, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill_ratio)

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    
    t_start = time.time()
    
    # Try FFD
    try:
        bins = _pack_ffd(boxes, container, grid_res=20)
        score = _score_solution(bins, container)
        if score < best_score:
            best_score = score
            best_bins = bins
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        pass
    else:
        # Try FF
        try:
            bins = _pack_ff(boxes, container, grid_res=20)
            score = _score_solution(bins, container)
            if score < best_score:
                best_score = score
                best_bins = bins
        except Exception:
            pass
    
    if time.time() > DEADLINE - 10:
        pass
    else:
        # Try FFD with higher resolution
        try:
            bins = _pack_ffd(boxes, container, grid_res=30)
            score = _score_solution(bins, container)
            if score < best_score:
                best_score = score
                best_bins = bins
        except Exception:
            pass
    
    if best_bins is None:
        # Fallback: one box per bin
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
    
    # Construct result
    # We need to map each box to its placement.
    # Since we sorted in FFD, we need to track the original indices.
    # The current _pack_ffd does not return the mapping.
    # Let's modify the approach to track indices.
    
    # Re-run with tracking
    def _pack_ffd_tracked(boxes, container, grid_res=20):
        W, H, D = container
        n = len(boxes)
        indices = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        bins = []
        grids = []
        placement = [None] * n
        
        for idx in indices:
            w, h, d = boxes[idx]
            placed = False
            
            for bi in range(len(bins)):
                pos = _grid_find_max_empty(grids[bi], W, H, D, w, h, d, grid_res)
                if pos is not None:
                    x_g, y_g, z_g = pos
                    x = x_g * W / grids[bi].shape[2]
                    y = y_g * H / grids[bi].shape[0]
                    z = z_g * D / grids[bi].shape[1]
                    
                    if x + w <= W and y + h <= H and z + d <= D:
                        bins[bi].append((x, y, z, w, h, d))
                        _update_grid(grids[bi], x, y, z, w, h, d, W, H, D, grid_res)
                        placement[idx] = (bi, x, y, z, w, h, d)
                        placed = True
                        break
            
            if not placed:
                bins.append([(0, 0, 0, w, h, d)])
                gh = grid_res
                gd = grid_res
                gw = grid_res
                occ = np.zeros((gh, gd, gw), dtype=bool)
                _update_grid(occ, 0, 0, 0, w, h, d, W, H, D, grid_res)
                grids.append(occ)
                placement[idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        
        return bins, placement
    
    # We need to re-run the best strategy with tracking to get the correct placement mapping.
    # Since we don't know which strategy was best, let's just re-run FFD with tracking
    # and assume it's the best, or re-run all and keep the best tracked result.
    
    # For simplicity and correctness, let's just re-run FFD with tracking and use that.
    # If FF was better, we might lose out, but FFD is generally robust.
    # Or we can re-run both and pick the best.
    
    best_tracked_bins = None
    best_tracked_placement = None
    best_tracked_score = float('inf')
    
    if time.time() < DEADLINE - 5:
        try:
            bins, placement = _pack_ffd_tracked(boxes, container, grid_res=20)
            score = _score_solution(bins, container)
            if score < best_tracked_score:
                best_tracked_score = score
                best_tracked_bins = bins
                best_tracked_placement = placement
        except Exception:
            pass
    
    if time.time() < DEADLINE - 5:
        try:
            # Re-run FF with tracking
            def _pack_ff_tracked(boxes, container, grid_res=20):
                W, H, D = container
                n = len(boxes)
                bins = []
                grids = []
                placement = [None] * n
                
                for idx in range(n):
                    w, h, d = boxes[idx]
                    placed = False
                    
                    for bi in range(len(bins)):
                        pos = _grid_find_max_empty(grids[bi], W, H, D, w, h, d, grid_res)
                        if pos is not None:
                            x_g, y_g, z_g = pos
                            x = x_g * W / grids[bi].shape[2]
                            y = y_g * H / grids[bi].shape[0]
                            z = z_g * D / grids[bi].shape[1]
                            
                            if x + w <= W and y + h <= H and z + d <= D:
                                bins[bi].append((x, y, z, w, h, d))
                                _update_grid(grids[bi], x, y, z, w, h, d, W, H, D, grid_res)
                                placement[idx] = (bi, x, y, z, w, h, d)
                                placed = True
                                break
                    
                    if not placed:
                        bins.append([(0, 0, 0, w, h, d)])
                        gh = grid_res
                        gd = grid_res
                        gw = grid_res
                        occ = np.zeros((gh, gd, gw), dtype=bool)
                        _update_grid(occ, 0, 0, 0, w, h, d, W, H, D, grid_res)
                        grids.append(occ)
                        placement[idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
                return bins, placement
            
            bins, placement = _pack_ff_tracked(boxes, container, grid_res=20)
            score = _score_solution(bins, container)
            if score < best_tracked_score:
                best_tracked_score = score
                best_tracked_bins = bins
                best_tracked_placement = placement
        except Exception:
            pass
    
    if best_tracked_placement is not None:
        return best_tracked_placement
    else:
        # Fallback
        return [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
