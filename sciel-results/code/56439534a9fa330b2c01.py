"""Improved 3D bin packing with best-fit placement and deadline guard."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _make_grid(W, H, D, res=4):
    """Create a downsampled occupancy grid for fast collision checks.
    Each cell represents a res x res x res block of the container.
    """
    gw = max(1, (W + res - 1) // res)
    gh = max(1, (H + res - 1) // res)
    gd = max(1, (D + res - 1) // res)
    return np.zeros((gw, gh, gd), dtype=np.uint8), (gw, gh, gd, res)

def _mark_grid(grid, shape, pos, W, H, D, res):
    """Mark a box as occupied in the grid."""
    x, y, z = pos
    w, h, d = shape
    x1, y1, z1 = x // res, y // res, z // res
    x2 = (x + w - 1) // res
    y2 = (y + h - 1) // res
    z2 = (z + d - 1) // res
    grid[x1:x2+1, y1:y2+1, z1:z2+1] = 1

def _fits_grid(grid, shape, pos, W, H, D, res):
    """Check if a box fits at position using the grid (fast approximate check)."""
    x, y, z = pos
    w, h, d = shape
    if x + w > W or y + h > H or z + d > D:
        return False
    x1, y1, z1 = x // res, y // res, z // res
    x2 = (x + w - 1) // res
    y2 = (y + h - 1) // res
    z2 = (z + d - 1) // res
    # Check if any cell in the range is occupied
    if x2 < 0 or y2 < 0 or z2 < 0:
        return False
    return not grid[x1:x2+1, y1:y2+1, z1:z2+1].any()

def _exact_fits(placed, shape, pos, W, H, D):
    """Exact overlap check against all placed boxes in a bin."""
    x, y, z = pos
    w, h, d = shape
    if x + w > W or y + h > H or z + d > D:
        return False
    for px, py, pz, pw, ph, pd in placed:
        # Check overlap in all 3 axes
        if not (x + w <= px or px + pw <= x):
            if not (y + h <= py or py + ph <= y):
                if not (z + d <= pz or pz + pd <= z):
                    return False
    return True

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    # Deadline guard: leave safety margin (aim to finish well before 120s)
    deadline = time.time() + 110.0
    
    # Sort boxes by volume descending for better packing
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # State: list of bins, each bin is a list of (x, y, z, w, h, d)
    bins = []  # list of lists of placed boxes
    grids = []  # corresponding grids
    grid_shapes = None
    
    # Placement result: for each original box index, store (bin_id, x, y, z, w, h, d)
    result = [None] * n
    
    # Try to place each box (in sorted order) into existing bins or a new bin
    for idx in order:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple shelf manner in a new bin
            # This ensures we always return a valid result
            for rem_idx in order[order.index(idx):]:
                # Just dump into a new bin with shelf packing
                if not bins:
                    bins.append([])
                    grids.append(_make_grid(W, H, D, 4)[0])
                    grid_shapes = _make_grid(W, H, D, 4)[1]
                # Try to place in last bin with simple shelf
                placed = bins[-1]
                # Simple: try to find a spot
                found = False
                # Try positions on a grid of candidate spots
                for cx in range(0, W - boxes[rem_idx][0] + 1, max(1, boxes[rem_idx][0]//2)):
                    for cy in range(0, H - boxes[rem_idx][1] + 1, max(1, boxes[rem_idx][1]//2)):
                        for cz in range(0, D - boxes[rem_idx][2] + 1, max(1, boxes[rem_idx][2]//2)):
                            if _exact_fits(placed, boxes[rem_idx], (cx, cy, cz), W, H, D):
                                result[rem_idx] = (len(bins)-1, cx, cy, cz, boxes[rem_idx][0], boxes[rem_idx][1], boxes[rem_idx][2])
                                placed.append((cx, cy, cz, boxes[rem_idx][0], boxes[rem_idx][1], boxes[rem_idx][2]))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    # New bin
                    bins.append([])
                    grids.append(_make_grid(W, H, D, 4)[0])
                    placed = bins[-1]
                    result[rem_idx] = (len(bins)-1, 0, 0, 0, boxes[rem_idx][0], boxes[rem_idx][1], boxes[rem_idx][2])
                    placed.append((0, 0, 0, boxes[rem_idx][0], boxes[rem_idx][1], boxes[rem_idx][2]))
            return result
        
        w, h, d = boxes[idx]
        best_bin = -1
        best_pos = None
        best_score = None
        
        # Try existing bins first (best-fit: prefer bin where the box fits with least waste)
        for b in range(len(bins)):
            if time.time() > deadline:
                break
            placed = bins[b]
            grid = grids[b]
            gw, gh, gd, res = grid_shapes if grid_shapes else _make_grid(W, H, D, 4)[1]
            
            # Generate candidate positions: use grid-based candidates
            # Try positions aligned to grid cells and also at 0
            candidates = set()
            candidates.add((0, 0, 0))
            
            # Add positions based on grid resolution
            for gx in range(0, (W - w + res - 1) // res + 1):
                for gy in range(0, (H - h + res - 1) // res + 1):
                    for gz in range(0, (D - d + res - 1) // res + 1):
                        cx, cy, cz = gx * res, gy * res, gz * res
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
            
            # Also try positions offset by box dimensions from existing boxes
            for px, py, pz, pw, ph, pd in placed:
                for cx in [px + pw, max(0, px - w)]:
                    for cy in [py + ph, max(0, py - h)]:
                        for cz in [pz + pd, max(0, pz - d)]:
                            if cx + w <= W and cy + h <= H and cz + d <= D:
                                candidates.add((cx, cy, cz))
            
            for pos in candidates:
                if not _fits_grid(grid, boxes[idx], pos, W, H, D, res):
                    continue
                if not _exact_fits(placed, boxes[idx], pos, W, H, D):
                    continue
                # Score: prefer lower position (pack from bottom), then left, then back
                # Use a score that favors compactness
                score = pos[2] * 10000 + pos[1] * 100 + pos[0]
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b
                    best_pos = pos
        
        if best_bin >= 0:
            placed = bins[best_bin]
            grid = grids[best_bin]
            gw, gh, gd, res = grid_shapes
            result[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            placed.append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            _mark_grid(grid, boxes[idx], best_pos, W, H, D, res)
        else:
            # Open a new bin
            grid, grid_info = _make_grid(W, H, D, 4)
            if grid_shapes is None:
                grid_shapes = grid_info
            bins.append([(0, 0, 0, w, h, d)])
            grids.append(grid)
            _mark_grid(grid, boxes[idx], (0, 0, 0), W, H, D, grid_info[3])
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
