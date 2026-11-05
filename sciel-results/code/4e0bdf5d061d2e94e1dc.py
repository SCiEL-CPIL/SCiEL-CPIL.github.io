"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending.
2. Place boxes into bins using a Best-Fit-Decreasing heuristic.
   - For each box, try existing bins (sorted by least remaining volume).
   - Within a bin, find the position with lowest z, then lowest y, then lowest x.
3. Post-processing:
   - Identify the last bin (highest label).
   - Try to move boxes from other bins into the last bin to increase its fill ratio.
   - This improves the score term 0.1 * (1 - fill_ratio) without increasing B.
4. Relabel bins to be consecutive.
"""
import time
import numpy as np

def _can_place(box, pos, placements, W, H, D, grid=None):
    """Check if box with dims (w,h,d) can be placed at (x,y,z) in a bin with given placements."""
    x, y, z = pos
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False

    if grid is not None:
        if bool(grid[x:x+w, y:y+h, z:z+d].any()):
            return False
        return True

    for (_, px, py, pz, pw, ph, pd) in placements:
        # Check overlap
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position(box, placements, W, H, D, grid=None):
    """
    Find the best position for a box in a bin.
    Strategy: Bottom-Left-Far (Lowest Z, then Lowest Y, then Lowest X).
    We generate candidate coordinates from existing box boundaries.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    # Candidate coordinates are 0 and right/top/front faces of existing boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])

    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)

    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])

    if not valid_xs or not valid_ys or not valid_zs:
        return None

    # Fast path using occupancy grid:
    # Build a boolean array of size (W, H, D) marking occupied cells.
    # For each candidate z, y, x, check if the sub-array [x:x+w, y:y+h, z:z+d] is all False.
    # We want the lexicographically smallest (z, y, x).
    # Since we iterate z, y, x in sorted order, the first valid one is the answer.
    if grid is not None:
        # grid is a 3D boolean array of shape (W, H, D)
        # Note: NumPy indexing is [x, y, z]
        for z in valid_zs:
            for y in valid_ys:
                for x in valid_xs:
                    if not grid[x:x+w, y:y+h, z:z+d].any():
                        return (x, y, z)
        return None

    # Fallback: manual check
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D, grid):
                    return (x, y, z)
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
    bin_grids = [] # List of numpy boolean arrays (W, H, D)
    result = [None] * n
    
    def _update_grid(grid, box, pos, add=True):
        x, y, z = pos
        w, h, d = box
        grid[x:x+w, y:y+h, z:z+d] = add

    # 2. Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency placement: each in new bin
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    grid = np.zeros((W, H, D), dtype=bool)
                    _update_grid(grid, ob, (0,0,0))
                    bin_grids.append(grid)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Estimate remaining volume for each bin
            bin_vols = []
            for bi in range(len(bins)):
                # Use grid sum for fast volume check
                vol = np.sum(bin_grids[bi])
                bin_vols.append((W*H*D - vol, bi))
            
            # Sort by remaining volume ascending (tightest bins first)
            bin_vols.sort(key=lambda x: x[0])
            
            for _, bi in bin_vols:
                if time.time() > deadline - 3.0:
                    break
                pos = _find_position(box, bins[bi], W, H, D, bin_grids[bi])
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    _update_grid(bin_grids[bi], box, pos)
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, box, (0,0,0))
            bin_grids.append(grid)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
            
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0,0,0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    # 3. Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 5.0:
        # Identify last bin (highest label)
        last_bin_id = max(r[0] for r in result)
        
        moved_any = True
        while moved_any and time.time() < deadline - 2.0:
            moved_any = False
            candidates = [i for i in range(n) if result[i][0] != last_bin_id]
            for idx in candidates:
                if time.time() > deadline - 2.0:
                    break
                
                cur_bin = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                cur_placements = bins[cur_bin]
                box_pos = result[idx][1:4]
                box_idx_in_bin = None
                for j, p in enumerate(cur_placements):
                    if p[1] == box_pos[0] and p[2] == box_pos[1] and p[3] == box_pos[2] and p[4]==w and p[5]==h and p[6]==d:
                        box_idx_in_bin = j
                        break
                
                if box_idx_in_bin is None:
                    continue
                
                # Temporarily remove from source bin
                _update_grid(bin_grids[cur_bin], box, box_pos, add=False)
                cur_placements_no_box = cur_placements[:box_idx_in_bin] + cur_placements[box_idx_in_bin+1:]
                
                # Try to place in last bin
                last_placements = bins[last_bin_id]
                pos = _find_position(box, last_placements, W, H, D, bin_grids[last_bin_id])
                
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    _update_grid(bin_grids[last_bin_id], box, pos)
                    bins[cur_bin] = cur_placements_no_box
                    result[idx] = (last_bin_id, x, y, z, w, h, d)
                    moved_any = True
                    break
                else:
                    # Restore if it didn't fit
                    _update_grid(bin_grids[cur_bin], box, box_pos, add=True)
                    bins[cur_bin] = cur_placements
                    
    # 4. Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
            
    return result
