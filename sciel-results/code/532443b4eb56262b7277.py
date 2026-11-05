"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending.
2. Use a Best-Fit-Decreasing heuristic with efficient candidate position generation.
3. For each box, find the position that minimizes (z, y, x) - pack towards origin.
4. Post-processing: try to consolidate bins and maximize last-bin fill.
"""
import time
import numpy as np
import sys

class Bin:
    """Represents a bin with efficient occupancy tracking."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.placements = []  # List of (box_idx, x, y, z, w, h, d)
        self.volume = 0
        self._rebuild_grid()
    
    def _rebuild_grid(self):
        """Rebuild the 3D occupancy grid from placements."""
        if not self.placements:
            self.grid = None
            return
        # Use a set of occupied unit cells for overlap checking
        # Only build if dimensions are small enough
        if self.W * self.H * self.D <= 10**7:
            self.grid = np.zeros((self.D, self.H, self.W), dtype=bool)
            for (_, x, y, z, w, h, d) in self.placements:
                self.grid[z:z+d, y:y+h, x:x+w] = True
        else:
            self.grid = None
    
    def add_placement(self, box_idx, x, y, z, w, h, d):
        """Add a placement to this bin."""
        self.placements.append((box_idx, x, y, z, w, h, d))
        self.volume += w * h * d
        if self.grid is not None:
            self.grid[z:z+d, y:y+h, x:x+w] = True
    
    def remove_placement(self, box_idx):
        """Remove a placement by box index."""
        for i, (bi, x, y, z, w, h, d) in enumerate(self.placements):
            if bi == box_idx:
                del self.placements[i]
                self.volume -= w * h * d
                if self.grid is not None:
                    self.grid[z:z+d, y:y+h, x:x+w] = False
                return True
        return False
    
    def find_position(self, w, h, d):
        """
        Find the best position for a box (w, h, d) in this bin.
        Returns (x, y, z) or None if no position fits.
        Uses candidate positions from box boundaries and checks via grid or brute force.
        """
        if w > self.W or h > self.H or d > self.D:
            return None
        
        # Generate candidate coordinates from box boundaries
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (_, px, py, pz, pw, ph, pd) in self.placements:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        
        # Filter candidates that fit within container
        valid_xs = sorted([x for x in xs if x + w <= self.W])
        valid_ys = sorted([y for y in ys if y + h <= self.H])
        valid_zs = sorted([z for z in zs if z + d <= self.D])
        
        if not valid_xs or not valid_ys or not valid_zs:
            return None
        
        # Check candidates in order of (z, y, x) - pack towards origin
        for z in valid_zs:
            for y in valid_ys:
                for x in valid_xs:
                    if self._can_place(x, y, z, w, h, d):
                        return (x, y, z)
        
        return None
    
    def _can_place(self, x, y, z, w, h, d):
        """Check if a box of size (w, h, d) can be placed at (x, y, z)."""
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
        
        if self.grid is not None:
            # Fast check using grid
            if not self.grid[z:z+d, y:y+h, x:x+w].any():
                return True
            return False
        
        # Slow check: brute force overlap detection
        for (_, px, py, pz, pw, ph, pd) in self.placements:
            if x < px + pw and px < x + w and \
               y < py + ph and py < y + h and \
               z < pz + pd and pz < z + d:
                return False
        return True


def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (150s limit)
    
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    bins = []  # List of Bin objects
    result = [None] * n  # result[i] = (bin_id, x, y, z, w, h, d)
    
    # 2. Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency placement: each in new bin
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    b = Bin(W, H, D)
                    b.add_placement(oi, 0, 0, 0, w, h, d)
                    bins.append(b)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Try existing bins, sorted by current occupied volume (descending)
            # This is "best fit" - prefer the fullest bin that can accommodate the box
            bin_indices = sorted(range(len(bins)), key=lambda bi: -bins[bi].volume)
            
            for bi in bin_indices:
                if time.time() > deadline - 3.0:
                    break
                pos = bins[bi].find_position(w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].add_placement(orig_idx, x, y, z, w, h, d)
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            b = Bin(W, H, D)
            b.add_placement(orig_idx, 0, 0, 0, w, h, d)
            bins.append(b)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
            
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            b = Bin(W, H, D)
            b.add_placement(i, 0, 0, 0, w, h, d)
            bins.append(b)
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    # 3. Phase 2: Try to consolidate bins
    if time.time() < deadline - 10.0:
        # Try to move boxes from higher-indexed bins to lower-indexed bins
        # to reduce the total number of bins
        for src_bin in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if not bins[src_bin].placements:
                continue
            
            # Try to move each box from src_bin to an earlier bin
            boxes_to_try = list(bins[src_bin].placements)
            for (box_idx, x, y, z, w, h, d) in boxes_to_try:
                if time.time() > deadline - 5.0:
                    break
                
                # Remove from source bin
                bins[src_bin].remove_placement(box_idx)
                
                # Try to place in earlier bins
                moved = False
                for dst_bin in range(src_bin):
                    if time.time() > deadline - 5.0:
                        break
                    pos = bins[dst_bin].find_position(w, h, d)
                    if pos is not None:
                        nx, ny, nz = pos
                        bins[dst_bin].add_placement(box_idx, nx, ny, nz, w, h, d)
                        result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                        moved = True
                        break
                
                if not moved:
                    # Restore
                    bins[src_bin].add_placement(box_idx, x, y, z, w, h, d)
                    result[box_idx] = (src_bin, x, y, z, w, h, d)
            
            # If bin is now empty, it will be removed in relabeling
            
    # 4. Phase 3: Optimize last bin fill
    if time.time() < deadline - 5.0:
        # Identify last bin (highest label)
        occupied_bins = sorted(set(r[0] for r in result if r is not None))
        if occupied_bins:
            last_bin_id = occupied_bins[-1]
            
            moved_any = True
            while moved_any and time.time() < deadline - 2.0:
                moved_any = False
                candidates = [i for i in range(n) if result[i] is not None and result[i][0] != last_bin_id]
                for idx in candidates:
                    if time.time() > deadline - 2.0:
                        break
                    
                    cur_bin = result[idx][0]
                    w, h, d = boxes[idx]
                    x, y, z = result[idx][1], result[idx][2], result[idx][3]
                    
                    # Remove from source bin
                    bins[cur_bin].remove_placement(idx)
                    
                    # Try to place in last bin
                    pos = bins[last_bin_id].find_position(w, h, d)
                    
                    if pos is not None:
                        # Move it
                        nx, ny, nz = pos
                        bins[last_bin_id].add_placement(idx, nx, ny, nz, w, h, d)
                        result[idx] = (last_bin_id, nx, ny, nz, w, h, d)
                        moved_any = True
                        break
                    else:
                        # Restore
                        bins[cur_bin].add_placement(idx, x, y, z, w, h, d)
                        result[idx] = (cur_bin, x, y, z, w, h, d)
    
    # 5. Relabel bins to be consecutive
    occupied_bins = sorted(set(r[0] for r in result if r is not None))
    label_map = {old: new for new, old in enumerate(occupied_bins)}
    
    final_result = []
    for i in range(n):
        if result[i] is not None:
            old_bin = result[i][0]
            new_bin = label_map.get(old_bin, 0)
            final_result.append((new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6]))
        else:
            # Should not happen, but safety fallback
            w, h, d = boxes[i]
            final_result.append((0, 0, 0, 0, w, h, d))
    
    return final_result
