"""3D Bin Packing with Best-Fit decreasing and corner-point heuristics."""
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    class Bin:
        def __init__(self):
            self.placed = []  # list of (x, y, z, w, h, d)
            self.corners = [(0, 0, 0)]  # candidate bottom-left-back corners
        
        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in self.placed:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True
        
        def place(self, x, y, z, w, h, d):
            self.placed.append((x, y, z, w, h, d))
            new_corners = []
            for nx, ny, nz in [(x+w, y, z), (x, y+h, z), (x, y, z+d)]:
                if 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
                    new_corners.append((nx, ny, nz))
            for c in new_corners:
                if c not in self.corners:
                    self.corners.append(c)
        
        def best_corner(self, w, h, d):
            best = None
            best_score = None
            for (cx, cy, cz) in self.corners:
                if self.can_place(cx, cy, cz, w, h, d):
                    score = cx + cy + cz
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (cx, cy, cz)
            return best
        
        def occupied_volume(self):
            return sum(w * h * d for (_, _, _, w, h, d) in self.placed)
    
    bins = []
    placements = []
    
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Switch to fast mode
            break
        
        best_bin = None
        best_pos = None
        best_score = None
        
        for b_idx, b in enumerate(bins):
            pos = b.best_corner(w, h, d)
            if pos is not None:
                cx, cy, cz = pos
                score = cx + cy + cz - 0.001 * b.occupied_volume()
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin is None:
            b = Bin()
            bins.append(b)
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        bx, by, bz = best_pos
        bins[best_bin].place(bx, by, bz, w, h, d)
        placements.append((best_bin, bx, by, bz, w, h, d))
    
    # If we broke early, place remaining boxes with a simple shelf heuristic
    placed_count = len(placements)
    if placed_count < n:
        # Simple shelf for remaining
        shelf_bins = []
        for i in range(placed_count, n):
            orig_idx, w, h, d = indexed_boxes[i]
            if not shelf_bins:
                shelf_bins.append({'x': 0, 'y': 0, 'z': 0, 'row_h': 0, 'layer_d': 0, 'placements': []})
            
            placed = False
            for sb in shelf_bins:
                x, y, z = sb['x'], sb['y'], sb['z']
                if x + w <= W:
                    if y + h <= H:
                        if z + d <= D:
                            sb['placements'].append((len(shelf_bins)-1, x, y, z, w, h, d))
                            sb['x'] = x + w
                            sb['row_h'] = max(sb['row_h'], h)
                            sb['layer_d'] = max(sb['layer_d'], d)
                            placed = True
                            break
                if x + w > W:
                    sb['x'] = 0
                    sb['y'] += sb['row_h']
                    sb['row_h'] = 0
                if sb['y'] + h > H:
                    sb['y'] = 0
                    sb['z'] += sb['layer_d']
                    sb['layer_d'] = 0
                    sb['x'] = 0
                    sb['row_h'] = 0
            if not placed:
                shelf_bins.append({'x': w, 'y': 0, 'z': 0, 'row_h': h, 'layer_d': d, 'placements': [(len(shelf_bins)-1, 0, 0, 0, w, h, d)]})
        
        # Map shelf bin indices to global bin indices
        # The shelf bins are new bins, so their indices start from len(bins)
        for sb_idx, sb in enumerate(shelf_bins):
            global_bin = len(bins) + sb_idx
            for (local_bin, x, y, z, w, h, d) in sb['placements']:
                placements.append((global_bin, x, y, z, w, h, d))
    
    return placements
