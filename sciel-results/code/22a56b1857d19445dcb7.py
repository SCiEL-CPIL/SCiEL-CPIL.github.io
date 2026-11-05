import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    container_vol = W * H * D
    if container_vol <= 0:
        # Degenerate container; place everything in bin 0 at origin (may overlap but no valid placement exists)
        return [(0, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    # Precompute box properties
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    def check_fit(x, y, z, w, h, d, bin_boxes):
        """Check if box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True
    
    def get_candidates_fast(box, bin_boxes):
        """Get candidate positions for a box in a bin. Returns list of (x,y,z)."""
        w, h, d = box
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidates from existing boxes' faces
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Right face
            candidates.add((bx + bw, by, bz))
            # Top face (in y)
            candidates.add((bx, by + bh, bz))
            # Back face (in z)
            candidates.add((bx, by, bz + bd))
            
            # Also try combinations of faces for better packing
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
        
        # Filter valid candidates
        valid = []
        for cx, cy, cz in candidates:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                if check_fit(cx, cy, cz, w, h, d, bin_boxes):
                    valid.append((cx, cy, cz))
        
        return valid
    
    def score_position(pos, box, bin_boxes):
        """Score a position. Lower is better. Uses bottom-fill priority."""
        cx, cy, cz = pos
        w, h, d = box
        
        # Primary: minimize z (bottom fill)
        # Secondary: minimize y (back fill)
        # Tertiary: minimize x (left fill)
        # This encourages compact, bottom-heavy packings
        
        # Compute the "occupied volume" impact - prefer positions that fill gaps
        # Simple heuristic: minimize z, then y, then x
        return (cz, cy, cx)
    
    def run_strategy(sort_key, placement_heuristic):
        """Run a single packing strategy. Returns (score, placements)."""
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=sort_key, reverse=True)
        
        bins = []  # List of lists of (x, y, z, w, h, d)
        placements = [None] * n
        
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            
            # Try to place in existing bins
            best_bin_idx = -1
            best_pos = None
            best_score = None
            
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates_fast(box, bin_boxes)
                if not candidates:
                    continue
                
                # Find best position in this bin
                for pos in candidates:
                    s = score_position(pos, box, bin_boxes)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = pos
                        best_bin_idx = b_idx
            
            if best_bin_idx != -1:
                cx, cy, cz = best_pos
                bins[best_bin_idx].append((cx, cy, cz, w, h, d))
                placements[orig_idx] = (best_bin_idx, cx, cy, cz, w, h, d)
            else:
                # Open new bin
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        
        if not bins:
            return float('inf'), placements
        
        B = len(bins)
        last_bin = bins[-1]
        occupied_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        fill = occupied_vol / container_vol
        strategy_score = B + 0.1 * (1 - fill)
        
        return strategy_score, placements
    
    # Define sorting strategies
    def sort_volume_desc(x):
        return x[1][0]*x[1][1]*x[1][2]
    
    def sort_maxdim_desc(x):
        return max(x[1])
    
    def sort_volume_then_maxdim(x):
        return (x[1][0]*x[1][1]*x[1][2], max(x[1]))
    
    def sort_height_desc(x):
        return x[1][2]
    
    def sort_width_desc(x):
        return x[1][0]
    
    def sort_depth_desc(x):
        return x[1][1]
    
    strategies = [
        (sort_volume_then_maxdim, "bottom_fill"),
        (sort_volume_desc, "bottom_fill"),
        (sort_maxdim_desc, "bottom_fill"),
        (sort_height_desc, "bottom_fill"),
        (sort_width_desc, "bottom_fill"),
        (sort_depth_desc, "bottom_fill"),
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for sort_key, _ in strategies:
        if time.time() > deadline:
            break
        
        s, placements = run_strategy(sort_key, "bottom_fill")
        if s < best_score:
            best_score = s
            best_placements = placements
    
    # If no strategy completed, use a simple fallback
    if best_placements is None:
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=sort_volume_desc, reverse=True)
        bins = []
        placements = [None] * n
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates_fast(box, bin_boxes)
                if candidates:
                    # Pick the position with minimum (z, y, x)
                    best_pos = min(candidates, key=lambda p: (p[2], p[1], p[0]))
                    cx, cy, cz = best_pos
                    bins[b_idx].append((cx, cy, cz, w, h, d))
                    placements[orig_idx] = (b_idx, cx, cy, cz, w, h, d)
                    placed = True
                    break
            if not placed:
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        best_placements = placements
    
    return best_placements
