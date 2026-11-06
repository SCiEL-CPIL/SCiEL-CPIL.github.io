import time
import sys

sys.setrecursionlimit(2000)

def pack(boxes, container):
    W, H, D = container
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    n = len(boxes)
    best_placements = [None] * n
    best_bins = n + 1
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    def solve(idx, bins, placements_map):
        nonlocal best_placements, best_bins, deadline
        
        if time.time() > deadline:
            return False
        
        if idx == n:
            current_bins = len(bins)
            if current_bins < best_bins:
                best_bins = current_bins
                temp_best = [None] * n
                for s_idx in range(n):
                    o_idx, w, h, d, vol = indexed_boxes[s_idx]
                    b_idx, x, y, z = placements_map[s_idx]
                    temp_best[o_idx] = (b_idx, x, y, z, w, h, d)
                best_placements = temp_best
            return False # Continue searching for better (fewer bins)
            
        w, h, d, vol = indexed_boxes[idx][1], indexed_boxes[idx][2], indexed_boxes[idx][3], indexed_boxes[idx][4]
        
        # Try existing bins
        for b_idx in range(len(bins)):
            spaces = bins[b_idx][:]
            for i, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
                if w <= sw and h <= sh and d <= sd:
                    new_spaces = []
                    for j, s in enumerate(spaces):
                        if j == i:
                            if sw - w > 0:
                                new_spaces.append((sx + w, sy, sz, sw - w, sh, sd))
                            if sh - h > 0:
                                new_spaces.append((sx, sy + h, sz, sw, sh - h, sd))
                            if sd - d > 0:
                                new_spaces.append((sx, sy, sz + d, w, h, sd - d))
                        else:
                            new_spaces.append(s)
                    
                    bins[b_idx] = new_spaces
                    placements_map[idx] = (b_idx, sx, sy, sz)
                    
                    solve(idx + 1, bins, placements_map)
                    
                    bins[b_idx] = spaces
                    del placements_map[idx]
        
        # Try new bin
        if len(bins) < best_bins:
            new_bin_spaces = [(0, 0, 0, W, H, D)]
            bins.append(new_bin_spaces)
            b_idx = len(bins) - 1
            placements_map[idx] = (b_idx, 0, 0, 0)
            
            solve(idx + 1, bins, placements_map)
            
            bins.pop()
            del placements_map[idx]
            
        return False

    solve(0, [], {})
    
    if any(p is None for p in best_placements):
        # Fallback to greedy
        W, H, D = container
        placements_seed = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements_seed.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements_seed

    return best_placements
