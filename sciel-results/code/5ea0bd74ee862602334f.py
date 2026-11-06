"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # Try multiple rotations for each box and pick the best placement
    # Use a simple shelf-based approach with rotation optimization
    
    def try_place(box, x, y, z, W, H, D):
        """Try to place box at position, return (success, new_x, new_y, new_z, w, h, d)"""
        w, h, d = box
        # Try all 6 rotations
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
        
        best_rot = None
        for rw, rh, rd in rotations:
            if x + rw <= W and y + rh <= H and z + rd <= D:
                # Prefer rotations that use less height (for shelf packing)
                score = -rh * 1000 - rd * 100 - rw
                if best_rot is None or score < best_rot[0]:
                    best_rot = (score, rw, rh, rd)
        
        if best_rot is None:
            return None
        return best_rot[1], best_rot[2], best_rot[3]
    
    def shelf_pack(boxes_list, W, H, D, start_time):
        """Shelf-based packing with rotation"""
        placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes_list)
        
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        
        for idx, (orig_idx, w, h, d) in enumerate(boxes_list):
            if time.time() > start_time + 10:  # Per-iteration timeout check
                break
            
            # Try to place in current position
            result = try_place((w, h, d), x, y, z, W, H, D)
            
            if result is None:
                # Try next row
                if y + row_h + max(h, d) <= H:
                    x, y, row_h = 0, y + row_h, 0
                    result = try_place((w, h, d), x, y, z, W, H, D)
            
            if result is None:
                # Try next layer
                if z + layer_d + max(h, d) <= D:
                    x, y, z, row_h, layer_d = 0, 0, z + layer_d, 0, 0
                    result = try_place((w, h, d), x, y, z, W, H, D)
            
            if result is not None:
                rw, rh, rd = result
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                x += rw
                row_h = max(row_h, rh)
                layer_d = max(layer_d, rd)
            # else: leave out
        
        return placements
    
    # First pass: sort by volume descending
    result1 = shelf_pack(indexed_boxes, W, H, D, time.time())
    
    # Second pass: sort by max dimension descending
    indexed_boxes2 = sorted([(i, w, h, d) for i, (w, h, d) in enumerate(boxes)],
                            key=lambda x: max(x[1], x[2], x[3]), reverse=True)
    result2 = shelf_pack(indexed_boxes2, W, H, D, time.time())
    
    # Third pass: sort by min dimension ascending (thin boxes first)
    indexed_boxes3 = sorted([(i, w, h, d) for i, (w, h, d) in enumerate(boxes)],
                            key=lambda x: min(x[1], x[2], x[3]))
    result3 = shelf_pack(indexed_boxes3, W, H, D, time.time())
    
    # Compute filled volume for each result
    def compute_filled(placements, W, H, D):
        total_vol = W * H * D
        filled = 0
        for bin_id, x, y, z, w, h, d in placements:
            if bin_id == 0:
                filled += w * h * d
        return filled / total_vol if total_vol > 0 else 0
    
    filled1 = compute_filled(result1, W, H, D)
    filled2 = compute_filled(result2, W, H, D)
    filled3 = compute_filled(result3, W, H, D)
    
    # Pick the best result
    best = max([(filled1, result1), (filled2, result2), (filled3, result3)], key=lambda x: x[0])
    return best[1]
# EVOLVE-BLOCK-END
