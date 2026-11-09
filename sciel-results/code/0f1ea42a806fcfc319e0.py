# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    def get_rotations(w, h, d):
        rots = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        unique_rots = []
        seen = set()
        for r in rots:
            if r not in seen:
                seen.add(r)
                unique_rots.append(r)
        return unique_rots
    
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: (-box_volumes[i], -max(boxes[i])))
    
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    placed = []
    corners = [(0, 0, 0)]
    
    max_x, max_y, max_z = 0, 0, 0
    
    def can_place(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bi, px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        return True

    def update_corners(x, y, z, w, h, d):
        new_corners = [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
            (x + w, y + h, z),
            (x + w, y, z + d),
            (x, y + h, z + d),
            (x + w, y + h, z + d)
        ]
        
        valid_new = []
        for nx, ny, nz in new_corners:
            if nx <= W and ny <= H and nz <= D:
                valid_new.append((nx, ny, nz))
        
        filtered_corners = []
        for cx, cy, cz in corners:
            if (x <= cx < x + w) and (y <= cy < y + h) and (z <= cz < z + d):
                continue
            filtered_corners.append((cx, cy, cz))
        
        filtered_corners.extend(valid_new)
        
        unique_corners = list(set(filtered_corners))
        return unique_corners

    for box_idx in sorted_indices:
        w0, h0, d0 = boxes[box_idx]
        rots = get_rotations(w0, h0, d0)
        
        # Sort corners by priority: lowest z, then lowest y, then lowest x
        sorted_corners = sorted(corners, key=lambda c: (c[2], c[1], c[0]))
        
        # Limit search to top K candidates to balance quality and speed
        K = 50
        candidates = sorted_corners[:K]
        
        best_score = float('inf')
        best_placement = None
        
        for cx, cy, cz in candidates:
            for rw, rh, rd in rots:
                if can_place(cx, cy, cz, rw, rh, rd):
                    new_max_x = max(max_x, cx + rw)
                    new_max_y = max(max_y, cy + rh)
                    new_max_z = max(max_z, cz + rd)
                    vol = new_max_x * new_max_y * new_max_z
                    
                    current_score = (vol, cz, cy, cx)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_placement = (cx, cy, cz, rw, rh, rd)
        
        if best_placement is not None:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            placements[box_idx] = (0, cx, cy, cz, rw, rh, rd)
            corners = update_corners(cx, cy, cz, rw, rh, rd)
            
            max_x = max(max_x, cx + rw)
            max_y = max(max_y, cy + rh)
            max_z = max(max_z, cz + rd)
            
    return placements
# EVOLVE-BLOCK-END
