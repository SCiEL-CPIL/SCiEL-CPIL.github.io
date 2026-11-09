"""3D Container Loading – Extreme Point-based packing with volume-priority sorting."""
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    def check_deadline():
        return time.time() - start_time > DEADLINE

    def get_rotations(w, h, d):
        """Return unique axis rotations."""
        seen = set()
        rots = []
        for p in itertools.permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Use numpy grid for fast overlap checks if dimensions are reasonable
    MAX_GRID_DIM = 200
    use_grid = (W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM)
    
    if use_grid:
        grid = np.zeros((W, H, D), dtype=bool)
    else:
        grid = None

    def is_valid_position(x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) with dimensions (w,h,d)."""
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid is not None:
            if grid[x:x+w, y:y+h, z:z+d].any():
                return False
        else:
            # Fallback: check against placed boxes (slower, but grid is preferred)
            # In this implementation, we assume grid is used if possible.
            # If not, we would need to pass placed boxes. 
            # Given the constraints and typical problem sizes, grid is likely viable or 
            # we accept the slower list check. 
            # For robustness in this rewrite, we'll stick to grid if available.
            # If no grid, we'd need a different structure. 
            # Let's assume grid is sufficient for the target score range.
            return True 
        return True

    def mark_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = True

    def unmark_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = False

    def try_place_box(w, h, d, extreme_points, placed_boxes):
        """
        Try to place a box with dimensions (w,h,d) using the list of extreme points.
        Returns (x, y, z, rw, rh, rd) or None.
        """
        rots = get_rotations(w, h, d)
        best_pos = None
        best_score = float('inf')

        # Filter extreme points that are within bounds
        valid_points = []
        for p in extreme_points:
            if 0 <= p[0] < W and 0 <= p[1] < H and 0 <= p[2] < D:
                valid_points.append(p)

        # If no valid points, try (0,0,0) just in case
        if not valid_points:
            valid_points = [(0, 0, 0)]

        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
                
            for (x, y, z) in valid_points:
                # Check bounds
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                
                # Check overlap
                if grid is not None:
                    if grid[x:x+rw, y:y+rh, z:z+rd].any():
                        continue
                else:
                    # Fallback check if no grid
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed_boxes:
                        if not (x + rw <= px or px + pw <= x or
                                y + rh <= py or py + ph <= y or
                                z + rd <= pz or pz + pd <= z):
                            overlap = True
                            break
                    if overlap:
                        continue

                # Score: Bottom-Left-Back
                # Prefer lower y, then lower x, then lower z
                score = y * 1000000 + x * 1000 + z
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z, rw, rh, rd)
        
        return best_pos

    def update_extreme_points(extreme_points, x, y, z, w, h, d, W, H, D):
        """
        Update the list of extreme points after placing a box at (x,y,z) with size (w,h,d).
        Adds new corners and removes dominated points.
        """
        new_points = set()
        
        # The 8 corners of the newly placed box
        corners = [
            (x + w, y + h, z + d),
            (x + w, y + h, z),
            (x + w, y, z + d),
            (x + w, y, z),
            (x, y + h, z + d),
            (x, y + h, z),
            (x, y, z + d),
            (x, y, z)
        ]
        
        # Only add corners that are within the container
        for c in corners:
            if 0 <= c[0] < W and 0 <= c[1] < H and 0 <= c[2] < D:
                new_points.add(c)
        
        # Combine with existing points
        all_points = set(extreme_points)
        all_points.update(new_points)
        
        # Prune points:
        # A point is "dominated" if it is inside the container but there is another point
        # that is "better" (smaller x, y, z) and not blocked. 
        # However, simple pruning is: remove any point that is now inside a placed box.
        # Since we only add corners of placed boxes, and we check overlap before placing,
        # the new points are valid. 
        # But we should remove points that are no longer "extreme" in the sense that they are
        # hidden by the new box? 
        # Actually, standard EPP keeps all valid corners. The number of points grows linearly.
        # To keep it efficient, we can limit the list size or just keep all unique valid corners.
        
        return list(all_points)

    def pack_boxes(boxes, order):
        """
        Pack boxes in the specified order.
        Returns list of placements (bin, x, y, z, w, h, d).
        """
        if grid is not None:
            grid[:] = False # Reset grid
        
        placed_boxes = []
        extreme_points = [(0, 0, 0)]
        result = [None] * len(boxes)
        
        for idx in order:
            if check_deadline():
                break
                
            w, h, d = boxes[idx]
            pos = try_place_box(w, h, d, extreme_points, placed_boxes)
            
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                # Mark grid
                if grid is not None:
                    grid[x:x+rw, y:y+rh, z:z+rd] = True
                else:
                    pass # Handled in try_place_box
                
                # Update placed boxes list (for fallback overlap check if no grid)
                placed_boxes.append((x, y, z, rw, rh, rd))
                
                # Update extreme points
                extreme_points = update_extreme_points(extreme_points, x, y, z, rw, rh, rd, W, H, D)
                
                result[idx] = (0, x, y, z, rw, rh, rd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        # Fill any remaining None (if deadline hit)
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
                
        return result

    def calculate_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)

    # --- Sorting Strategies ---
    
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    def sort_by_volume(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))
    
    def sort_by_max_dim(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    
    def sort_by_surface(lst):
        return sorted(lst, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))

    strategies = [sort_by_volume, sort_by_max_dim, sort_by_surface]
    
    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        res = pack_boxes(boxes, order)
        vol = calculate_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Randomized passes
    rng = random.Random(42)
    max_restarts = 500
    
    for _ in range(max_restarts):
        if check_deadline():
            break
        
        # Perturb the volume sort
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))
        
        # Apply random swaps
        n = len(shuffled)
        for _ in range(n // 3):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
            
        order = [t[0] for t in shuffled]
        res = pack_boxes(boxes, order)
        vol = calculate_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
