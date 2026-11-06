import math

def construct():
    # Initial grid 4x5
    points = []
    for i in range(4):
        for j in range(5):
            points.append((i, j))
    
    # Repair to remove 4-collinearities
    # We iterate and if a point is part of a 4-collinear set, we move it.
    # To be deterministic, we always move to the first valid spot in a spiral.
    
    def is_valid(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        # Check if 4 points are collinear
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Cross product check
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                        # and (x2-x1)(y4-y1) - (y2-y1)(x4-x1) == 0
                        if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1) and (x2-x1)*(y4-y1) == (y2-y1)*(x4-x1):
                            return False
        return True

    # This O(N^4) check is too slow if called repeatedly in a loop for 20 points?
    # 20^4 = 160,000 checks. Each check is cheap. 160k ops is fine.
    # But we need to find *which* point to move.
    
    # Better: Count collinearities for each point.
    # If a point is in a 4-collinear line, move it.
    
    def count_4_collinear_involving(points, idx):
        count = 0
        p_idx = points[idx]
        for i in range(len(points)):
            if i == idx: continue
            for j in range(i+1, len(points)):
                if j == idx: continue
                for k in range(j+1, len(points)):
                    if k == idx: continue
                    # Check if p_idx, p_i, p_j, p_k are collinear
                    p1 = p_idx
                    p2 = points[i]
                    p3 = points[j]
                    p4 = points[k]
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1) and (x2-x1)*(y4-y1) == (y2-y1)*(x4-x1):
                        count += 1
        return count

    # Repair loop
    changed = True
    while changed:
        changed = False
        for idx in range(len(points)):
            if count_4_collinear_involving(points, idx) > 0:
                # Move points[idx] to a new location
                old_p = points[idx]
                # Try nearby points
                found = False
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0: continue
                        new_p = (old_p[0] + dx, old_p[1] + dy)
                        if new_p in points: continue
                        if new_p[0] < -100 or new_p[0] > 100: continue
                        if new_p[1] < -100 or new_p[1] > 100: continue
                        points[idx] = new_p
                        if count_4_collinear_involving(points, idx) == 0:
                            # Check if this created a new 4-collinear with OTHERS?
                            # The function count_4_collinear_involving only checks lines through idx.
                            # Moving idx might break a line through idx, but could it create one?
                            # No, moving a point away from a line removes it.
                            # But it might land on a new line with 3 others?
                            # We need to ensure the NEW position doesn't form a 4-collinear.
                            # The check above ensures the new position does not form a 4-collinear *with the other 3 points of the old line*.
                            # But it might form a 4-collinear with a *different* set of 3 points.
                            # So we must check if the new point is in ANY 4-collinear.
                            if is_valid(points):
                                found = True
                                changed = True
                                break
                    if found: break
                if not found:
                    # Fallback: random-ish deterministic move
                    points[idx] = (old_p[0] + 1, old_p[1])
                    changed = True
        # Ensure distinctness
        # The move logic ensures new_p not in points, so distinctness is maintained.
        
    return [[p[0], p[1]] for p in points]
