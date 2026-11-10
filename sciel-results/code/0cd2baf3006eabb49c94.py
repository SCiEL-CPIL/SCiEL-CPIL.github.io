"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    N = 30
    MIN_COORD = -30
    MAX_COORD = 30
    TIME_LIMIT = 55.0  # Leave some buffer

    def calculate_areas(pts):
        """Calculate all triangle areas for the given points."""
        areas = []
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i + 1, N):
                x2, y2 = pts[j]
                for k in range(j + 1, N):
                    x3, y3 = pts[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area > 0:
                        areas.append(area)
        return areas

    def get_max_multiplicity(areas):
        if not areas:
            return 0, 0
        counts = {}
        for a in areas:
            counts[a] = counts.get(a, 0) + 1
        max_val = max(counts.values())
        # Find one area that has this max count
        best_area = [a for a, c in counts.items() if c == max_val][0]
        return max_val, best_area

    def local_search(pts):
        start_time = time.time()
        # Initial state
        areas = calculate_areas(pts)
        current_max, current_best_area = get_max_multiplicity(areas)
        
        # Precompute pairs for fast area calculation? 
        # Area(i, j, k) = 0.5 * |det(P_i - P_k, P_j - P_k)|
        # We will update counts dynamically.
        
        # We need to be able to quickly get the areas of triangles involving point i.
        # Let's maintain a list of areas? No, we maintain a dict `area_counts`.
        # And we need to know which triangles are "active".
        
        # Actually, let's just recalculate the counts for the affected triangles.
        # When point i moves, triangles (i, j, k) for all j,k != i change.
        
        area_counts = {}
        for a in areas:
            area_counts[a] = area_counts.get(a, 0) + 1
            
        best_score = current_max
        
        while time.time() - start_time < TIME_LIMIT:
            improved = False
            # Try to move each point
            for i in range(N):
                old_x, old_y = pts[i]
                # Try a few random nearby positions
                # Also try positions on the current "best area" grid? 
                # Simple random walk in a small box
                for _ in range(10):
                    new_x = old_x + random.randint(-3, 3)
                    new_y = old_y + random.randint(-3, 3)
                    
                    if new_x < MIN_COORD or new_x > MAX_COORD or new_y < MIN_COORD or new_y > MAX_COORD:
                        continue
                    if (new_x, new_y) in pts: # Must be distinct
                        continue
                        
                    # Calculate new areas for triangles involving i
                    # Decrement old counts
                    for j in range(N):
                        if j == i: continue
                        for k in range(j+1, N):
                            if k == i: continue
                            # Area (i, j, k)
                            x1, y1 = old_x, old_y
                            x2, y2 = pts[j]
                            x3, y3 = pts[k]
                            old_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                            if old_area > 0:
                                area_counts[old_area] -= 1
                                if area_counts[old_area] == 0:
                                    del area_counts[old_area]
                            
                            # Area (i', j, k)
                            x1, y1 = new_x, new_y
                            new_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                            if new_area > 0:
                                area_counts[new_area] = area_counts.get(new_area, 0) + 1
                            
                    # Get new max
                    if area_counts:
                        new_max = max(area_counts.values())
                    else:
                        new_max = 0
                    
                    if new_max > best_score:
                        # Accept move
                        pts[i] = (new_x, new_y)
                        best_score = new_max
                        improved = True
                        break # Move to next point? Or restart? Restart is safer for consistency
                    else:
                        # Revert counts
                        # We need to revert the changes we just made to area_counts
                        # This is expensive if we do it by iterating again.
                        # Better: calculate the delta and apply/revert?
                        # Or simply: if we don't accept, we must undo the updates.
                        # Let's undo by iterating again.
                        for j in range(N):
                            if j == i: continue
                            for k in range(j+1, N):
                                if k == i: continue
                                x1, y1 = new_x, new_y
                                x2, y2 = pts[j]
                                x3, y3 = pts[k]
                                new_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                                if new_area > 0:
                                    area_counts[new_area] -= 1
                                    if area_counts[new_area] == 0:
                                        del area_counts[new_area]
                                        
                                x1, y1 = old_x, old_y
                                old_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                                if old_area > 0:
                                    area_counts[old_area] = area_counts.get(old_area, 0) + 1
                        # Continue trying other positions for this point?
                        # Or skip? Let's skip to save time if not improving.
                        pass
                if improved:
                    break
            
            if not improved:
                # Stuck in local optimum. Try a random restart or larger jump?
                # Let's try a random restart with a few random points perturbed.
                for _ in range(5):
                    idx = random.randint(0, N-1)
                    old_x, old_y = pts[idx]
                    new_x = random.randint(MIN_COORD, MAX_COORD)
                    new_y = random.randint(MIN_COORD, MAX_COORD)
                    if (new_x, new_y) not in pts:
                        # Update counts
                        for j in range(N):
                            if j == idx: continue
                            for k in range(j+1, N):
                                if k == idx: continue
                                x1, y1 = old_x, old_y
                                x2, y2 = pts[j]
                                x3, y3 = pts[k]
                                old_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                                if old_area > 0:
                                    area_counts[old_area] -= 1
                                    if area_counts[old_area] == 0:
                                        del area_counts[old_area]
                                    
                                x1, y1 = new_x, new_y
                                new_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                                if new_area > 0:
                                    area_counts[new_area] = area_counts.get(new_area, 0) + 1
                        pts[idx] = (new_x, new_y)
                        break
        
        return pts, best_score

    # Initialize with a grid
    # 5x6 grid centered
    rows, cols = 5, 6
    shift_x = -(cols - 1) // 2
    shift_y = -(rows - 1) // 2
    pts = []
    for x in range(cols):
        for y in range(rows):
            pts.append((x + shift_x, y + shift_y))
    
    best_pts, best_score = local_search(pts)
    
    return best_pts
# EVOLVE-BLOCK-END
