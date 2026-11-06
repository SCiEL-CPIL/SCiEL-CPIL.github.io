import random
import itertools
import math
import numpy as np
from collections import defaultdict

def construct():
    """
    Find 16 distinct integer points with |x|, |y| <= 1000, no five collinear,
    maximizing the number of lines containing exactly four of the points.
    
    Uses an efficient simulated annealing approach with incremental line count updates.
    """
    random.seed(42)
    np.random.seed(42)
    
    N = 16
    MAX_COORD = 1000
    
    # Helper to get canonical line key
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        # a*x + b*y + c = 0
        # dy_n * x - dx_n * y + c = 0
        a = dy_n
        b = -dx_n
        c = -(a * x1 + b * y1)
        g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
        # Canonical form: a > 0 or (a == 0 and b > 0)
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def random_point():
        return (random.randint(-MAX_COORD, MAX_COORD), random.randint(-MAX_COORD, MAX_COORD))

    # State variables
    # points[i] = (x, y)
    # line_to_pts: dict { line_key: set of point indices }
    # pt_to_lines: list of dicts { point_index: { line_key: None } } - actually just a set of line_keys is enough for removal, but we need to know which lines to update.
    # Actually, for point i, we need to know all lines it participates in to remove it.
    # So pt_to_lines[i] is a set of line_keys.
    
    def init_state(points):
        line_to_pts = defaultdict(set)
        pt_to_lines = [set() for _ in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                key = get_line_key(points[i], points[j])
                if key is None: continue # shouldn't happen with distinct points
                line_to_pts[key].add(i)
                line_to_pts[key].add(j)
                pt_to_lines[i].add(key)
                pt_to_lines[j].add(key)
                
        return line_to_pts, pt_to_lines

    def get_score(line_to_pts):
        count = 0
        for key, pts in line_to_pts.items():
            if len(pts) == 4:
                count += 1
        return count

    def is_valid(line_to_pts):
        for key, pts in line_to_pts.items():
            if len(pts) >= 5:
                return False
        return True

    # Generate a random initial valid configuration
    def generate_initial():
        while True:
            points = []
            used = set()
            for _ in range(N):
                while True:
                    p = random_point()
                    if p not in used:
                        break
                points.append(p)
                used.add(p)
            
            line_to_pts, pt_to_lines = init_state(points)
            if is_valid(line_to_pts):
                return points, line_to_pts, pt_to_lines

    points, line_to_pts, pt_to_lines = generate_initial()
    current_score = get_score(line_to_pts)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    max_iter = 200000 # Can afford more iterations due to O(N) update
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        # Choose a random point to move
        idx = random.randint(0, N-1)
        old_pos = points[idx]
        
        # Try a few random new positions or specific smart moves?
        # Let's stick to random for now but maybe bias towards existing lines?
        # For efficiency, just random.
        new_pos = random_point()
        
        if new_pos == old_pos:
            continue
            
        # Check if new_pos is already used by another point
        # We can maintain a set of used points for O(1) check
        # But we don't have one in the state. Let's add it or just check list.
        # Checking list is O(N). N=16. Negligible.
        if any(p == new_pos for p in points if points.index(p) != idx):
            continue

        # Tentatively move point idx to new_pos
        # 1. Remove idx from all lines it is currently on
        affected_lines = list(pt_to_lines[idx])
        for key in affected_lines:
            line_to_pts[key].remove(idx)
            # If line becomes empty, we can leave it in dict for now, or delete.
            # Deleting might be safer to avoid stale data, but we will re-add.
            # We'll just update counts.
            
        # 2. Add idx to new lines formed with all other points
        new_affected_lines = set()
        for j in range(N):
            if j == idx:
                continue
            key = get_line_key(new_pos, points[j])
            if key is None: continue
            line_to_pts[key].add(idx)
            new_affected_lines.add(key)
            
        # 3. Update pt_to_lines for idx
        pt_to_lines[idx] = new_affected_lines
        
        # 4. Check validity (no 5 collinear)
        # We only need to check the lines that were affected (removed or added)
        # But removing a point might make a line valid if it was 5? No, we start valid.
        # Adding a point might make a line invalid (become 5).
        # So we just need to check if any line in new_affected_lines has size >= 5.
        # Also, lines in affected_lines might have been 5? No, they were valid before.
        # So we only check new_affected_lines.
        
        valid = True
        for key in new_affected_lines:
            if len(line_to_pts[key]) >= 5:
                valid = False
                break
                
        if not valid:
            # Rollback
            # Remove idx from new_affected_lines
            for key in new_affected_lines:
                line_to_pts[key].remove(idx)
            # Re-add idx to old lines
            for key in affected_lines:
                line_to_pts[key].add(idx)
            pt_to_lines[idx] = set(affected_lines)
            # T update
            T *= cooling
            continue

        # 5. Calculate score change
        # We need to know how many 4-point lines we lost and gained.
        # Lost: lines in affected_lines that had size 4 before removal (so size 3 after).
        #       OR lines that had size 5? No, valid state means no 5.
        #       So a line in affected_lines was size k. After removal, size k-1.
        #       If k was 4, it becomes 3. We lose a 4-line.
        #       If k was 3, it becomes 2. No change in 4-count.
        #       If k was 2, it becomes 1. No change.
        # Gained: lines in new_affected_lines that had size 3 before addition (so size 4 after).
        #         If size was 3, becomes 4. We gain a 4-line.
        #         If size was 2, becomes 3. No change.
        
        delta_score = 0
        for key in affected_lines:
            # Current size in line_to_pts is k-1
            # Original size was k
            # If original size was 4, current is 3. We lost 1.
            if len(line_to_pts[key]) == 3:
                delta_score -= 1
            # Note: if original size was 5, it's invalid, but we start valid.
            # If original size was >4, we would have rejected it earlier.
            
        for key in new_affected_lines:
            # Current size in line_to_pts is k+1
            # Original size was k
            # If original size was 3, current is 4. We gained 1.
            if len(line_to_pts[key]) == 4:
                delta_score += 1
                
        new_score = current_score + delta_score
        
        # Acceptance criterion
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            # Accept move
            points[idx] = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            # Reject move, rollback
            for key in new_affected_lines:
                line_to_pts[key].remove(idx)
            for key in affected_lines:
                line_to_pts[key].add(idx)
            pt_to_lines[idx] = set(affected_lines)
            
        T *= cooling

    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
