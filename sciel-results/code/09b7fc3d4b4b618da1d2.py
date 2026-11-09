import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate a good initial configuration: 4x4 Grid
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = fast_count(points)
    best_points = [p[:] for p in points]
    best_score = current_score

    if current_score < 0:
        return best_points

    # 2. Simulated Annealing with Incremental Line Counting
    # State: points (list of 16 [x,y])
    # We maintain a dictionary of line_keys -> set of point_indices on that line.
    # Score = number of lines with exactly 4 points.
    # Constraint: No line with >= 5 points.

    def build_line_structure(pts):
        lines = {}
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        return lines

    def get_score_from_lines(lines):
        score = 0
        for key, indices in lines.items():
            if len(indices) == 4:
                score += 1
            elif len(indices) >= 5:
                return -1
        return score

    def remove_point_from_lines(lines, idx, pts):
        """Remove point idx from all lines it participates in. Returns list of (key, old_count) for lines that had idx."""
        affected = []
        # We need to find all lines containing idx.
        # Iterating all lines is O(L). L is ~120-200. This is fast.
        to_delete = []
        for key, indices in lines.items():
            if idx in indices:
                old_count = len(indices)
                indices.remove(idx)
                affected.append((key, old_count))
                if not indices:
                    to_delete.append(key)
        for key in to_delete:
            del lines[key]
        return affected

    def add_point_to_lines(lines, idx, pts):
        """Add point idx to all lines it participates in. Returns list of (key, new_count) for lines that now have idx."""
        affected = []
        p = pts[idx]
        n = len(pts)
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(p, pts[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(idx)
            new_count = len(lines[key])
            affected.append((key, new_count))
        return affected

    # Pre-compute all potential line keys for the bounding box? No, dynamic is better.
    # But we can speed up 'add_point_to_lines' by just checking against existing points.
    
    # Initialize line structure
    lines = build_line_structure(points)
    
    # SA Parameters
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9999
    # We want to run as many iterations as possible.
    # Each iteration:
    # 1. Pick random idx (1)
    # 2. Pick random delta (1)
    # 3. Remove point (O(L) ~ 200 ops)
    # 4. Check validity (O(L) ~ 200 ops) - actually we can check during add
    # 5. Add point (O(N) ~ 16 ops to find lines, then update sets)
    # 6. Calculate score change
    
    # Actually, recalculating score from scratch is O(L).
    # We can track score incrementally.
    
    current_score = get_score_from_lines(lines)
    
    # To make this fast, we track score.
    # When removing a point from a line with count k:
    #   if k == 4: score -= 1
    #   if k == 5: score += 1 (it becomes 4) -- wait, if k>=5 it's invalid.
    #   if k > 5: score unchanged (still invalid)
    
    # Let's implement a strict incremental scorer.
    
    class LineTracker:
        def __init__(self, pts):
            self.lines = build_line_structure(pts)
            self.score = 0
            self.invalid_count = 0 # Number of lines with >= 5 points
            for key, indices in self.lines.items():
                c = len(indices)
                if c == 4:
                    self.score += 1
                elif c >= 5:
                    self.invalid_count += 1

        def remove_point(self, idx):
            # Returns delta_score, delta_invalid
            delta_score = 0
            delta_invalid = 0
            to_delete = []
            for key, indices in self.lines.items():
                if idx in indices:
                    old_c = len(indices)
                    if old_c == 4:
                        delta_score -= 1
                    elif old_c == 5:
                        delta_invalid -= 1
                        delta_score += 1 # It becomes 4, so it's now a valid 4-line
                    # If old_c > 5, no change in score/invalid count (still invalid, not counted)
                    
                    indices.remove(idx)
                    if not indices:
                        to_delete.append(key)
            for key in to_delete:
                del self.lines[key]
            
            self.score += delta_score
            self.invalid_count += delta_invalid
            return delta_score, delta_invalid

        def add_point(self, idx, pts):
            delta_score = 0
            delta_invalid = 0
            p = pts[idx]
            n = len(pts)
            
            # Find all lines containing p and another point
            # We must be careful not to double count lines if p is added to a line 
            # that already has other points.
            # get_line_key(p, pts[j]) gives the unique line.
            # If we add p to line L, the count increases by 1.
            
            touched_lines = set()
            
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(p, pts[j])
                if key is None:
                    continue
                if key in touched_lines:
                    continue
                touched_lines.add(key)
                
                if key not in self.lines:
                    self.lines[key] = set()
                
                old_c = len(self.lines[key])
                self.lines[key].add(idx)
                new_c = old_c + 1
                
                if old_c == 3:
                    delta_score += 1 # Becomes 4
                elif old_c == 4:
                    delta_score -= 1 # Becomes 5 (invalid)
                    delta_invalid += 1
                elif old_c >= 5:
                    # Was invalid, still invalid
                    pass
            
            self.score += delta_score
            self.invalid_count += delta_invalid
            return delta_score, delta_invalid

    tracker = LineTracker(points)
    current_score = tracker.score
    if current_score < 0:
        return best_points

    best_score = current_score
    best_points = [p[:] for p in points]

    # Simulated Annealing Loop
    # We need to handle the "invalid" state.
    # If invalid_count > 0, we strongly penalize.
    
    def get_effective_score(tracker):
        if tracker.invalid_count > 0:
            return -1000 - tracker.invalid_count
        return tracker.score

    current_eff_score = get_effective_score(tracker)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Pick a random small move
        # To escape local optima, occasionally make a larger jump
        if random.random() < 0.1:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
        else:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check collision with other points
        collision = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                collision = True
                break
        if collision:
            continue
            
        # Perform move
        points[idx] = [new_x, new_y]
        
        # Update tracker
        # 1. Remove old point
        tracker.remove_point(idx)
        # 2. Add new point
        tracker.add_point(idx, points)
        
        new_eff_score = get_effective_score(tracker)
        
        delta = new_eff_score - current_eff_score
        
        if delta >= 0:
            current_eff_score = new_eff_score
            if tracker.score > best_score and tracker.invalid_count == 0:
                best_score = tracker.score
                best_points = [p[:] for p in points]
        else:
            # Accept with probability exp(delta / T)
            # If invalid, delta is very negative, so prob is ~0
            if random.random() < pow(2.71828, delta / T):
                current_eff_score = new_eff_score
            else:
                # Revert
                points[idx] = [old_x, old_y]
                tracker.remove_point(idx)
                tracker.add_point(idx, points)
                # Score should be back to current_eff_score
                # (Note: floating point or logic errors could drift, but sets are exact)
                
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck at T_min for a while, restart with higher T?
            # Simple restart:
            # if random.random() < 0.01:
            #    T = 1.0

    # Final check
    final_score = fast_count(best_points)
    if final_score > best_score:
        best_score = final_score
    elif final_score < 0:
        # Fallback to grid
        grid_pts, grid_score = grid_configuration()
        return grid_pts

    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate a good initial configuration: 4x4 Grid
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = fast_count(points)
    best_points = [p[:] for p in points]
    best_score = current_score

    if current_score < 0:
        return best_points

    # 2. Simulated Annealing with Incremental Line Counting
    class LineTracker:
        def __init__(self, pts):
            self.lines = {}
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    key = get_line_key(pts[i], pts[j])
                    if key is None:
                        continue
                    if key not in self.lines:
                        self.lines[key] = set()
                    self.lines[key].add(i)
                    self.lines[key].add(j)
            self.score = 0
            self.invalid_count = 0
            for key, indices in self.lines.items():
                c = len(indices)
                if c == 4:
                    self.score += 1
                elif c >= 5:
                    self.invalid_count += 1

        def remove_point(self, idx):
            delta_score = 0
            delta_invalid = 0
            to_delete = []
            for key, indices in self.lines.items():
                if idx in indices:
                    old_c = len(indices)
                    if old_c == 4:
                        delta_score -= 1
                    elif old_c == 5:
                        delta_invalid -= 1
                        delta_score += 1
                    indices.remove(idx)
                    if not indices:
                        to_delete.append(key)
            for key in to_delete:
                del self.lines[key]
            self.score += delta_score
            self.invalid_count += delta_invalid

        def add_point(self, idx, pts):
            delta_score = 0
            delta_invalid = 0
            p = pts[idx]
            n = len(pts)
            touched_lines = set()
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(p, pts[j])
                if key is None:
                    continue
                if key in touched_lines:
                    continue
                touched_lines.add(key)
                if key not in self.lines:
                    self.lines[key] = set()
                old_c = len(self.lines[key])
                self.lines[key].add(idx)
                new_c = old_c + 1
                if old_c == 3:
                    delta_score += 1
                elif old_c == 4:
                    delta_score -= 1
                    delta_invalid += 1
            self.score += delta_score
            self.invalid_count += delta_invalid

    tracker = LineTracker(points)
    current_score = tracker.score
    
    def get_effective_score(tracker):
        if tracker.invalid_count > 0:
            return -1000 - tracker.invalid_count
        return tracker.score

    current_eff_score = get_effective_score(tracker)
    best_score = current_score
    best_points = [p[:] for p in points]

    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9999

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        if random.random() < 0.1:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
        else:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        collision = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                collision = True
                break
        if collision:
            continue
            
        points[idx] = [new_x, new_y]
        tracker.remove_point(idx)
        tracker.add_point(idx, points)
        
        new_eff_score = get_effective_score(tracker)
        delta = new_eff_score - current_eff_score
        
        if delta >= 0:
            current_eff_score = new_eff_score
            if tracker.score > best_score and tracker.invalid_count == 0:
                best_score = tracker.score
                best_points = [p[:] for p in points]
        else:
            if random.random() < pow(2.71828, delta / T):
                current_eff_score = new_eff_score
            else:
                points[idx] = [old_x, old_y]
                tracker.remove_point(idx)
                tracker.add_point(idx, points)
                
        T *= cooling_rate
        if T < T_min:
            T = T_min

    final_score = fast_count(best_points)
    if final_score < 0:
        grid_pts, _ = grid_configuration()
        return grid_pts

    return best_points
# EVOLVE-BLOCK-END
