"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(x1, y1, x2, y2):
    """Return normalized (a, b, c) for line ax+by+c=0."""
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

class State:
    def __init__(self, points):
        self.points = [list(p) for p in points]
        self.n = len(self.points)
        self.line_map = defaultdict(list)
        self.ordinary_count = 0
        self._build_line_map()

    def _build_line_map(self):
        self.line_map.clear()
        n = self.n
        for i in range(n):
            x1, y1 = self.points[i]
            for j in range(i + 1, n):
                x2, y2 = self.points[j]
                key = _normalize_line(x1, y1, x2, y2)
                self.line_map[key].append(i)
        
        # Count ordinary lines
        self.ordinary_count = sum(1 for pts in self.line_map.values() if len(pts) == 2)

    def get_ordinary_count(self):
        return self.ordinary_count

    def move_point(self, idx, new_x, new_y):
        """Move point idx to (new_x, new_y) and update state incrementally."""
        old_x, old_y = self.points[idx]
        
        # 1. Remove idx from all lines it was on
        # We need to find all lines containing idx.
        # Instead of scanning all lines, we can just remove idx from line_map entries.
        # But we need to know which lines to update.
        # Efficient way: 
        # The lines containing idx are exactly the lines (idx, j) for j != idx.
        # We will update the line_map for these specific keys.
        
        # First, collect the lines that will be affected by removal.
        lines_to_update = set()
        for j in range(self.n):
            if j == idx: continue
            x2, y2 = self.points[j]
            key = _normalize_line(old_x, old_y, x2, y2)
            lines_to_update.add(key)
            
        # Remove idx from these lines
        for key in lines_to_update:
            if key in self.line_map:
                lst = self.line_map[key]
                if idx in lst:
                    lst.remove(idx)
                    # If line size became 2, it's now ordinary (if it wasn't before? No, it was size >=3 or 2)
                    # If size was 3 -> 2: ordinary_count increases by 1
                    # If size was 2 -> 1: ordinary_count decreases by 1 (but lines of size 1 are not counted)
                    # If size was 2 -> 1, the line effectively disappears from "ordinary" count.
                    # Actually, we only count lines with size == 2.
                    # So:
                    # If len(lst) was 3 and is now 2: +1 ordinary
                    # If len(lst) was 2 and is now 1: -1 ordinary
                    # If len(lst) was >3 and is now >=3: 0 change
                    # If len(lst) was 2 and is now 1: -1
                    # We need to track the size before removal.
                    pass
        
        # This incremental update is tricky because we need the size before removal.
        # Let's do it carefully.
        
        # Re-implementation of move_point with careful counting
        
        # Store old sizes
        old_sizes = {}
        for key in lines_to_update:
            if key in self.line_map:
                old_sizes[key] = len(self.line_map[key])
                
        # Perform removal
        for key in lines_to_update:
            if key in self.line_map:
                lst = self.line_map[key]
                if idx in lst:
                    lst.remove(idx)
                    if len(lst) == 0:
                        del self.line_map[key]
                        
        # Update ordinary_count based on old_sizes and new sizes
        for key in lines_to_update:
            old_size = old_sizes.get(key, 0)
            new_size = len(self.line_map.get(key, []))
            
            # Transition:
            # old_size == 2, new_size == 1: -1
            # old_size == 3, new_size == 2: +1
            # old_size == 2, new_size == 2: 0 (impossible, size changed)
            # old_size > 3, new_size >= 3: 0
            
            if old_size == 2 and new_size == 1:
                self.ordinary_count -= 1
            elif old_size == 3 and new_size == 2:
                self.ordinary_count += 1
            # Note: If old_size was 1, it wasn't in line_map as an ordinary line candidate? 
            # No, line_map contains all lines with >= 2 points? 
            # My _build_line_map adds pairs. So lines with 1 point are not in the map.
            # So old_size is always >= 2.
            # If old_size == 2 and new_size == 1, the line is removed from map.
            # If old_size == 3 and new_size == 2, the line remains.

        # 2. Add idx to new position
        self.points[idx] = [new_x, new_y]
        
        new_lines_to_update = set()
        for j in range(self.n):
            if j == idx: continue
            x2, y2 = self.points[j]
            key = _normalize_line(new_x, new_y, x2, y2)
            new_lines_to_update.add(key)
            
        # Store new sizes (before adding)
        new_sizes = {}
        for key in new_lines_to_update:
            if key in self.line_map:
                new_sizes[key] = len(self.line_map[key])
            else:
                new_sizes[key] = 0
                
        # Perform addition
        for key in new_lines_to_update:
            if key not in self.line_map:
                self.line_map[key] = []
            self.line_map[key].append(idx)
            
        # Update ordinary_count
        for key in new_lines_to_update:
            old_size = new_sizes.get(key, 0)
            new_size = len(self.line_map[key])
            
            # Transition:
            # old_size == 1, new_size == 2: +1 (new ordinary line)
            # old_size == 2, new_size == 3: -1 (ordinary line becomes non-ordinary)
            # old_size == 0, new_size == 1: 0 (line created, but size 1 not counted)
            
            if old_size == 1 and new_size == 2:
                self.ordinary_count += 1
            elif old_size == 2 and new_size == 3:
                self.ordinary_count -= 1
            # If old_size == 0, new_size == 1: no change to ordinary count.
            # If old_size >= 3, no change.

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    # Initial state: 3x3 grid + 3 random points
    # 3x3 grid has 12 ordinary lines.
    base_pts = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 more points far away to avoid accidental collinearity initially?
    # Or just start with 12 random points?
    # Let's start with a perturbed 3x3 grid.
    
    best_state = None
    best_score = float('inf')
    
    # Try a few random initializations
    for init in range(5):
        if time.time() > deadline - 10: break
        
        pts = []
        if init == 0:
            # 3x3 grid
            pts = [[x, y] for x in range(3) for y in range(3)]
            # Add 3 more points
            pts.append([3, 0])
            pts.append([0, 3])
            pts.append([3, 3])
        else:
            # Random
            for _ in range(12):
                pts.append([random.randint(-10, 10), random.randint(-10, 10)])
        
        # Ensure distinct
        seen = set()
        valid_pts = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                valid_pts.append(p)
        while len(valid_pts) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                valid_pts.append(p)
        
        state = State(valid_pts)
        score = state.get_ordinary_count()
        
        if score < best_score:
            best_score = score
            best_state = state
        
        # Simulated Annealing
        current_temp = 10.0
        cooling = 0.99
        current_state = state
        current_score = score
        
        steps = 0
        max_steps = 5000
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            idx = random.randint(0, 11)
            ox, oy = current_state.points[idx]
            
            # Move to a nearby point
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            nx, ny = ox + dx, oy + dy
            
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            
            # Check distinctness
            valid = True
            for j in range(12):
                if j == idx: continue
                if current_state.points[j][0] == nx and current_state.points[j][1] == ny:
                    valid = False
                    break
            if not valid:
                continue
            
            # Try move
            old_score = current_score
            current_state.move_point(idx, nx, ny)
            new_score = current_state.get_ordinary_count()
            
            delta = new_score - old_score
            
            if delta < 0 or (random.random() < pow(2.71828, -delta / current_temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_state = current_state
                    if best_score <= 6:
                        return [list(p) for p in best_state.points]
            else:
                # Revert
                current_state.move_point(idx, ox, oy)
                # Verify score is restored
                # (move_point is symmetric, so it should be)
            
            current_temp *= cooling
            
            if current_temp < 0.01:
                break
        
        # Restart with perturbation of best_state if found
        if best_state and init > 0:
             # Perturb best_state slightly for next restart?
             pass

    if best_state:
        return [list(p) for p in best_state.points]
    
    # Fallback: 3x4 grid
    return [[x, y] for x in range(4) for y in range(3)]

# EVOLVE-BLOCK-END
