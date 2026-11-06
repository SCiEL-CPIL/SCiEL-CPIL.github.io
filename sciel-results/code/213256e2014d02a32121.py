import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Helper to generate a grid
    def make_grid(w, h, offset_x=0, offset_y=0):
        return [(x + offset_x, y + offset_y) for x in range(w) for y in range(h)]
    
    # Initial candidates
    # Center grids to allow movement in all directions
    candidates = [
        make_grid(6, 5, -2, -2), # 6x5 centered roughly
        make_grid(5, 6, -2, -3),
        make_grid(3, 10, -1, -5),
        make_grid(10, 3, -5, -1),
        make_grid(2, 15, -1, -7),
        make_grid(15, 2, -7, -1),
    ]
    
    # Add a random one
    all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    candidates.append(random.sample(all_pts, 30))
    
    best_state = None
    best_score = -1
    
    for c in candidates:
        # Ensure valid
        if len(set(c)) != 30:
            continue
        s = State(c)
        if s.max_count > best_score:
            best_score = s.max_count
            best_state = s
            
    current_state = best_state
    current_score = best_score
    
    # Local Search
    while time.time() < deadline:
        i = random.randint(0, 29)
        old_x, old_y = current_state.points[i]
        
        # 80% local, 20% global
        if random.random() < 0.8:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
        else:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pos = (new_x, new_y)
        if new_pos == current_state.points[i]:
            continue
            
        # Try move
        # We need to check collision before calling move_point which assumes valid?
        # My move_point checks collision.
        
        # Save state to revert if needed?
        # move_point modifies state.
        # We need to know if we want to keep it.
        
        # Simplest: Copy points, apply move to copy, check score?
        # No, State.move_point is incremental.
        # Let's make move_point return success and allow revert?
        # Or just snapshot the points list?
        # Snapshotting 30 points is cheap.
        
        old_points = current_state.points[:]
        
        if current_state.move_point(i, new_pos):
            new_score = current_state.max_count
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    # Deep copy best state
                    best_state = State(current_state.points)
            elif new_score == current_score:
                if random.random() < 0.5:
                    # Keep
                    pass
                else:
                    # Revert
                    current_state.points = old_points
                    current_state._update_all()
                    current_score = current_state.max_count
            else:
                # Revert
                current_state.points = old_points
                current_state._update_all()
                current_score = current_state.max_count
        else:
            # Collision or same point, do nothing
            pass
            
    return best_state.points

class State:
    def __init__(self, points):
        self.points = points
        self.n = len(points)
        self.areas_counter = Counter()
        self.max_count = 0
        self._update_all()
        
    def _update_all(self):
        self.areas_counter.clear()
        n = self.n
        pts = self.points
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    d = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    if d > 0:
                        self.areas_counter[d] += 1
        if self.areas_counter:
            self.max_count = max(self.areas_counter.values())
        else:
            self.max_count = 0

    def move_point(self, i, new_pos):
        old_pos = self.points[i]
        if old_pos == new_pos:
            return False
            
        for j in range(self.n):
            if j != i and self.points[j] == new_pos:
                return False
                
        other_indices = [x for x in range(self.n) if x != i]
        
        # Remove
        for idx_a in range(len(other_indices)):
            j = other_indices[idx_a]
            for idx_b in range(idx_a + 1, len(other_indices)):
                k = other_indices[idx_b]
                x1, y1 = old_pos
                x2, y2 = self.points[j]
                x3, y3 = self.points[k]
                d = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                if d > 0:
                    self.areas_counter[d] -= 1
                    if self.areas_counter[d] == 0:
                        del self.areas_counter[d]
                    
        self.points[i] = new_pos
        
        # Add
        for idx_a in range(len(other_indices)):
            j = other_indices[idx_a]
            for idx_b in range(idx_a + 1, len(other_indices)):
                k = other_indices[idx_b]
                x1, y1 = new_pos
                x2, y2 = self.points[j]
                x3, y3 = self.points[k]
                d = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                if d > 0:
                    self.areas_counter[d] += 1
                    
        if self.areas_counter:
            self.max_count = max(self.areas_counter.values())
        else:
            self.max_count = 0
            
        return True
