import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * p1[0] - dy * p1[1]
    return (dx, dy, c)

class PointConfig:
    def __init__(self, points):
        self.points = [tuple(p) for p in points]
        self.n = len(self.points)
        self.line_map = defaultdict(set)
        self.score = 0
        self._build()

    def _build(self):
        self.line_map.clear()
        for i in range(self.n):
            for j in range(i+1, self.n):
                key = _line_key(self.points[i], self.points[j])
                if key:
                    self.line_map[key].add(i)
                    self.line_map[key].add(j)
        self.score = sum(1 for s in self.line_map.values() if len(s) == 4)
        
    def _get_lines_through(self, idx):
        """Return set of line keys passing through point idx."""
        p = self.points[idx]
        lines = set()
        for j in range(self.n):
            if j == idx: continue
            key = _line_key(p, self.points[j])
            if key:
                lines.add(key)
        return lines

    def try_move(self, idx, new_pos):
        """
        Try to move point idx to new_pos.
        Returns (is_valid, delta_score)
        """
        old_pos = self.points[idx]
        if old_pos == new_pos:
            return False, 0
        
        # Check if new_pos is already occupied
        # We can maintain a set of occupied points for O(1) check
        # But for now, linear search is O(N)=16, which is fine.
        if any(p == new_pos for p in self.points):
            return False, 0

        # Get lines affected by removal and addition
        lines_remove = self._get_lines_through(idx)
        
        # Lines through new_pos with other points
        lines_add = set()
        for j in range(self.n):
            if j == idx: continue
            key = _line_key(new_pos, self.points[j])
            if key:
                lines_add.add(key)

        # Combine unique lines
        all_lines = lines_remove.union(lines_add)
        
        delta_score = 0
        is_valid = True
        
        # We need to simulate the change to check validity and score
        # For each line in all_lines, determine old_size and new_size
        # old_size: current size in line_map
        # new_size: 
        #   if line in lines_remove: old_size - 1
        #   if line in lines_add: old_size + 1
        #   if both: old_size (remove then add)
        
        # However, validity check: no line should have size >= 5.
        # We must check the new_size for all lines in all_lines.
        
        for key in all_lines:
            old_size = len(self.line_map[key])
            
            in_remove = key in lines_remove
            in_add = key in lines_add
            
            if in_remove and in_add:
                new_size = old_size
            elif in_remove:
                new_size = old_size - 1
            elif in_add:
                new_size = old_size + 1
            else:
                new_size = old_size # Should not happen for keys in all_lines

            if new_size >= 5:
                is_valid = False
                break
            
            # Score change
            if in_remove and not in_add:
                if old_size == 4:
                    delta_score -= 1
                # if old_size == 5, it becomes 4, but we break if >=5, so this is safe
            elif in_add and not in_remove:
                if new_size == 4: # old_size == 3
                    delta_score += 1
                # if old_size == 4, new_size == 5, invalid (caught above)
            elif in_remove and in_add:
                # Size unchanged, score unchanged
                pass

        return is_valid, delta_score

    def apply_move(self, idx, new_pos):
        old_pos = self.points[idx]
        lines_remove = self._get_lines_through(idx)
        
        lines_add = set()
        for j in range(self.n):
            if j == idx: continue
            key = _line_key(new_pos, self.points[j])
            if key:
                lines_add.add(key)

        all_lines = lines_remove.union(lines_add)
        
        # Update line_map
        for key in all_lines:
            in_remove = key in lines_remove
            in_add = key in lines_add
            
            if in_remove:
                self.line_map[key].remove(idx)
            if in_add:
                self.line_map[key].add(idx)
                
            # Clean up empty lines? Optional.
            if len(self.line_map[key]) == 0:
                del self.line_map[key]

        self.points[idx] = new_pos
        # Recalculate score? No, we can update it, but recalculating is O(L) which is small.
        # Let's just recalculate for safety and simplicity, as L is ~120.
        self.score = sum(1 for s in self.line_map.values() if len(s) == 4)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial 4x4 grid
    init_points = [[x, y] for x in range(4) for y in range(4)]
    config = PointConfig(init_points)
    
    best_score = config.score
    best_points = config.points[:]
    
    rng = random.Random(42)
    
    # Candidate generation strategy
    # 1. Points on lines with 3 points
    # 2. Random points
    
    while time.time() < deadline:
        improved = False
        
        # Try to find an improving move
        # We can iterate through points and try a set of candidates
        # To be fast, we limit candidates per point per iteration
        
        # Generate candidates
        candidates = set()
        
        # Add points from lines with exactly 3 points
        # We need to find lines with 3 points
        # This requires iterating line_map
        for key, pts in config.line_map.items():
            if len(pts) == 3:
                # Get the line equation from key
                dx, dy, c = key
                # Line: dx*x - dy*y = c
                # We want integer points (x,y) on this line within bounds
                # x, y in [-100, 100] maybe?
                # To find points, we can iterate x or y.
                # If dy != 0, y = (dx*x - c) / dy
                # If dx != 0, x = (c + dy*y) / dx
                
                # Simple approach: iterate a small range around existing points
                # Or just add the existing 3 points' neighbors?
                # Better: Iterate x in a range and check if y is integer.
                # Range: [-100, 100]
                # This might be slow if done for every 3-point line every iteration.
                # There are at most ~120 lines.
                # Let's do it once per "epoch" or if we are stuck.
                
                pass # Skip for now, use random + local grid

        # Local grid around current points
        for p in config.points:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = p[0] + dx, p[1] + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.add((nx, ny))
        
        # Add some random points
        for _ in range(50):
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            candidates.add((x, y))
        
        # Shuffle candidates
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        # Try moves
        # To allow escape, we can accept moves with delta > 0.
        # If no move with delta > 0, try a random move with delta >= -2 (simulated annealing-like).
        
        best_delta = -100
        best_move = None
        
        for idx in range(16):
            for cand in cand_list:
                if cand == config.points[idx]:
                    continue
                is_valid, delta = config.try_move(idx, cand)
                if is_valid and delta > best_delta:
                    best_delta = delta
                    best_move = (idx, cand)
                    
        if best_move is not None and best_delta > 0:
            idx, cand = best_move
            config.apply_move(idx, cand)
            improved = True
            if config.score > best_score:
                best_score = config.score
                best_points = config.points[:]
        else:
            # No improving move found
            # Try a random valid move (neutral or slightly negative) to escape
            # Or just break and restart with a different seed?
            # Let's try a random valid move
            random_move_found = False
            for _ in range(20): # Try 20 random candidates
                idx = rng.randint(0, 15)
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                cand = (x, y)
                if cand == config.points[idx]:
                    continue
                is_valid, delta = config.try_move(idx, cand)
                if is_valid:
                    # Accept if delta >= -2
                    if delta >= -2:
                        config.apply_move(idx, cand)
                        random_move_found = True
                        break
            
            if not random_move_found:
                # Stuck completely? Restart
                # Try a new random configuration?
                # Or just break
                break

    return [list(p) for p in best_points]
