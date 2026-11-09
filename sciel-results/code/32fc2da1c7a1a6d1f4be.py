import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    MIN_COORD = -100
    MAX_COORD = 100
    GRID_POINTS = [(x, y) for x in range(MIN_COORD, MAX_COORD + 1) for y in range(MIN_COORD, MAX_COORD + 1)]
    
    N = 20
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            return (0, 1, x1)
        if dy == 0:
            return (1, 0, y1)
        
        c = dx * y1 - dy * x1
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        c_n = c // g
        
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
            c_n = -c_n
            
        return (dx_n, dy_n, c_n)

    class State:
        def __init__(self, points):
            self.points = points
            self.line_counts = {}
            self.triples = 0
            self._update_from_scratch()
            
        def _update_from_scratch(self):
            self.line_counts = {}
            for p1, p2 in itertools.combinations(self.points, 2):
                key = get_line_key(p1, p2)
                self.line_counts[key] = self.line_counts.get(key, 0) + 1
            self.triples = 0
            for count in self.line_counts.values():
                if count >= 4:
                    self.triples = -1 # Invalid
                    return
                if count == 3:
                    self.triples += 1
                    
        def is_valid(self):
            return self.triples >= 0

        def swap(self, idx, new_point):
            """
            Tries to swap points[idx] with new_point.
            Returns True if valid and accepted (updates state), False otherwise.
            """
            old_point = self.points[idx]
            if old_point == new_point:
                return False
            if new_point in self.points:
                return False
                
            # 1. Remove old_point
            # Find all lines containing old_point
            # We need to iterate over all other points to find lines involving old_point
            # This is O(N)
            
            lines_to_decrement = []
            for j, p_j in enumerate(self.points):
                if j == idx: continue
                key = get_line_key(old_point, p_j)
                lines_to_decrement.append(key)
                
            # Decrement counts
            # Note: A line might be counted multiple times if we're not careful?
            # No, each pair (old, p_j) defines a unique line.
            # But multiple p_j might be on the same line as old_point.
            # So we need to count how many times each line key appears.
            
            removal_counts = {}
            for key in lines_to_decrement:
                removal_counts[key] = removal_counts.get(key, 0) + 1
                
            # Apply removals
            delta_triples = 0
            for key, cnt in removal_counts.items():
                old_count = self.line_counts.get(key, 0)
                new_count = old_count - cnt
                if new_count < 0:
                    return False # Should not happen
                if new_count == 0:
                    del self.line_counts[key]
                else:
                    self.line_counts[key] = new_count
                    
                # Update triples
                if old_count == 3:
                    delta_triples -= 1 # Lost a triple
                # If old_count was 4, it was invalid, so we assume valid state.
                
            # 2. Add new_point
            lines_to_increment = []
            for j, p_j in enumerate(self.points):
                if j == idx: continue
                key = get_line_key(new_point, p_j)
                lines_to_increment.append(key)
                
            addition_counts = {}
            for key in lines_to_increment:
                addition_counts[key] = addition_counts.get(key, 0) + 1
                
            # Apply additions and check validity
            for key, cnt in addition_counts.items():
                old_count = self.line_counts.get(key, 0)
                new_count = old_count + cnt
                if new_count >= 4:
                    # Invalid! Revert changes
                    # Re-add old_point effects
                    for key_r, cnt_r in removal_counts.items():
                        self.line_counts[key_r] = self.line_counts.get(key_r, 0) + cnt_r
                    # Re-subtract new_point effects
                    for key_a, cnt_a in addition_counts.items():
                        self.line_counts[key_a] = self.line_counts.get(key_a, 0) - cnt_a
                    return False
                    
                if new_count == 0:
                    del self.line_counts[key]
                else:
                    self.line_counts[key] = new_count
                    
                if new_count == 3:
                    delta_triples += 1 # Gained a triple
                # If old_count was 2 and new_count 3, gain 1.
                # If old_count was 1 and new_count 2, gain 0.
                
            # Commit change
            self.points[idx] = new_point
            self.triples += delta_triples
            return True

    # Initialization
    best_state = None
    best_triples = -1
    
    # Try a few random starts
    for init_iter in range(10):
        if time.time() > deadline: break
        
        pts = random.sample(GRID_POINTS, N)
        state = State(pts)
        if not state.is_valid():
            continue
            
        if state.triples > best_triples:
            best_triples = state.triples
            best_state = state.points[:]
            
        # Local Search
        current_triples = state.triples
        no_improve = 0
        max_no_improve = 500
        
        for step in range(2000):
            if time.time() > deadline: break
            
            idx = random.randint(0, N - 1)
            
            # Generate candidates
            # 1. Random
            candidates = [random.choice(GRID_POINTS)]
            
            # 2. Targeted: Complete triples
            # Pick a random pair from remaining points
            other_indices = [i for i in range(N) if i != idx]
            if len(other_indices) >= 2:
                i1, i2 = random.sample(other_indices, 2)
                p1 = state.points[i1]
                p2 = state.points[i2]
                
                # Calculate 3rd point
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                
                # Try small steps
                for t in [-2, -1, 1, 2]:
                    x3 = x1 + t * dx
                    y3 = y1 + t * dy
                    if MIN_COORD <= x3 <= MAX_COORD and MIN_COORD <= y3 <= MAX_COORD:
                        c = (x3, y3)
                        if c not in state.points:
                            candidates.append(c)
                            
            # Try candidates
            improved = False
            for c in candidates:
                # Create a copy of state to test? 
                # Or just test and revert if failed?
                # State.swap modifies in place and reverts on failure.
                # But we need to know if it improved.
                
                old_triples = state.triples
                old_point = state.points[idx]
                
                if state.swap(idx, c):
                    new_triples = state.triples
                    if new_triples > current_triples:
                        current_triples = new_triples
                        improved = True
                        if current_triples > best_triples:
                            best_triples = current_triples
                            best_state = state.points[:]
                    else:
                        # Revert if not improved (Hill Climbing)
                        # We need a revert function or just swap back
                        state.swap(idx, old_point)
                # If swap failed (invalid), state is already reverted
                
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            else:
                no_improve = 0

    if best_state is None:
        return [[x, x * x] for x in range(-10, 10)]
        
    return [[p[0], p[1]] for p in best_state]
