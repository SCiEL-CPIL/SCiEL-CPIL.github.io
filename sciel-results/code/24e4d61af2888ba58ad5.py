import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def is_valid_move(path, used, candidate, is_left_end):
        """
        Check if candidate can be added to the path.
        is_left_end: True if adding to the front, False if adding to the back.
        """
        if used[candidate]:
            return False
        
        # The candidate is adjacent to the current end.
        # We must ensure candidate is NOT adjacent to any other vertex in the path.
        
        # If adding to the left, the current end is path[0].
        # We need to check against path[1], path[2], ...
        # If adding to the right, the current end is path[-1].
        # We need to check against path[-2], path[-3], ...
        
        # Instead of iterating path, we check neighbors of candidate.
        # If any neighbor of candidate (other than the current end) is in `used`, it's invalid.
        
        current_end = path[0] if is_left_end else path[-1]
        
        for u in neighbors[candidate]:
            if u == current_end:
                continue
            if used[u]:
                return False
        return True

    def get_valid_moves(path, used, is_left_end):
        """Return list of valid candidates for the specified end."""
        current_end = path[0] if is_left_end else path[-1]
        valid = []
        for v in neighbors[current_end]:
            if is_valid_move(path, used, v, is_left_end):
                valid.append(v)
        return valid

    def bidirectional_grow(initial_path=None):
        if initial_path is None:
            # Start with a random edge
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(neighbors[v0])
            path = [v0, v1]
        else:
            path = list(initial_path)
        
        if len(path) < 2:
            return path
            
        used = [False] * MAX_VERTS
        for v in path:
            used[v] = True
            
        while True:
            if time.time() - start_time > DEADLINE:
                return path
            
            valid_left = get_valid_moves(path, used, True)
            valid_right = get_valid_moves(path, used, False)
            
            if not valid_left and not valid_right:
                return path
            
            # Warnsdorff: pick the end with fewer options
            if len(valid_left) == 0:
                next_v = random.choice(valid_right)
                path.append(next_v)
                used[next_v] = True
            elif len(valid_right) == 0:
                next_v = random.choice(valid_left)
                path.insert(0, next_v)
                used[next_v] = True
            else:
                if len(valid_left) < len(valid_right):
                    next_v = random.choice(valid_left)
                    path.insert(0, next_v)
                    used[next_v] = True
                elif len(valid_right) < len(valid_left):
                    next_v = random.choice(valid_right)
                    path.append(next_v)
                    used[next_v] = True
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(valid_left)
                        path.insert(0, next_v)
                        used[next_v] = True
                    else:
                        next_v = random.choice(valid_right)
                        path.append(next_v)
                        used[next_v] = True

    def generate_initial_greedy():
        """Generate a short initial path using greedy forward growth."""
        v0 = random.randint(0, MAX_VERTS - 1)
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        # Greedy forward
        for _ in range(20): # Limit initial length
            current = path[-1]
            valid_next = []
            for v in neighbors[current]:
                if used[v]:
                    continue
                # Check validity
                valid = True
                for u in neighbors[v]:
                    if u == current:
                        continue
                    if used[u]:
                        valid = False
                        break
                if valid:
                    valid_next.append(v)
            
            if not valid_next:
                break
            
            # Pick one with fewest options (Warnsdorff)
            # Simple heuristic: just pick one
            next_v = random.choice(valid_next)
            path.append(next_v)
            used[next_v] = True
            
        return path

    attempts = 0
    while time.time() - start_time < DEADLINE:
        # Strategy: Generate a short greedy path, then extend bidirectionally
        initial = generate_initial_greedy()
        
        if len(initial) < 2:
            continue
            
        p = bidirectional_grow(initial)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        attempts += 1
        
    if not best_path:
        # Fallback
        best_path = [0, 1]
        
    return best_path
