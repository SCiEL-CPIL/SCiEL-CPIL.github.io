import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 4, 7],
        list(range(10)),
        [0, 1, 3, 9, 27], # Powers of 3
    ]
    
    best_set = candidates[0]
    best_score = -np.inf
    
    def score_set(U):
        if not U:
            return -np.inf
        U = np.array(sorted(U))
        if U[0] != 0:
            # Ensure 0 is in U, but construct should return valid set.
            # If 0 not in U, add it? No, task says containing 0.
            # Our search maintains 0.
            pass
        if len(U) == 1:
            return 1.0 # Degenerate?
        
        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]
        
        # Unique counts
        # For small U, np.unique is fine.
        # For large U, this is O(n^2 log n).
        len_sums = len(np.unique(sums))
        len_diffs = len(np.unique(diffs))
        
        max_u = U[-1]
        if max_u == 0:
            return 1.0
            
        denom = np.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        ratio = len_diffs / len_sums
        if ratio <= 0:
            return -np.inf
            
        return 1.0 + np.log(ratio) / denom

    # Evaluate initial candidates
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]

    # Local Search
    # We will perform random restarts and hill climbing
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start from a random small set or best_set
        if r == 0:
            current_set = best_set[:]
        else:
            # Random initialization
            size = np.random.randint(3, 20)
            # Generate random sorted set with 0
            vals = np.sort(np.random.randint(0, 50, size=size))
            vals = np.unique(vals)
            if 0 not in vals:
                vals = np.insert(vals, 0, 0)
            current_set = list(vals)
            
        current_score = score_set(current_set)
        
        # Hill climbing
        max_steps = 1000
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Generate a move
            move_type = np.random.choice(['add', 'remove', 'shift'])
            
            new_set = current_set[:]
            
            if move_type == 'add':
                # Add a random element
                max_val = max(new_set) if new_set else 0
                # Try adding near existing elements or far
                if np.random.rand() < 0.5:
                    base = np.random.choice(new_set)
                    offset = np.random.randint(-10, 10)
                    val = base + offset
                else:
                    val = np.random.randint(0, max_val + 20)
                
                if val >= 0 and val not in new_set:
                    new_set.append(val)
                    new_set.sort()
                    
            elif move_type == 'remove':
                if len(new_set) > 1:
                    # Don't remove 0
                    idx = np.random.randint(1, len(new_set))
                    new_set.pop(idx)
                    
            elif move_type == 'shift':
                if len(new_set) > 1:
                    idx = np.random.randint(1, len(new_set))
                    old_val = new_set[idx]
                    # Shift to nearby
                    new_val = old_val + np.random.randint(-5, 5)
                    if new_val >= 0 and new_val not in new_set:
                        new_set[idx] = new_val
                        new_set.sort()
                    else:
                        continue # Skip invalid move
            
            new_score = score_set(new_set)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
            else:
                # Occasionally accept worse moves to escape local minima (Simulated Annealing style)
                # But simple hill climbing might be enough for now.
                # Let's try a simple probability of acceptance if close
                if np.random.rand() < 0.01:
                    current_set = new_set
                    current_score = new_score

        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]

    return best_set
