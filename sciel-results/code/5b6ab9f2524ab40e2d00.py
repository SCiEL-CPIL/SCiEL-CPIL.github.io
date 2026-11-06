import time
import random

N = 9
# The evaluator kills after 150s. We use 145s to be safe.
DEADLINE = 145.0

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Symmetry breaking:
    # 1. Fix first vertex to 0.
    # 2. Fix first step to flip bit 0. So path starts [0, 1].
    # This is valid because any snake can be translated and rotated/reflected to start this way.
    
    path = [0, 1]
    visited = {0, 1}
    
    def dfs(path, visited):
        nonlocal best_path, best_len, start_time
        
        # Time check
        if time.time() - start_time > DEADLINE:
            return
        
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best_path = path.copy()
            
        cur = path[-1]
        
        # Generate all valid next moves
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in visited:
                continue
            
            # Check if v is adjacent to any previous vertex (except cur)
            # If so, it creates a shortcut, violating induced path property.
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            
            if valid:
                candidates.append(v)
        
        if not candidates:
            return
            
        # Shuffle candidates to randomize the search path
        # This helps escape local minima in the search tree
        random.shuffle(candidates)
        
        for v in candidates:
            path.append(v)
            visited.add(v)
            
            dfs(path, visited)
            
            path.pop()
            visited.remove(v)
            
            # Time check inside loop
            if time.time() - start_time > DEADLINE:
                return
                
    dfs(path, visited)
    
    return best_path
