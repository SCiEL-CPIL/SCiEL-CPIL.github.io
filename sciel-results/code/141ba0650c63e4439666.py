import time
import random
import sys

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def is_valid_path(path):
        """Check if path is a valid induced path (snake)."""
        if len(path) < 2:
            return True
        used = set(path)
        for i in range(len(path) - 1):
            if abs(bin(path[i] ^ path[i+1]).count('1') - 1) > 0:
                return False
        # Check non-consecutive vertices differ in at least 2 bits
        for i in range(len(path)):
            for j in range(i+2, len(path)):
                if bin(path[i] ^ path[j]).count('1') < 2:
                    return False
        return True
    
    def path_length(path):
        return len(path) - 1
    
    def local_improve(path):
        """Improve path using insertion, reversal, and extension moves."""
        nonlocal deadline, best_path, best_len
        
        path = path[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to extend at both ends
            for end in [0, -1]:
                if time.time() > deadline:
                    return path
                cur = path[end]
                for n in neighbors[cur]:
                    if n not in path:
                        # Check if adding n maintains validity
                        new_path = [n] + path if end == 0 else path + [n]
                        if is_valid_path(new_path):
                            path = new_path
                            improved = True
                            if len(path) - 1 > best_len:
                                best_len = len(path) - 1
                                best_path = path[:]
                            break
            
            # Try insertion moves: remove vertex at position i, insert at position j
            if time.time() > deadline:
                return path
            for i in range(1, len(path) - 1):
                if time.time() > deadline:
                    return path
                v = path[i]
                # Remove v
                path_without_v = path[:i] + path[i+1:]
                # Check if path_without_v is still valid
                if not is_valid_path(path_without_v):
                    continue
                
                # Try inserting v at different positions
                for j in range(len(path_without_v)):
                    if time.time() > deadline:
                        return path
                    # Don't insert adjacent to where it was
                    if abs(j - i) <= 1:
                        continue
                    # Try inserting before position j
                    new_path = path_without_v[:j] + [v] + path_without_v[j:]
                    if is_valid_path(new_path):
                        # Check if this improves something (for now, just accept if valid)
                        # We could add a heuristic here
                        path = new_path
                        improved = True
                        if len(path) - 1 > best_len:
                            best_len = len(path) - 1
                            best_path = path[:]
                        break
                if improved:
                    break
            
            # Try reversal moves: reverse a subpath
            if time.time() > deadline:
                return path
            for i in range(len(path) - 2):
                if time.time() > deadline:
                    return path
                for j in range(i + 2, len(path)):
                    if time.time() > deadline:
                        return path
                    # Reverse path[i:j+1]
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    if is_valid_path(new_path):
                        path = new_path
                        improved = True
                        if len(path) - 1 > best_len:
                            best_len = len(path) - 1
                            best_path = path[:]
                        break
                if improved:
                    break
        
        return path
    
    # Start with a greedy snake
    def greedy_snake():
        """Build a greedy snake by always extending to a vertex with fewest options."""
        path = [random.randint(0, MAX_V - 1)]
        used = set(path)
        
        while True:
            cur = path[-1]
            candidates = []
            for n in neighbors[cur]:
                if n not in used:
                    # Count how many valid neighbors n has (Warnsdorff)
                    count = 0
                    for nn in neighbors[n]:
                        if nn not in used:
                            count += 1
                    candidates.append((count, n))
            
            if not candidates:
                break
            
            # Choose the candidate with fewest options (Warnsdorff's rule)
            candidates.sort()
            next_v = candidates[0][1]
            path.append(next_v)
            used.add(next_v)
        
        return path
    
    # Run multiple restarts with local search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial greedy snake
        path = greedy_snake()
        
        # Apply local search
        improved_path = local_improve(path)
        
        if len(improved_path) - 1 > best_len:
            best_len = len(improved_path) - 1
            best_path = improved_path[:]
    
    # Ensure we return a valid path
    if not best_path:
        return [0]
        
    return best_path
