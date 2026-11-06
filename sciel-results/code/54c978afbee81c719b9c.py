import time
import random
import sys
from collections import deque

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute neighbor sets for O(1) chord check
NEIGHBOR_SETS = [set(NEIGHBORS[v]) for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Best path found so far
    best_path = None
    best_len = -1
    
    def is_valid_snake(path):
        """Check if a list of vertices forms a valid snake."""
        if len(path) <= 1:
            return True
        if len(set(path)) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                dist = (path[i] ^ path[j]).bit_count()
                if i + 1 == j:
                    if dist != 1:
                        return False
                else:
                    if dist < 2:
                        return False
        return True
    
    def greedy_extend(path, randomize=False):
        """Greedily extend a snake path."""
        path = list(path)
        visited = set(path)
        current = path[-1] if path else 0
        
        max_steps = 500
        for _ in range(max_steps):
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                # Check chord condition: nb must not be adjacent to any path vertex except the last one
                # Using precomputed neighbor sets for O(1) check
                is_chord = False
                for v in path[:-1]:
                    if nb in NEIGHBOR_SETS[v]:
                        is_chord = True
                        break
                if not is_chord:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            if randomize:
                next_v = random.choice(valid_nexts)
            else:
                # Heuristic: pick the neighbor with fewest valid options (most constrained)
                def count_options(nb):
                    count = 0
                    for nb2 in NEIGHBORS[nb]:
                        if nb2 in visited:
                            continue
                        if nb2 == current:
                            continue
                        # Check if nb2 is a valid next from nb
                        is_valid = True
                        for p in path:
                            if nb2 in NEIGHBOR_SETS[p]:
                                is_valid = False
                                break
                        if is_valid:
                            count += 1
                    return count
                valid_nexts.sort(key=lambda x: count_options(x))
                next_v = valid_nexts[0]
            
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        return path
    
    def ida_star_search(path, time_limit):
        """
        Iterative Deepening A* search to find the longest snake.
        Uses a heuristic based on the number of free neighbors to guide the search.
        """
        path = list(path)
        visited = set(path)
        
        def get_valid_nexts(current, path, visited):
            """Get valid next vertices from current."""
            valids = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                # Check chord: nb must not be adjacent to any vertex in path[:-1]
                is_valid = True
                for v in path[:-1]:
                    if nb in NEIGHBOR_SETS[v]:
                        is_valid = False
                        break
                if is_valid:
                    valids.append(nb)
            return valids
        
        def heuristic(path, visited):
            """
            Heuristic: estimate the potential length of the path.
            Use the number of free neighbors of the current vertex as a proxy for potential.
            """
            if not path:
                return 0
            current = path[-1]
            free_count = 0
            for nb in NEIGHBORS[current]:
                if nb not in visited:
                    # Check if nb is a valid next
                    is_valid = True
                    for v in path[:-1]:
                        if nb in NEIGHBOR_SETS[v]:
                            is_valid = False
                            break
                    if is_valid:
                        free_count += 1
            return free_count
        
        best_local = path[:]
        max_local_len = len(path)
        
        # Iterative deepening: start with a small depth limit and increase
        # Since we don't know the optimal depth, we'll use a time-limited DFS with A* guidance
        sys.setrecursionlimit(10000)
        
        def dfs_a_star(path, visited, current, depth, bound):
            nonlocal best_local, max_local_len
            
            if time.time() > time_limit:
                return -1  # Time out
            
            # Update best if this path is longer
            if len(path) > max_local_len:
                max_local_len = len(path)
                best_local = path[:]
            
            # Get valid nexts
            valids = get_valid_nexts(current, path, visited)
            
            if not valids:
                return 0  # Dead end, return 0 additional length
            
            # Sort valids by heuristic (descending) to explore promising branches first
            valids.sort(key=lambda x: -heuristic([x], visited))
            
            min_g = float('inf')
            for nb in valids:
                if time.time() > time_limit:
                    return -1
                
                path.append(nb)
                visited.add(nb)
                
                g = dfs_a_star(path, visited, nb, depth + 1, bound)
                
                path.pop()
                visited.remove(nb)
                
                if g == -1:
                    return -1  # Time out
                
                min_g = min(min_g, g)
                
                if g + depth + 1 > bound:
                    return g + depth + 1
            
            return min_g
        
        # Use a simple DFS with A* guidance, increasing the bound if needed
        # For simplicity, we'll just do a time-limited DFS with A* ordering
        def dfs_simple(path, visited, current):
            nonlocal best_local, max_local_len
            
            if time.time() > time_limit:
                return
            
            # Update best if this path is longer
            if len(path) > max_local_len:
                max_local_len = len(path)
                best_local = path[:]
            
            # Get valid nexts
            valids = get_valid_nexts(current, path, visited)
            
            if not valids:
                return
            
            # Sort valids by heuristic (descending) to explore promising branches first
            valids.sort(key=lambda x: -heuristic([x], visited))
            
            for nb in valids:
                if time.time() > time_limit:
                    return
                
                path.append(nb)
                visited.add(nb)
                
                dfs_simple(path, visited, nb)
                
                path.pop()
                visited.remove(nb)
        
        dfs_simple(path, visited, path[-1] if path else 0)
        return best_local
    
    def local_search(path, time_limit):
        """
        Local search: try to improve the path by finding better extensions 
        or by reordering segments.
        """
        path = list(path)
        visited = set(path)
        best = path[:]
        best_len = len(path)
        
        # Try extending from the end
        ext = greedy_extend(path, randomize=True)
        if len(ext) > best_len:
            best = ext
            best_len = len(ext)
        
        # Try A* search from the current path
        if time.time() < time_limit - 5:
            bt = ida_star_search(path, min(time_limit, time.time() + 10))
            if len(bt) > best_len:
                best = bt
                best_len = len(bt)
        
        return best
    
    # Phase 1: Generate initial snakes using greedy with different starts
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        start_v = random.randint(0, SIZE - 1)
        path = [start_v]
        path = greedy_extend(path, randomize=(i % 3 != 0))
        
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
            
            # Validate
            if not is_valid_snake(best_path):
                print("Invalid snake found during initial generation!")
                best_path = [0]
                best_len = 0
    
    # Phase 2: Local search on the best path
    if time.time() < deadline - 15 and best_path:
        improved = local_search(best_path, min(deadline - 5, time.time() + 30))
        if len(improved) - 1 > best_len:
            best_len = len(improved) - 1
            best_path = improved
    
    # Phase 3: More A* searches from different starting points
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start from a random vertex, do a quick greedy then A*
        start_v = random.randint(0, SIZE - 1)
        path = [start_v]
        path = greedy_extend(path, randomize=True)
        
        # Try to extend with A*
        if time.time() < deadline - 5:
            bt = ida_star_search(path, min(deadline - 2, time.time() + 5))
            if len(bt) - 1 > best_len:
                best_len = len(bt) - 1
                best_path = bt[:]
    
    # Phase 4: Final greedy extension on the best path
    if best_path and time.time() < deadline - 2:
        ext = greedy_extend(best_path, randomize=False)
        if len(ext) - 1 > best_len:
            best_len = len(ext) - 1
            best_path = ext
    
    if best_path is None:
        return [0]
    
    # Final validation
    if not is_valid_snake(best_path):
        # Fallback to a simple valid snake
        return [0, 1]
    
    return best_path

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
