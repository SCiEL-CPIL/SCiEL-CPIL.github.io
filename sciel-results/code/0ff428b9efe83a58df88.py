import random
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Precompute adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Convert to list for faster iteration if needed, but set is good for membership
# Let's also have a list of neighbors for each node for faster iteration in some cases
adj_list = [list(s) for s in adj]

def is_valid_path(path):
    if not path:
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in adj[path[i]]:
            return False
    return True

def greedy_build(start, rng):
    """Build a path starting from 'start' using Warnsdorff's rule."""
    path = [start]
    visited = {start}
    
    while True:
        current = path[-1]
        candidates = [n for n in adj_list[current] if n not in visited]
        if not candidates:
            break
        
        # Warnsdorff: pick neighbor with fewest unvisited neighbors
        def score(node):
            return sum(1 for nb in adj_list[node] if nb not in visited)
        
        # Add slight randomness to break ties or explore
        best_score = min(score(n) for n in candidates)
        # Pick randomly among those with min score to allow exploration
        min_candidates = [n for n in candidates if score(n) == best_score]
        next_node = rng.choice(min_candidates)
        
        path.append(next_node)
        visited.add(next_node)
        
    return path

def try_insertions(path, rng):
    """Try to insert unvisited vertices into the path."""
    visited = set(path)
    unvisited = [i for i in range(N) if i not in visited]
    
    improved = True
    while improved:
        improved = False
        # Shuffle unvisited to break symmetry
        rng.shuffle(unvisited)
        
        for u in unvisited:
            # Check if u can be inserted between any consecutive pair in path
            # path = [v0, v1, ..., vk]
            # Check insertion between vi and vi+1
            for i in range(len(path) - 1):
                v_i = path[i]
                v_ip1 = path[i+1]
                
                if u in adj[v_i] and u in adj[v_ip1]:
                    # Insert u
                    path.insert(i+1, u)
                    visited.add(u)
                    unvisited.remove(u)
                    improved = True
                    break # Restart inner loop as path changed
            
            if not unvisited:
                break
                
    return path

def two_opt(path, rng):
    """Apply a random 2-opt move if valid."""
    if len(path) < 4:
        return path
    
    i, j = rng.sample(range(1, len(path) - 1), 2)
    if i > j:
        i, j = j, i
        
    # Segment to reverse: path[i:j+1]
    # Check connectivity at boundaries
    # Left boundary: path[i-1] and path[j] must be connected
    # Right boundary: path[i] and path[j+1] must be connected
    
    if i > 0:
        if path[j] not in adj[path[i-1]]:
            return path
    if j < len(path) - 1:
        if path[i] not in adj[path[j+1]]:
            return path
            
    # Perform reversal
    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
    return new_path

def local_search(initial_path, rng, max_iters=1000):
    """Perform local search starting from initial_path."""
    path = initial_path[:]
    visited = set(path)
    unvisited = [i for i in range(N) if i not in visited]
    
    for _ in range(max_iters):
        if not unvisited:
            break
            
        # Try insertions first
        new_path = try_insertions(path, rng)
        if len(new_path) > len(path):
            path = new_path
            visited = set(path)
            unvisited = [i for i in range(N) if i not in visited]
            if not unvisited:
                break
            continue
        
        # Try 2-opt
        candidate = two_opt(path, rng)
        if candidate != path:
            path = candidate
            # After 2-opt, try insertions again as it might open up possibilities
            new_path = try_insertions(path, rng)
            if len(new_path) > len(path):
                path = new_path
                visited = set(path)
                unvisited = [i for i in range(N) if i not in visited]
                if not unvisited:
                    break
                    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit for safety
    rng = random.Random(748) # Use fixed seed for reproducibility of the search strategy itself
    
    best_path = []
    
    # Initial greedy runs from all nodes to find a decent starting point
    # But we will do random restarts
    
    # First, do a quick greedy from each node to find a long initial path
    for start in range(N):
        if time.time() > deadline:
            break
        p = greedy_build(start, rng)
        if len(p) > len(best_path):
            best_path = p
            
    # Local search on the best path found so far
    if best_path:
        improved = local_search(best_path, rng, max_iters=200)
        if len(improved) > len(best_path):
            best_path = improved

    # Multi-start random local search
    # We will try many random starts
    while time.time() < deadline:
        # Pick a random start
        start = rng.randint(0, N-1)
        
        # Greedy build
        current_path = greedy_build(start, rng)
        
        # Local search
        improved_path = local_search(current_path, rng, max_iters=500)
        
        if len(improved_path) > len(best_path):
            best_path = improved_path
            
        # Optional: If we found a very long path, we might want to polish it more
        # But the loop continues.
        
    # Final validation
    if not is_valid_path(best_path):
        # Fallback to a simple valid path if something went wrong
        # Find any edge
        for u, v in P['edges']:
            if u not in best_path and v not in best_path:
                return [u, v]
        if len(best_path) == 0:
            return [0]
            
    return best_path
# EVOLVE-BLOCK-END
