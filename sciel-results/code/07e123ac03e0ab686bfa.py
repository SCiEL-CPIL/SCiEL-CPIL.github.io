import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

def _is_valid_extension(v, path, used):
    """Check if v can be appended to path."""
    if v in used:
        return False
    # Check non-consecutive vertices: all in path except the last one (current)
    # Since v is a neighbor of current, dist(v, current) == 1, which is fine for consecutive.
    # But we must ensure dist(v, p) >= 2 for all p in path[:-1] (which includes current? No.)
    # Wait, path is the list of vertices. path[-1] is the current vertex.
    # The new vertex v will be path[-1].
    # The constraint is: for all i < len(path)-1, dist(path[i], v) >= 2.
    # path[:-1] contains path[0] ... path[len-2].
    # path[len-1] is the current vertex. v is a neighbor of path[len-1], so dist(v, path[len-1]) == 1.
    # We do NOT check dist(v, path[len-1]) because they are consecutive.
    # So we check against path[:-1].
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Greedy search with lookahead.
    Returns a list of vertices forming a valid snake.
    """
    random.seed(seed_offset)
    path = [0]
    used = {0}
    current = 0
    
    # Precompute neighbors for speed? 
    # N=9, so 9 neighbors per vertex. 
    # Total vertices 512. 
    # We can precompute a list of neighbors for all 512 vertices.
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for b in range(N):
            v = current ^ (1 << b)
            if _is_valid_extension(v, path, used):
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        # This is O(N * N * len(path)) per step.
        # len(path) can be up to ~200. N=9.
        # 9 * 9 * 200 = 16200 ops per step.
        # 200 steps * 16200 = 3.2M ops.
        # This is fast enough in Python if optimized.
        
        scored_neighbors = []
        for v in valid_neighbors:
            # Count valid neighbors of v, assuming v is added to path
            # New path: path + [v]
            # New used: used + {v}
            # We need to count u in neighbors(v) such that:
            # 1. u not in used and u != v (v is in used now)
            # 2. dist(u, p) >= 2 for all p in (path + [v])[:-1]
            #    (path + [v])[:-1] is path.
            #    So we check dist(u, p) >= 2 for all p in path.
            
            count = 0
            for b2 in range(N):
                u = v ^ (1 << b2)
                if u in used or u == v:
                    continue
                # Check against all p in path
                valid = True
                for p in path:
                    if (u ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    count += 1
            scored_neighbors.append((count, v))
            
        # Sort by count descending, pick randomly from top ties
        # To avoid full sort, find max count and collect ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = random.choice(candidates)
        
        path.append(next_v)
        used.add(next_v)
        current = next_v
        
    return path

def _local_improvement(path):
    """
    Try to improve the path by swapping or inserting?
    For snakes, local search is tricky. 
    A common move is to take a dead-end and try to extend from an earlier point?
    Or "backtrack and try a different branch".
    
    Given time constraints, we'll stick to multiple randomized greedy runs.
    We can also try to "repair" a path if it gets stuck by checking if a non-consecutive 
    vertex can be jumped to? No, that breaks the path.
    
    Let's try a different heuristic: 
    Instead of just counting valid neighbors, we can also consider the "degree" of the 
    neighbor in the remaining graph? 
    The current lookahead does exactly that (count of valid neighbors).
    
    Let's add a second phase: if we find a long path, try to extend it from the beginning?
    No, the path is directed.
    
    Let's just run many greedy searches with different seeds.
    """
    return path

def construct():
    start_time = time.time()
    best_path = []
    
    # Precompute neighbors for all 512 vertices to speed up _is_valid_extension?
    # _is_valid_extension is called frequently.
    # Let's optimize _is_valid_extension by precomputing bit counts for XOR?
    # Or just rely on .bit_count() which is fast in Python 3.10+.
    
    # Initial deterministic greedy (lowest bit)
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    best_path = path
    best_len = len(best_path)
    
    # Randomized greedy searches
    i = 0
    while time.time() - start_time < MAX_TIME:
        p = _greedy_search(i)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        i += 1
        
        # If we have a lot of time left, we could do more sophisticated search.
        # But for now, just keep running greedy.
        
    return best_path
