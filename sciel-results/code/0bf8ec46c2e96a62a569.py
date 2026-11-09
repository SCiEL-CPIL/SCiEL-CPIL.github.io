import time
import random
from collections import deque

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    total_edges = len(edge_order)
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Build position map
    pos_map = {}
    for i, (u, v) in enumerate(edge_order):
        key = (min(u, v), max(u, v))
        pos_map[key] = i

    def get_score(cycle):
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            key = (min(u, v), max(u, v))
            pos = pos_map[key] + 1
            if pos > max_pos:
                max_pos = pos
        return max_pos

    # Precompute adjacency for each prefix length (lazy cache)
    adj_cache = {}
    def get_adj(T):
        """Return adjacency list using edges with 0-based position <= T."""
        if T in adj_cache:
            return adj_cache[T]
        adj = [set() for _ in range(n)]
        for i in range(min(T + 1, total_edges)):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        adj_cache[T] = adj
        return adj

    def find_hamilton_cycle(T, rng, time_limit):
        """Randomized backtracking with strong pruning to find a Hamilton cycle
        in the subgraph of edges with 0-based position <= T."""
        adj = get_adj(T)
        
        # Quick degree check
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # Check vertex 0 has degree >= 2
        if len(adj[0]) < 2:
            return None

        visited = [False] * n
        visited[0] = True
        path = [0]
        path_set = {0}
        
        def backtrack():
            if time.time() - start_time > deadline:
                return None
            
            current = path[-1]
            remaining = n - len(path)
            
            if remaining == 0:
                # All vertices visited; check if we can close to 0
                if 0 in adj[current]:
                    return path[:]
                return None
            
            # Candidates: unvisited neighbors of current
            candidates = [v for v in adj[current] if not visited[v]]
            
            if not candidates:
                return None
            
            # If this is the last vertex to place, it must connect back to 0
            if remaining == 1:
                candidates = [v for v in candidates if 0 in adj[v]]
                if not candidates:
                    return None
            
            # Strong pruning: for each candidate v, check if placing v
            # would leave any unvisited vertex (other than possibly v itself)
            # with no available unvisited neighbors (dead-end).
            # Also check: if v is placed, every other unvisited vertex must
            # still have at least 2 unvisited neighbors (counting v as one
            # if adjacent) OR be adjacent to 0 (for the closing edge).
            
            def is_dead_end(v):
                """Check if placing v creates a dead-end for some other unvisited vertex."""
                # After placing v, v becomes visited.
                # For each other unvisited vertex w:
                #   w needs at least 2 connections to {visited path vertices that w is adjacent to} 
                #   plus unvisited neighbors... but actually for Hamilton path completion:
                #   w must have at least 2 neighbors in (path ∪ {v}) ∪ (unvisited \ {w})
                #   More simply: w must have at least 2 neighbors among (visited ∪ {v}) 
                #   that are NOT the single predecessor constraint... 
                #   
                # Simpler and effective check: after placing v, for each unvisited w != v:
                #   count neighbors of w in (visited ∪ {v})
                #   If count == 0 and w is not 0: dead end (w can't connect to path)
                #   If count == 1: w must have that neighbor, and needs one more from unvisited
                #     OR w == 0 (which needs 2 total: one from path, one from unvisited or closing)
                # 
                # Let's use: for each unvisited w (w != v, w != 0):
                #   neighbors_in_visited_plus_v = {x for x in adj[w] if visited[x] or x == v}
                #   if len(neighbors_in_visited_plus_v) == 0: return True (dead end)
                #   if len(neighbors_in_visited_plus_v) == 1:
                #       # w must use that one edge; needs another from unvisited
                #       unvisited_neighbors = {x for x in adj[w] if not visited[x] and x != v and x != w}
                #       if len(unvisited_neighbors) == 0: return True
                # 
                # For w == 0:
                #   neighbors_in_visited_plus_v = {x for x in adj[0] if visited[x] or x == v}
                #   if len(neighbors_in_visited_plus_v) == 0: return True
                #   # 0 needs 2 total in the cycle; one will be the closing edge
                #   # So 0 needs at least 1 from visited∪{v} and the other can be closing
                #   # Actually 0 already has some visited neighbors (at least the start)
                #   # 0's degree in final cycle is 2: one to path[0]=0's first neighbor, one to last
                #   # Since 0 is already in path, its two cycle edges are: to path[1] and to last vertex
                #   # path[1] is already fixed. So 0 needs exactly one more edge to the last vertex.
                #   # So 0 just needs at least 1 neighbor in visited∪{v} that isn't path[1]... 
                #   # This is getting complex. Let's use a simpler check for 0.
                #   pass
                
                for w in range(n):
                    if visited[w] or w == v:
                        continue
                    # Count neighbors of w that are in visited or equal to v
                    cnt = 0
                    for x in adj[w]:
                        if visited[x] or x == v:
                            cnt += 1
                    if w == 0:
                        # 0 is the start; it needs its closing edge to the last vertex
                        # It already has one edge to path[1] (if path length > 1)
                        # So it needs at least 1 more neighbor in visited∪{v} (the closing)
                        # But actually the closing edge is to the LAST vertex, which is unvisited
                        # So 0 needs: one edge to path[1] (already in visited) + one edge to last (unvisited)
                        # The "one more" for 0 is the closing edge, which goes to an unvisited vertex
                        # So 0 doesn't need extra visited neighbors beyond path[1]
                        # Skip 0 for this check
                        continue
                    else:
                        if cnt == 0:
                            return True
                        if cnt == 1:
                            # Needs one more from unvisited (not v, not w)
                            unv = 0
                            for x in adj[w]:
                                if not visited[x] and x != v and x != w:
                                    unv += 1
                            if unv == 0:
                                return True
                return False
            
            # Sort candidates: prefer those with fewer unvisited neighbors (most constrained)
            def candidate_key(v):
                unv_deg = 0
                for w in adj[v]:
                    if not visited[w] and w != v:
                        unv_deg += 1
                return unv_deg
            
            candidates.sort(key=candidate_key)
            
            # Shuffle among equal-constraint candidates for randomization
            # Group by key and shuffle within groups
            if len(candidates) > 1:
                # Partial shuffle: swap random pairs
                for i in range(len(candidates) - 1, 0, -1):
                    if rng.random() < 0.3:
                        j = rng.randint(0, i)
                        candidates[i], candidates[j] = candidates[j], candidates[i]
            
            for v in candidates:
                # Dead-end pruning
                if remaining > 1 and is_dead_end(v):
                    continue
                
                # Additional pruning: if v's only unvisited neighbor (besides current) 
                # is a vertex that would be stranded
                visited[v] = True
                path.append(v)
                
                result = backtrack()
                if result is not None:
                    return result
                
                visited[v] = False
                path.pop()
            
            return None
        
        return backtrack()

    # Multi-start search across different edge thresholds
    # Strategy: try thresholds in a smart order, with multiple random restarts per threshold
    
    best_cycle = None
    best_score = total_edges + 1
    
    # First, get a quick upper bound using full graph
    full_adj = get_adj(total_edges - 1)
    
    # Quick greedy to get initial solution
    def greedy_initial():
        visited = [False] * n
        visited[0] = True
        path = [0]
        while len(path) < n:
            current = path[-1]
            best_v = None
            best_pos = float('inf')
            for v in full_adj[current]:
                if not visited[v]:
                    pos = pos_map[(min(current, v), max(current, v))]
                    if pos < best_pos:
                        best_pos = pos
                        best_v = v
            if best_v is None:
                return None
            visited[best_v] = True
            path.append(best_v)
        if 0 in full_adj[path[-1]]:
            return path
        return None
    
    gc = greedy_initial()
    if gc is not None:
        s = get_score(gc)
        best_score = s
        best_cycle = gc

    # Determine search range
    # Lower bound: need at least n edges (24), but also need min degree >= 2
    # Find smallest T where all vertices have degree >= 2
    min_T_for_degree2 = 0
    for T in range(23, total_edges):
        adj = get_adj(T)
        if all(len(adj[i]) >= 2 for i in range(n)):
            min_T_for_degree2 = T
            break
    else:
        min_T_for_degree2 = total_edges - 1
    
    # Search thresholds from low to high, with multiple random attempts
    # Use a priority: try lower thresholds first (they give better scores)
    search_thresholds = list(range(min_T_for_degree2, min(best_score - 1, total_edges)))
    
    # If we have a good upper bound, focus search there
    if best_score < total_edges:
        # Prioritize thresholds near the current best
        search_thresholds = sorted(search_thresholds, key=lambda T: abs(T - (best_score - 1)))
    
    # Multi-start: for each threshold, try multiple random restarts
    rng = random.Random(42)
    attempts_per_threshold = 5
    max_total_attempts = 500
    
    total_attempts = 0
    for T in search_thresholds:
        if time.time() > deadline - 5:
            break
        if total_attempts >= max_total_attempts:
            break
        
        found = False
        for attempt in range(attempts_per_threshold):
            if time.time() > deadline - 5:
                break
            if total_attempts >= max_total_attempts:
                break
            
            total_attempts += 1
            cycle = find_hamilton_cycle(T, rng, time_limit=0)
            if cycle is not None:
                s = get_score(cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = cycle
                    found = True
                    # Found at this threshold, try slightly lower
                    # But first, let's see if we can do even better at this T
                    break
        
        if found and best_score - 1 <= min_T_for_degree2:
            break
        
        # If we found a cycle at T, we know T works. Try T-1 next.
        # The sorted order should handle this naturally.
    
    # Final pass: try to improve by checking thresholds just below current best
    if best_cycle is not None and best_score > min_T_for_degree2 + 1:
        for T in range(min_T_for_degree2, best_score - 1):
            if time.time() > deadline - 3:
                break
            cycle = find_hamilton_cycle(T, rng, time_limit=0)
            if cycle is not None:
                s = get_score(cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = cycle
    
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle
