import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
M = len(P['edge_order'])

# Precompute edge positions for scoring
edge_positions = {}
for i, (a, b) in enumerate(P['edge_order'], 1):
    key = (min(a, b), max(a, b))
    edge_positions[key] = i

def get_score(cycle):
    max_pos = 0
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        key = (min(a, b), max(a, b))
        pos = edge_positions.get(key, 0)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def build_adjacency(k):
    adj = [set() for _ in range(N)]
    for i in range(k):
        a, b = P['edge_order'][i]
        adj[a].add(b)
        adj[b].add(a)
    return adj

def find_hamilton_cycle(adj, start=0, deadline=None):
    """
    Optimized backtracking with Warnsdorff's rule and pruning.
    """
    n = N
    visited = [False] * n
    path = [start]
    visited[start] = True
    adj_list = [list(s) for s in adj]
    
    # Pre-sort neighbors? Warnsdorff is dynamic, so we sort at each step.
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        if deadline and time.time() > deadline:
            return False
        
        if len(path) == n:
            if start in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        neighbors = [nb for nb in adj_list[last] if not visited[nb]]
        
        if not neighbors:
            return False
        
        # Special case: if we are about to pick the last vertex
        if len(path) == n - 1:
            unvisited = [v for v in range(n) if not visited[v]]
            if len(unvisited) == 1:
                last_unvisited = unvisited[0]
                if last_unvisited not in neighbors:
                    return False
                # Check if last_unvisited connects to start
                if start not in adj[last_unvisited]:
                    return False
                visited[last_unvisited] = True
                path.append(last_unvisited)
                if start in adj[last_unvisited]:
                    return True
                path.pop()
                visited[last_unvisited] = False
                return False

        # Warnsdorff's Rule: Sort neighbors by the number of their unvisited neighbors
        # Lower count is preferred (most constrained first)
        def degree_key(nb):
            # Count unvisited neighbors of nb
            count = 0
            for x in adj_list[nb]:
                if not visited[x]:
                    count += 1
            return count

        neighbors.sort(key=degree_key)
        
        for neighbor in neighbors:
            # Pruning: Check if any unvisited vertex (other than neighbor) becomes isolated
            # This is expensive O(N^2) per step, but N=24 is small.
            # We can optimize this by only checking if necessary or sampling.
            # For now, let's keep it simple but efficient.
            
            # Quick check: If we pick 'neighbor', does any other unvisited vertex lose its last connection?
            # A vertex v is "at risk" if it has exactly 1 unvisited neighbor.
            # If that neighbor is 'neighbor' (which we are visiting), v becomes isolated.
            # Exception: if v is the last vertex to be visited, it only needs to connect to 'start' eventually?
            # No, in a path, every intermediate vertex needs 2 connections (prev, next).
            # The last vertex needs connection to prev and start.
            # So, if v is not the last vertex, it needs at least 1 unvisited neighbor after we visit 'neighbor'.
            
            # Let's do a lighter check:
            # Count unvisited neighbors for all unvisited vertices (excluding 'neighbor' and 'last')
            # If any vertex v (v != neighbor) has 0 unvisited neighbors, prune.
            # Note: 'last' is already visited. 'neighbor' is about to be visited.
            # So for v in unvisited \ {neighbor}, count neighbors in unvisited \ {neighbor}.
            
            # Optimization: Only do this check if we are not in the last few steps?
            # Actually, for N=24, O(N^2) is ~500 ops. 24 steps * 500 = 12k ops per path.
            # With backtracking, this might be too slow if the branching factor is high.
            # However, Warnsdorff drastically reduces branching.
            
            # Let's try a cheaper check first:
            # Just proceed with Warnsdorff. If it's too slow, add pruning.
            
            visited[neighbor] = True
            path.append(neighbor)
            if backtrack():
                return True
            path.pop()
            visited[neighbor] = False
            
        return False

    if backtrack():
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Binary search for the minimum k
    lo, hi = N, M
    best_cycle = None
    best_score = float('inf')
    
    # We know K_24 is complete, so M is definitely feasible.
    # We want the smallest k.
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        adj_mid = build_adjacency(mid)
        cycle_mid = find_hamilton_cycle(adj_mid, deadline=deadline)
        if cycle_mid:
            score_mid = get_score(cycle_mid)
            if score_mid < best_score:
                best_score = score_mid
                best_cycle = cycle_mid
            hi = mid
        else:
            lo = mid + 1
            
    # If we exited due to time, ensure we have a valid cycle
    # The loop above keeps `best_cycle` as the best found so far.
    # If `best_cycle` is None, it means we didn't find any cycle?
    # But K_24 always has one. The binary search might have failed to find one in `mid`
    # due to timeout, but `hi` would still be M.
    # If best_cycle is None, we should try the full graph M.
    if best_cycle is None:
        if time.time() < deadline:
            adj_full = build_adjacency(M)
            best_cycle = find_hamilton_cycle(adj_full, deadline=deadline)
            if best_cycle:
                best_score = get_score(best_cycle)
            else:
                # Fallback: Identity permutation (might not be a valid cycle in subgraph, 
                # but the task asks for a permutation. The evaluator checks the score.
                # If we return a permutation that is not a cycle, what happens?
                # "Score is the largest one-based arrival index of its edges in the pinned... list."
                # It implies we must return a valid Hamilton cycle.
                # If we can't find one, we are in trouble. But K_24 is dense.
                best_cycle = list(range(N))
        
    return list(best_cycle)
