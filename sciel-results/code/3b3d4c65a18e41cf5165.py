"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
M = len(P['edge_order'])

# Precompute edge positions
edge_pos = {}
for i, (a, b) in enumerate(P['edge_order'], 1):
    key = (min(a, b), max(a, b))
    edge_pos[key] = i

def get_score(cycle):
    """Compute the score: largest one-based arrival index of edges in the cycle."""
    max_pos = 0
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        key = (min(a, b), max(a, b))
        pos = edge_pos.get(key, 0)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Build adjacency with edge positions
    adj = [[] for _ in range(N)]  # adj[v] = list of (neighbor, edge_position)
    for i, (a, b) in enumerate(P['edge_order'], 1):
        adj[a].append((b, i))
        adj[b].append((a, i))
    
    # Sort neighbors by edge position (earlier edges first)
    for v in range(N):
        adj[v].sort(key=lambda x: x[1])
    
    best_cycle = None
    best_score = M + 1
    
    sys.setrecursionlimit(10000)
    
    def find_cycle_with_threshold(threshold, start=0, deadline=None):
        """Find Hamilton cycle using only edges with position <= threshold."""
        # Build filtered adjacency
        filt_adj = [[] for _ in range(N)]
        for v in range(N):
            for (nb, pos) in adj[v]:
                if pos <= threshold:
                    filt_adj[v].append(nb)
        
        # Check if all vertices have degree >= 2 (necessary for Hamilton cycle)
        for v in range(N):
            if len(filt_adj[v]) < 2:
                return None
        
        # DFS with backtracking and strong pruning
        visited = [False] * N
        path = [start]
        visited[start] = True
        
        # Precompute which vertices are connected to start (for closing the cycle)
        start_neighbors = set(filt_adj[start])
        
        def backtrack():
            nonlocal deadline
            if deadline and time.time() > deadline:
                return False
            if len(path) == N:
                # Check if we can close the cycle
                if start in filt_adj[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            
            # Pruning: if only 2 vertices unvisited, both must connect to each other and one to start
            if len(path) == N - 2:
                unvisited = [v for v in range(N) if not visited[v]]
                if len(unvisited) == 2:
                    u, w = unvisited
                    # Either path -> u -> w -> start or path -> w -> u -> start
                    valid_uw = u in filt_adj[w] or w in filt_adj[u]
                    if not valid_uw:
                        return False
                    # Check if we can go from last to one of them and the other to start
                    can_go_u = u in filt_adj[last] and w in start_neighbors
                    can_go_w = w in filt_adj[last] and u in start_neighbors
                    if not (can_go_u or can_go_w):
                        return False
                    # Only try valid neighbors
                    for neighbor in filt_adj[last]:
                        if not visited[neighbor]:
                            if (neighbor == u and w in start_neighbors) or (neighbor == w and u in start_neighbors):
                                visited[neighbor] = True
                                path.append(neighbor)
                                other = w if neighbor == u else u
                                if other in filt_adj[neighbor] and start in filt_adj[other]:
                                    return True
                                path.pop()
                                visited[neighbor] = False
                    return False
            
            # Pruning: check for isolated vertices (vertices with no unvisited neighbors and not in path)
            if len(path) >= 3:
                unvisited_count = N - len(path)
                if unvisited_count > 0:
                    for v in range(N):
                        if not visited[v] and v != start:
                            # v must have at least one neighbor in path (to eventually connect) 
                            # and at least one unvisited neighbor (or be the last one)
                            has_unvisited_neighbor = False
                            has_path_neighbor = False
                            for nb in filt_adj[v]:
                                if not visited[nb] and nb != v:
                                    has_unvisited_neighbor = True
                                elif visited[nb] and nb != start:
                                    has_path_neighbor = True
                            if unvisited_count > 1 and not has_unvisited_neighbor:
                                return False
                            if not has_path_neighbor and v not in start_neighbors:
                                # v can only connect to start, but start is already in path
                                # This means v must be the last vertex before closing
                                if unvisited_count > 1:
                                    return False
            
            # Try neighbors in order of edge position
            for neighbor in filt_adj[last]:
                if not visited[neighbor]:
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
    
    # Binary search for minimum threshold
    lo, hi = N, M
    best_threshold = M
    
    # First check if full graph works
    cycle_full = find_cycle_with_threshold(M, deadline=deadline)
    if cycle_full:
        best_cycle = cycle_full
        best_score = get_score(best_cycle)
        best_threshold = M
    else:
        # Should not happen for complete graph, but handle gracefully
        return list(range(N))
    
    # Binary search for minimum threshold
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle_mid = find_cycle_with_threshold(mid, deadline=deadline)
        if cycle_mid:
            score_mid = get_score(cycle_mid)
            if score_mid < best_score:
                best_score = score_mid
                best_cycle = cycle_mid
                best_threshold = mid
            hi = mid
        else:
            lo = mid + 1
    
    # Try to improve with local optimization - more aggressive version
    def local_optimize(cycle, max_iterations=100):
        """2-opt style local optimization with multiple passes."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 2, N):
                    if i == 0 and j == N - 1:
                        continue
                    # Try reversing segment from i+1 to j
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    new_score = get_score(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        cycle = new_cycle
                        improved = True
                        break
                if improved:
                    break
        return cycle
    
    if time.time() < deadline and best_cycle:
        best_cycle = local_optimize(best_cycle, max_iterations=50)
    
    # Try multiple starting vertices with the best threshold found
    if time.time() < deadline:
        for start in range(1, N):
            if time.time() > deadline:
                break
            cycle_s = find_cycle_with_threshold(best_threshold, start=start, deadline=deadline)
            if cycle_s:
                score_s = get_score(cycle_s)
                if score_s < best_score:
                    best_score = score_s
                    best_cycle = cycle_s
                    # Local optimize
                    if time.time() < deadline:
                        best_cycle = local_optimize(best_cycle, max_iterations=30)
    
    # Final aggressive local search: try all pairs of edge swaps
    if time.time() < deadline and best_cycle:
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(N):
                    if i == j or (i + 1) % N == j or (j + 1) % N == i:
                        continue
                    # Swap: remove edges (i, i+1) and (j, j+1), add (i, j) and (i+1, j+1)
                    a1, b1 = best_cycle[i], best_cycle[(i+1) % N]
                    a2, b2 = best_cycle[j], best_cycle[(j+1) % N]
                    new_e1 = (min(a1, b2), max(a1, b2))
                    new_e2 = (min(b1, a2), max(b1, a2))
                    pos1 = edge_pos.get(new_e1, M + 1)
                    pos2 = edge_pos.get(new_e2, M + 1)
                    if pos1 <= M and pos2 <= M:
                        # Create new cycle
                        new_cycle = best_cycle[:]
                        # This is a 2-opt move: reverse the segment between i and j
                        if i < j:
                            new_cycle = new_cycle[:i+1] + new_cycle[i+1:j+1][::-1] + new_cycle[j+1:]
                        else:
                            # Wrap around case
                            new_cycle = new_cycle[:i+1] + new_cycle[i+1:][::-1] + new_cycle[:j+1][::-1]
                        new_score = get_score(new_cycle)
                        if new_score < best_score:
                            best_score = new_score
                            best_cycle = new_cycle
                            improved = True
                            break
                if improved:
                    break
    
    return list(best_cycle)
# EVOLVE-BLOCK-END
