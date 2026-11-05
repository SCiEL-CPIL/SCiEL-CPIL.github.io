"""Optimized Hamilton cycle finder using binary search + backtracking with strong pruning."""
import sys
import time
sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_order = P['edge_order']

# Build edge index: edge -> position (1-based)
edge_pos = {}
for i, (u, v) in enumerate(edge_order):
    if u > v:
        u, v = v, u
    edge_pos[(u, v)] = i + 1

def get_edge_pos(u, v):
    if u > v:
        u, v = v, u
    return edge_pos[(u, v)]

# Precompute adjacency lists sorted by edge position
adj_by_pos = [[] for _ in range(N)]
for i, (u, v) in enumerate(edge_order):
    pos = i + 1
    adj_by_pos[u].append((pos, v))
    adj_by_pos[v].append((pos, u))
for i in range(N):
    adj_by_pos[i].sort(key=lambda x: x[0])

def find_ham_cycle(threshold):
    """Find a Hamiltonian cycle using only edges with position <= threshold."""
    # Build allowed adjacency
    allowed = [set() for _ in range(N)]
    for i in range(N):
        for pos, v in adj_by_pos[i]:
            if pos <= threshold:
                allowed[i].add(v)
            else:
                break
    
    # Check degree >= 2 for all vertices
    for i in range(N):
        if len(allowed[i]) < 2:
            return None
    
    # Check connectivity of the graph
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in allowed[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    if count < N:
        return None
    
    # Precompute sorted neighbor lists for faster iteration
    allowed_sorted = [sorted(allowed[i]) for i in range(N)]
    
    # Backtracking: build path from vertex 0
    path = [0]
    used = [False] * N
    used[0] = True
    
    def check_feasibility():
        """Check if the current partial path can be extended to a Hamilton cycle."""
        # Check that all unused vertices have at least 2 available neighbors 
        # (considering they need to connect within the remaining set + path endpoints)
        remaining = [v for v in range(N) if not used[v]]
        if not remaining:
            # Last vertex must connect to 0
            return 0 in allowed[path[-1]]
        
        # For each remaining vertex, count how many neighbors are in remaining or are path endpoints
        # A vertex in the final cycle needs degree 2
        # Internal vertices in remaining need 2 neighbors in remaining
        # But some might connect to path endpoints
        
        # Simpler check: each remaining vertex must have at least 2 neighbors 
        # in (remaining ∪ {path[0], path[-1]})
        endpoints = {path[0], path[-1]}
        for v in remaining:
            avail = 0
            for u in allowed[v]:
                if not used[u] or u in endpoints:
                    avail += 1
            if avail < 2:
                return False
        
        # Check connectivity of remaining vertices + endpoints
        # The remaining vertices must form a connected subgraph when we include 
        # edges between them and to the two endpoints
        if len(remaining) == 1:
            v = remaining[0]
            # v must connect to both endpoints (or at least be able to complete the cycle)
            # v connects to path[-1] (next in path) and to path[0] (closing)
            # So v must be adjacent to path[0]
            if path[0] not in allowed[v]:
                return False
            return True
        
        # BFS/DFS on remaining vertices using edges within remaining
        rem_set = set(remaining)
        rem_visited = [False] * N
        rem_stack = [remaining[0]]
        rem_visited[remaining[0]] = True
        rem_count = 1
        while rem_stack:
            u = rem_stack.pop()
            for v in allowed[u]:
                if v in rem_set and not rem_visited[v]:
                    rem_visited[v] = True
                    rem_count += 1
                    rem_stack.append(v)
        
        # All remaining vertices must be connected to each other OR 
        # connected through the endpoints
        # Actually, the remaining vertices must form a path from path[-1] to path[0]
        # So they need to be connected when we consider the two endpoints as part of the graph
        # Let's check: can we reach all remaining vertices from path[-1] using edges 
        # that go through remaining vertices?
        
        # More precisely: the subgraph induced by remaining ∪ {path[-1]} must have 
        # all remaining vertices reachable from path[-1], AND path[0] must be reachable 
        # from some remaining vertex (for the closing edge)
        
        # Simpler: check that remaining vertices form a connected component 
        # in the graph where we also allow edges to endpoints
        # Actually the key constraint is that the remaining vertices must form 
        # a single path from path[-1] to path[0]
        
        # Connectivity check: remaining vertices must be connected among themselves
        # (since they form a path, they must be connected)
        if rem_count < len(remaining):
            # Not all remaining are connected among themselves
            # But they could still form a path through... no, a path is connected
            return False
        
        # Check that path[-1] can reach at least one remaining vertex
        # and path[0] can reach at least one remaining vertex
        # (for the path to start and end)
        can_start = False
        can_end = False
        for v in remaining:
            if path[-1] in allowed[v]:
                can_start = True
            if path[0] in allowed[v]:
                can_end = True
        if not can_start or not can_end:
            return False
        
        return True
    
    def backtrack():
        if len(path) == N:
            # Check if last vertex connects back to 0
            last = path[-1]
            if 0 in allowed[last]:
                return True
            return False
        
        current = path[-1]
        # Candidates: neighbors of current that are not used
        candidates = []
        for v in allowed_sorted[current]:
            if not used[v]:
                candidates.append(v)
        
        if not candidates:
            return False
        
        # Sort by degree in remaining graph (fewest options first) - MRV heuristic
        # For each candidate, count how many unused neighbors it has
        def score_candidate(v):
            cnt = 0
            for u in allowed_sorted[v]:
                if not used[u]:
                    cnt += 1
            return cnt
        
        candidates.sort(key=score_candidate)
        
        for v in candidates:
            if len(path) == N - 1:
                # v is the last vertex, must connect to 0
                if 0 not in allowed[v]:
                    continue
                path.append(v)
                used[v] = True
                if backtrack():
                    return True
                path.pop()
                used[v] = False
            else:
                # v is internal, needs to be extendable
                # Check that v has at least one other unused neighbor
                other_neighbors = [u for u in allowed_sorted[v] if u != current and not used[u]]
                if len(other_neighbors) < 1:
                    continue
                
                path.append(v)
                used[v] = True
                
                # Feasibility check (only for deeper levels to save time)
                if len(path) >= 4:
                    if not check_feasibility():
                        path.pop()
                        used[v] = False
                        continue
                
                if backtrack():
                    return True
                path.pop()
                used[v] = False
        
        return False
    
    if backtrack():
        return path
    return None

# Binary search for minimum threshold
lo, hi = 1, len(edge_order)
best_cycle = None
start_time = time.time()

while lo <= hi:
    if time.time() - start_time > 50:  # Leave 10s buffer
        break
    mid = (lo + hi) // 2
    cycle = find_ham_cycle(mid)
    if cycle is not None:
        best_cycle = cycle
        hi = mid - 1
    else:
        lo = mid + 1

# If we ran out of time, try a few more values near the best found
if best_cycle is not None and lo <= hi and time.time() - start_time < 55:
    for t in range(max(1, hi), min(lo, hi + 5)):
        if time.time() - start_time > 55:
            break
        cycle = find_ham_cycle(t)
        if cycle is not None:
            best_cycle = cycle
            break

def construct():
    if best_cycle is not None:
        return best_cycle
    # Fallback: identity permutation
    return list(range(N))
# EVOLVE-BLOCK-END
