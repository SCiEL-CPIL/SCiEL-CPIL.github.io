import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_n = P['n']
_edge_order = P['edge_order']

# Precompute edge position map (1-based)
_edge_pos = {}
for i, (u, v) in enumerate(_edge_order, 1):
    _edge_pos[(u, v)] = i
    _edge_pos[(v, u)] = i

# For each vertex, list of (neighbor, edge_position) sorted by position
_adj = [None] * _n
for v in range(_n):
    lst = []
    for u in range(_n):
        if u != v:
            pos = _edge_pos[(v, u)]
            lst.append((pos, u))
    lst.sort()
    _adj[v] = lst

def _exists_ham_cycle_with_threshold(T, deadline):
    """
    Check if there exists a Hamilton cycle where all edges have position <= T.
    Returns the cycle if found, else None.
    Uses backtracking with strong pruning.
    """
    # Build adjacency lists for threshold T
    adj_T = [set() for _ in range(_n)]
    for v in range(_n):
        for pos, u in _adj[v]:
            if pos <= T:
                adj_T[v].add(u)
            else:
                break
    
    # Check minimum degree - each vertex must have degree >= 2
    for v in range(_n):
        if len(adj_T[v]) < 2:
            return None
    
    # Try different starting vertices
    for start in range(_n):
        if time.time() > deadline:
            return None
        
        path = [start]
        used = [False] * _n
        used[start] = True
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == _n:
                # Check closing edge
                last = path[-1]
                first = path[0]
                if first in adj_T[last]:
                    return path[:]
                return None
            
            last = path[-1]
            # Sort candidates by some heuristic - prefer vertices with fewer options
            candidates = list(adj_T[last] - set(used))
            
            # Additional pruning: check if any unused vertex (except possibly last) 
            # has only one available neighbor that isn't used yet
            # This is a "forced move" heuristic
            
            # Sort by degree in remaining graph (fewer options first)
            def candidate_key(v):
                remaining_neighbors = adj_T[v] - set(used)
                # If v is not the last to be placed, it needs at least 2 neighbors
                # (one for incoming, one for outgoing) unless it's the final vertex
                return len(remaining_neighbors)
            
            candidates.sort(key=candidate_key)
            
            for nxt in candidates:
                if nxt not in adj_T[last]:
                    continue
                
                # Pruning: if we're placing the second-to-last vertex,
                # check that the remaining vertex can close the cycle
                if len(path) == _n - 2:
                    # Find the one remaining unused vertex
                    remaining = [v for v in range(_n) if not used[v] and v != nxt]
                    if len(remaining) != 1:
                        continue
                    rem_v = remaining[0]
                    # Check: nxt -> rem_v exists AND rem_v -> start exists
                    if rem_v not in adj_T[nxt] or start not in adj_T[rem_v]:
                        continue
                
                # Pruning: check if any unused vertex will be isolated
                # (only has connections to used vertices or to itself)
                isolated = False
                for v in range(_n):
                    if not used[v] and v != nxt:
                        # v needs at least 2 neighbors in the remaining set
                        # (including nxt if adjacent)
                        avail = adj_T[v] - set(used)
                        # After placing nxt, v's available neighbors shrink
                        avail_after = avail - {nxt}
                        # v needs at least 1 neighbor among unused (excluding itself)
                        # plus possibly nxt
                        # Actually, v needs degree >= 2 in the final cycle
                        # In the remaining path, v needs at least 2 connections
                        # This is complex; simple check: at least 1 unused neighbor
                        if len(avail) < 1:
                            isolated = True
                            break
                
                if isolated:
                    continue
                
                path.append(nxt)
                used[nxt] = True
                result = backtrack()
                if result is not None:
                    return result
                used[nxt] = False
                path.pop()
            
            return None
        
        result = backtrack()
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    # Get baseline score from identity cycle
    base_score = float('inf')
    for i in range(_n):
        u = i
        v = (i + 1) % _n
        base_score = max(base_score, _edge_pos[(u, v)])
    
    best_cycle = list(range(_n))
    best_score = base_score
    
    # Binary search on threshold T
    # Lower bound: at least n edges needed, so T >= n (but realistically higher)
    # Upper bound: best_score (we can always do at least as well as baseline)
    lo = _n  # minimum possible (24 edges, but positions are 1..276)
    hi = best_score
    
    # First, try to find a cycle with a much lower threshold
    # Use binary search
    best_T = hi
    best_cycle_found = None
    
    # Try a few key thresholds first to get a good upper bound quickly
    # The theoretical minimum for a Hamilton cycle on 24 vertices is 24 edges
    # but the positions depend on the specific edge ordering
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _exists_ham_cycle_with_threshold(mid, deadline)
        if cycle is not None:
            best_T = mid
            best_cycle_found = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle_found is not None:
        best_cycle = best_cycle_found
        best_score = best_T
    
    return best_cycle
# EVOLVE-BLOCK-END
