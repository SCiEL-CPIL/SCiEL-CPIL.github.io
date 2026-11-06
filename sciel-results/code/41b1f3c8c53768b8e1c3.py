import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Precompute arrival index for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def has_hamiltonian_cycle(adj, deadline):
    """
    Checks if the graph defined by adj (list of sets of neighbors) has a Hamiltonian cycle.
    Returns the cycle as a list of vertices if found, else None.
    Uses backtracking with pruning.
    """
    n = len(adj)
    if n < 3:
        return None
    
    # Degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
            
    # Connectivity check
    visited = [False] * n
    dq = deque([0])
    visited[0] = True
    count = 0
    while dq:
        u = dq.popleft()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                dq.append(w)
    if count != n:
        return None
        
    # 2-Connectivity check (Articulation Points)
    # If there is an articulation point, removing it disconnects the graph,
    # so no Hamiltonian cycle can exist (which requires 2-connectivity).
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    ap = [False] * n
    time_counter = [0]
    
    def tarjan(u):
        nonlocal time_counter
        disc[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                children += 1
                parent[v] = u
                tarjan(v)
                low[u] = min(low[u], low[v])
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
                
    tarjan(0)
    if any(ap):
        return None

    # Backtracking
    # Heuristic: Start with vertex 0
    start = 0
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Precompute neighbor lists sorted by degree (Warnsdorff's heuristic)
    # Actually, dynamic sorting is better but slower. Let's try static first, or simple dynamic.
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            if start in adj[last]:
                return path[:]
            return None
        
        cur = path[-1]
        # Get unvisited neighbors
        candidates = []
        for nb in adj[cur]:
            if not visited[nb]:
                # Count unvisited neighbors of nb (onward moves)
                onward = 0
                for w in adj[nb]:
                    if not visited[w]:
                        onward += 1
                # If it's the last step, nb must connect to start
                if len(path) == n - 1:
                    if start not in adj[nb]:
                        continue
                candidates.append((onward, nb))
                
        # Sort by fewest onward moves (Warnsdorff)
        candidates.sort()
        
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            # Pruning: Check if the remaining unvisited vertices are connected to the current path
            # Simple check: Are all unvisited vertices reachable from nb within the unvisited set + nb?
            # This is expensive. Let's do a cheaper check: 
            # If any unvisited vertex has degree 0 in the subgraph induced by unvisited vertices + {nb, cur...}, it's stuck.
            # Actually, just checking connectivity of the unvisited set + {nb} is a good prune.
            # But it's O(N). Let's do it only if the set is small or periodically?
            # For N=24, O(N) is fine.
            
            unvisited_mask = 0
            for i in range(n):
                if not visited[i]:
                    unvisited_mask |= (1 << i)
            
            # BFS from nb in the subgraph of unvisited vertices + nb
            # Note: We are at nb. We can move to any unvisited neighbor of nb.
            # The "frontier" is nb. The "target" is all unvisited vertices.
            # Wait, the path continues from nb. So nb is "visited" in the sense that we are there.
            # The remaining vertices are unvisited.
            # We need to ensure that all unvisited vertices are reachable from nb using only unvisited vertices (plus nb as start).
            # Actually, the edges between unvisited vertices must allow a path covering all of them.
            # A necessary condition: The subgraph induced by unvisited vertices must be connected? No, it can be disconnected if we can jump via visited vertices? No, we can't jump via visited vertices because we can't revisit them.
            # So, the unvisited vertices must form a connected component in the graph where we remove visited vertices?
            # No, that's not right. We are at nb. We will visit some unvisited vertex x. Then from x, we go to y...
            # The sequence of unvisited vertices must form a path in the subgraph induced by unvisited vertices.
            # So, the subgraph induced by unvisited vertices must be connected? Not necessarily, but if it's disconnected, we can only visit one component.
            # So, if the unvisited vertices are not connected in the induced subgraph, we can't visit them all in one go.
            # Wait, we are at nb (which is visited). We pick an unvisited neighbor x. Then we are at x.
            # From x, we can only go to unvisited neighbors.
            # So yes, the unvisited vertices must be connected in the induced subgraph?
            # Actually, if the induced subgraph is disconnected, we can only visit the component containing the first unvisited vertex we pick.
            # So, if the induced subgraph of unvisited vertices is not connected, we fail?
            # Not necessarily. What if we enter a component, visit all of it, and then... we can't leave it because we can't go back to visited vertices.
            # So, yes, the unvisited vertices must be connected in the induced subgraph.
            
            # Let's verify this prune.
            # If unvisited vertices are disconnected, we can only visit one component. Since we must visit ALL, we fail.
            # So, check connectivity of the subgraph induced by unvisited vertices.
            
            if unvisited_mask:
                # Pick a start vertex from unvisited
                start_unvisited = -1
                for i in range(n):
                    if (unvisited_mask >> i) & 1:
                        start_unvisited = i
                        break
                
                seen = 0
                dq = deque([start_unvisited])
                seen |= (1 << start_unvisited)
                
                while dq:
                    u = dq.popleft()
                    for w in adj[u]:
                        if (unvisited_mask >> w) & 1:
                            if not (seen >> w) & 1:
                                seen |= (1 << w)
                                dq.append(w)
                                
                if seen != unvisited_mask:
                    path.pop()
                    visited[nb] = False
                    continue

            res = backtrack()
            if res:
                return res
            
            path.pop()
            visited[nb] = False
            
        return None
        
    return backtrack()

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin
    
    # Binary search for the minimum K
    # Score is K+1. We want to minimize K+1.
    # Check if there is a HC in graph with edges arrival <= K.
    
    lo = 0
    hi = m - 1
    best_cycle = None
    best_score = m + 1  # Initial upper bound
    
    # We can also try to get a good initial upper bound with a greedy approach
    # Greedy 2-regular
    degree = [0] * n
    adj_greedy = [set() for _ in range(n)]
    parent = list(range(n))
    size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if size[rx] < size[ry]: rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True

    for idx in range(m):
        if time.time() > deadline: break
        u, v = edge_order[idx]
        if degree[u] >= 2 or degree[v] >= 2: continue
        ru, rv = find(u), find(v)
        if ru == rv:
            if size[ru] < n: continue
        degree[u] += 1
        degree[v] += 1
        adj_greedy[u].add(v)
        adj_greedy[v].add(u)
        union(u, v)
        if all(d == 2 for d in degree):
            # Check if single cycle
            root = find(0)
            ok = all(find(i) == root for i in range(n))
            if ok:
                # Extract cycle
                cycle = [0]
                prev = -1
                cur = 0
                for _ in range(n - 1):
                    nxts = list(adj_greedy[cur])
                    nxt = nxts[0] if nxts[0] != prev else nxts[1]
                    prev, cur = cur, nxt
                    cycle.append(cur)
                sc = cycle_score(cycle)
                best_score = sc
                best_cycle = cycle
                break

    # Binary search
    # We know best_score is an upper bound. So K_max = best_score - 1.
    # If best_cycle is None, K_max = m - 1.
    if best_cycle is None:
        hi = m - 1
    else:
        hi = best_score - 1
        
    lo = 0
    
    # We will iterate K from lo to hi? No, binary search.
    # But the property "has HC" is monotonic: if it has HC for K, it has HC for K+1.
    # So we can binary search for the smallest K such that has_hc(K) is True.
    
    # To be safe and fast, we can just iterate K from 0 upwards?
    # No, binary search is better if the range is large.
    # However, the cost of has_hc(K) varies.
    # For small K, graph is sparse, fails fast.
    # For large K, graph is dense, succeeds fast.
    # The "hard" K is in the middle.
    
    # Let's do a linear search from 0 up to hi, but with a time limit per check?
    # Or binary search?
    # Binary search might hit a "hard" K in the middle.
    # Linear search starts with easy (sparse) checks.
    # Given N=24, linear search up to ~100 might be okay if each check is fast.
    # Let's try linear search from 0 to hi.
    
    current_time = time.time()
    for K in range(0, hi + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Build adj for K
        adj = [set() for _ in range(n)]
        for idx in range(K + 1):
            u, v = edge_order[idx]
            adj[u].add(v)
            adj[v].add(u)
            
        # Quick degree check
        valid = True
        for v in range(n):
            if len(adj[v]) < 2:
                valid = False
                break
        if not valid:
            continue
            
        # Run HC check
        # Allocate time based on remaining budget
        check_deadline = min(deadline - 2.0, time.time() + 2.0)
        cycle = has_hamiltonian_cycle(adj, check_deadline)
        
        if cycle:
            sc = cycle_score(cycle)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle
            # Since we found a cycle, we can stop?
            # Yes, because we are searching for the MINIMUM K.
            # If we found a cycle at K, the answer is K+1.
            # Any larger K will have a score >= K+1.
            # So we can break.
            break
            
    if best_cycle is None:
        # Fallback: return a default cycle or the greedy one if it failed?
        # Greedy should have worked or we should have found one.
        # If not, return identity permutation (score will be high)
        return list(range(n))
        
    return best_cycle

# EVOLVE-BLOCK-END
