import time
import sys
from heapq import heappush, heappop

# EVOLVE-BLOCK-START

n = 24
edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

m = len(edge_order)

# Precompute arrival index for each edge (0-based)
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def edge_cost(u, v):
    """Return the 0-based arrival index of edge (u,v)."""
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    """Compute the 1-based max arrival index (score) of a Hamilton cycle."""
    L = len(cycle)
    max_arr = 0
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_cost(a, b)
        if ar > max_arr:
            max_arr = ar
    return max_arr + 1

def find_min_bottleneck_cycle(deadline):
    """
    Dijkstra-style best-first search over (mask, last) states with bottleneck cost.
    State: (bottleneck_cost, mask, last_vertex)
    We seek a Hamilton cycle starting and ending at vertex 0.
    The bottleneck cost is the max edge arrival index used so far.
    """
    FULL_MASK = (1 << n) - 1
    START = 0
    
    # dist[mask][last] = minimum bottleneck cost to reach state (mask, last)
    # Using a dict for sparse access
    dist = {}
    # parent tracking for path reconstruction
    parent = {}
    
    # Initial state: at vertex 0, mask has only vertex 0
    init_mask = 1 << START
    init_cost = 0  # No edges used yet
    
    dist[(init_mask, START)] = init_cost
    # Heap entries: (bottleneck_cost, mask, last)
    heap = [(init_cost, init_mask, START)]
    
    best_cycle = None
    best_cost = m  # upper bound: worst case is last edge
    
    while heap:
        cost, mask, last = heappop(heap)
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # If we've already found a better cost for this state, skip
        if dist.get((mask, last), m + 1) < cost:
            continue
        
        # If we've visited all vertices and can close the cycle
        if mask == FULL_MASK:
            # Check if last vertex connects back to START
            close_cost = edge_cost(last, START)
            total_cost = max(cost, close_cost)
            if total_cost < best_cost:
                best_cost = total_cost
                # Reconstruct path
                path = [last]
                cur_mask = mask
                cur_last = last
                while cur_last != START:
                    key = (cur_mask, cur_last)
                    if key not in parent:
                        break
                    prev_mask, prev_last = parent[key]
                    path.append(prev_last)
                    cur_mask = prev_mask
                    cur_last = prev_last
                path.reverse()
                best_cycle = path
                # Since we're using best-first, the first complete cycle found
                # with this cost is optimal. But let's continue a bit to be safe.
                # Actually, in Dijkstra with bottleneck, once we pop a complete state,
                # that's the optimal. But we might have multiple states with same cost.
                # Let's break since we found the minimum bottleneck.
                break
        
        # Expand: try all unvisited neighbors
        # For efficiency, iterate over all vertices not in mask
        for nb in range(n):
            if mask & (1 << nb):
                continue
            # Edge cost from last to nb
            ec = edge_cost(last, nb)
            new_cost = max(cost, ec)
            
            # Prune: if new_cost >= best_cost, no point
            if new_cost >= best_cost:
                continue
            
            new_mask = mask | (1 << nb)
            state_key = (new_mask, nb)
            
            if new_cost < dist.get(state_key, m + 1):
                dist[state_key] = new_cost
                parent[state_key] = (mask, last)
                heappush(heap, (new_cost, new_mask, nb))
    
    return best_cycle, best_cost + 1  # +1 for 1-based

def greedy_two_regular(deadline):
    """
    Greedy approach: add edges in order, maintaining max degree 2 and no premature cycles.
    Returns a cycle if found, else None.
    """
    degree = [0] * n
    adj = [set() for _ in range(n)]
    parent = list(range(n))
    size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True
    
    for idx in range(m):
        if time.time() > deadline:
            break
        u, v = edge_order[idx]
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            # Would create a cycle; only allow if it's the final cycle covering all vertices
            if size[ru] < n:
                continue
        # Add edge
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        union(u, v)
        
        # Check if we have a complete 2-regular connected graph
        if all(d == 2 for d in degree):
            # Verify it's a single cycle
            root = find(0)
            if all(find(i) == root for i in range(n)):
                # Extract cycle
                cycle = [0]
                prev = -1
                cur = 0
                for _ in range(n - 1):
                    nxts = list(adj[cur])
                    nxt = nxts[0] if nxts[0] != prev else nxts[1]
                    prev, cur = cur, nxt
                    cycle.append(cur)
                return cycle
    
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin before 90s limit
    
    best_cycle = None
    best_score = m + 1  # Worst case upper bound
    
    # Phase 1: Greedy 2-regular for a quick initial upper bound
    cycle = greedy_two_regular(deadline)
    if cycle:
        sc = cycle_score(cycle)
        if sc < best_score:
            best_score = sc
            best_cycle = cycle
    
    # Phase 2: Dijkstra best-first search for minimum bottleneck
    # This should find a much better solution
    cycle2, score2 = find_min_bottleneck_cycle(deadline)
    if cycle2:
        if score2 < best_score:
            best_score = score2
            best_cycle = cycle2
    
    # Fallback: if no cycle found, return identity
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END
