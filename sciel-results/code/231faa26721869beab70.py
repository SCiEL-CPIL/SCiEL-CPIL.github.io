import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to seed logic for other parameters
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Specific case k=4, g=10
    # Strategy: Tensor product of cycles C_L x C_L, then local search.
    
    # Choose L. L=20 gives N=400. L=18 gives N=324.
    # The best known is 320. 324 is very close.
    # Let's try L=18 first.
    L = 18
    n = L * L
    
    # Build initial graph: Tensor product of C_L and C_L
    # Vertices are (i, j) for i,j in 0..L-1.
    # Edge exists between (i,j) and (i',j') if i ~ i' in C_L and j ~ j' in C_L.
    # In C_L, i is connected to (i+1)%L and (i-1)%L.
    
    adj = {i: set() for i in range(n)}
    edges = set()
    
    for i in range(L):
        for j in range(L):
            u = i * L + j
            # Neighbors in C_L for i: (i+1)%L, (i-1)%L
            # Neighbors in C_L for j: (j+1)%L, (j-1)%L
            for di in [1, -1]:
                ni = (i + di) % L
                for dj in [1, -1]:
                    nj = (j + dj) % L
                    v = ni * L + nj
                    if u < v:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.add((u, v))
    edges = list(edges)
    
    # Helper to count short cycles (length 4, 6, 8)
    def count_short_cycles(adj, n, max_len=8):
        # This is a heuristic count. 
        # For speed, we just check for 4-cycles first, then 6, then 8.
        # A 4-cycle is u-v-w-x-u.
        # Count = sum over u < w of |N(u) intersect N(w)|
        # This counts each 4-cycle 4 times (once for each vertex) or 2 times?
        # Each 4-cycle u-v-w-x-u corresponds to pairs (u,w) and (v,x).
        # So sum |N(u) \cap N(w)| counts each 4-cycle 2 times? No.
        # If u and w are opposite, N(u) \cap N(w) contains v and x. So 2 per pair.
        # Number of pairs (u,w) with common neighbors is C(4,2)/2? No.
        # Let's just count 4-cycles: 
        count4 = 0
        for u in range(n):
            for w in range(u+1, n):
                common = adj[u].intersection(adj[w])
                count4 += len(common)
        count4 //= 4 # Each 4-cycle is counted 4 times? 
        # Cycle u-v-w-x-u. 
        # Pairs (u,w): common {v,x} -> 2
        # Pairs (v,x): common {u,w} -> 2
        # Total 4. So divide by 4.
        
        # For 6-cycles, it's more complex.
        # We can just try to eliminate 4-cycles first.
        return count4, 0, 0 # Returning count4 for now, others 0 to save time
        
    # Local Search
    # We want to minimize the number of short cycles.
    # We'll iterate: find a 4-cycle, try to break it by swapping an edge.
    
    def get_4_cycle(adj, n):
        # Find any 4-cycle
        for u in range(n):
            for w in range(u+1, n):
                common = list(adj[u].intersection(adj[w]))
                if len(common) >= 2:
                    v, x = common[0], common[1]
                    return (u, v, w, x)
        return None

    def try_swap(adj, n, e1, e2, swap_type=0):
        # e1 = (u, v), e2 = (x, y)
        # Swap 1: (u,x), (v,y)
        # Swap 2: (u,y), (v,x)
        # Check if new edges are valid (not loops, not existing)
        u, v = e1
        x, y = e2
        
        # Check validity
        if u == x or u == y or v == x or v == y:
            return False
        new_e1 = (min(u,x), max(u,x)) if swap_type == 0 else (min(u,y), max(u,y))
        new_e2 = (min(v,y), max(v,y)) if swap_type == 0 else (min(v,x), max(v,x))
        
        if new_e1 in adj[u] or new_e2 in adj[v]: # Rough check, need set
             pass
            
        # Proper check
        if swap_type == 0:
             ne1 = (u, x)
             ne2 = (v, y)
        else:
             ne1 = (u, y)
             ne2 = (v, x)
             
        # Check if edges exist
        if x in adj[u] or y in adj[v] or y in adj[u] or x in adj[v]:
             # Might be existing edges, which is bad for simple graph if we add them?
             # We are removing e1, e2. Adding ne1, ne2.
             # If ne1 is already an edge (not e1/e2), it's a multi-edge.
             # We assume adj is a set of neighbors.
             if ne1[1] in adj[ne1[0]] and ne1 != e1 and ne1 != e2 and (ne1[0], ne1[1]) != (e1[1], e1[0]) and (ne1[0], ne1[1]) != (e2[1], e2[0]):
                  return False
             if ne2[1] in adj[ne2[0]] and ne2 != e1 and ne2 != e2 and (ne2[0], ne2[1]) != (e1[1], e1[0]) and (ne2[0], ne2[1]) != (e2[1], e2[0]):
                  return False
                  
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].add(ne1[1])
        adj[ne1[1]].add(u)
        adj[v].add(ne2[1])
        adj[ne2[1]].add(v)
        
        return True

    # Main search loop
    # Time limit 120s.
    start_time = time.time()
    
    # Initial count
    c4, _, _ = count_short_cycles(adj, n)
    
    while time.time() - start_time < 110:
        cycle = get_4_cycle(adj, n)
        if not cycle:
            break # No 4-cycles left. Check for 6/8?
            
        u, v, w, x = cycle
        # Edges in cycle: (u,v), (v,w), (w,x), (x,u)
        # We can try to swap one of these edges with another edge to break the cycle.
        # Or swap two edges from the cycle?
        # Standard edge switch: pick two edges (u,v) and (a,b) not in the cycle, 
        # and swap to (u,a) and (v,b).
        
        # Let's pick edge (u,v).
        # Pick a random other edge (a,b) from the graph.
        # To be efficient, we can pick a random vertex a, then a neighbor b.
        
        improved = False
        for _ in range(50): # Try 50 swaps per found cycle
             # Pick random edge
             a = random.randint(0, n-1)
             if not adj[a]: continue
             b = random.choice(list(adj[a]))
             
             # Skip if (a,b) is in the cycle
             if (a == u and b == v) or (a == v and b == u) or (a == v and b == w) or (a == w and b == v) or (a == w and b == x) or (a == x and b == w) or (a == x and b == u) or (a == u and b == x):
                 continue
                 
             # Try swap (u,v) and (a,b)
             # Save state
             adj_copy = {k: set(v) for k, v in adj.items()}
             
             # Remove (u,v) and (a,b)
             adj[u].remove(v)
             adj[v].remove(u)
             adj[a].remove(b)
             adj[b].remove(a)
             
             # Add (u,a) and (v,b)
             # Check if valid
             if a in adj[u] or b in adj[v]:
                 # Restore
                 adj[u].add(v)
                 adj[v].add(u)
                 adj[a].add(b)
                 adj[b].add(a)
                 continue
                 
             adj[u].add(a)
             adj[a].add(u)
             adj[v].add(b)
             adj[b].add(v)
             
             # Check if this reduced 4-cycles
             new_c4, _, _ = count_short_cycles(adj, n)
             
             if new_c4 < c4:
                 c4 = new_c4
                 improved = True
                 break
             else:
                 # Restore
                 adj = adj_copy
                 
        if not improved:
             # Try different strategies or break if stuck
             # We can try to swap edges NOT in the cycle to break it indirectly?
             # Or just break to avoid infinite loop on stuck state.
             # For now, break if no improvement after several tries.
             # To keep it simple, we just break.
             break
             
    # Final check: is girth >= 10?
    # We need to verify no cycles of length 6 or 8 either.
    # If we only removed 4-cycles, we might still have 6-cycles.
    # Let's run a quick check for 6-cycles.
    
    def has_6_cycle(adj, n):
        # Heuristic: check a few random starts or all?
        # N=324. Checking all is 324 * (degree^2) ~ 324 * 16 ~ 5000 ops per start?
        # Total ~ 1.6M ops. Fast.
        for start in range(n):
             # BFS to depth 3
             # State: (node, parent, depth)
             # If we find a node at depth 3 that is connected to start? No, that's length 4.
             # We need to find a path of length 5 from start to a neighbor of start?
             # Or path of length 6 from start to start?
             
             # Let's do BFS from start, tracking parents.
             # If we reach a node already visited, check cycle length.
             dist = {start: 0}
             parent = {start: -1}
             queue = deque([(start, -1)])
             
             while queue:
                 u, p = queue.popleft()
                 for v in adj[u]:
                     if v == p:
                         continue
                     if v == start:
                         # Found cycle
                         # Length = dist[u] + 1
                         # If dist[u] + 1 == 6, we have a 6-cycle.
                         # But BFS finds shortest path.
                         # If shortest cycle from start is 6, we are good for this start?
                         # No, we need to ensure NO cycle < 10.
                         # So if we find a cycle of length 4, 6, or 8, it's bad.
                         # If we find length 6, return True.
                         if dist[u] + 1 == 6:
                             return True
                         # If length < 6, we would have found it earlier?
                         # BFS explores by depth.
                         # If we find a back edge to start, the cycle length is dist[u]+1.
                         # Since we process in order of distance, the first back edge to start
                         # gives the shortest cycle through start.
                         # If shortest cycle through start is >= 10, we are good for start.
                         # But we must check all starts.
                         pass
                     elif v not in dist:
                         dist[v] = dist[u] + 1
                         parent[v] = u
                         queue.append((v, u))
                     else:
                         # Back edge
                         # Cycle length = dist[u] + dist[v] + 1
                         # But this might not be the shortest cycle in the graph,
                         # just a cycle involving u and v.
                         # We need to check if any cycle is < 10.
                         # If dist[u] + dist[v] + 1 < 10, return True.
                         # Note: This detects cycles that pass through the LCA of u and v.
                         # It might miss cycles not passing through LCA?
                         # No, any cycle has a unique LCA in the BFS tree?
                         # Not necessarily.
                         # However, if a short cycle exists, it will be detected
                         # by some start node.
                         # Actually, this check `dist[u] + dist[v] + 1` is only valid
                         # if u and v are in the same BFS tree branch?
                         # If they are in different branches, the path goes up to LCA and down.
                         # The length is dist[u] + dist[v] + 1.
                         # This is a valid cycle length.
                         # If this length is < 10, we have a short cycle.
                         if dist[u] + dist[v] + 1 < 10:
                             return True
                             
             # If no short cycle found from start, continue
        return False

    if has_6_cycle(adj, n):
        # If we still have short cycles, the score might be invalid or high.
        # We should try to return a valid graph.
        # If we can't guarantee girth 10, we might fall back to a known valid construction
        # or just return the current one and hope the evaluator handles it?
        # The task says "no cycle may be shorter than g".
        # If we return an invalid graph, score is 0.
        # So we MUST ensure validity.
        
        # Fallback: Use the seed logic which is guaranteed valid (n=2048).
        # Or use a hardcoded known 320 graph?
        # We don't have one.
        # Let's try to run the local search longer or with different parameters.
        # Or just use the seed fallback for safety if local search fails.
        
        # Let's check time.
        if time.time() - start_time > 115:
             # Timeout risk, use seed
             pass
             
        # For now, if has_6_cycle is True, we use the seed fallback.
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Convert adj to edge list
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append((u, v))
                
    return n, final_edges
