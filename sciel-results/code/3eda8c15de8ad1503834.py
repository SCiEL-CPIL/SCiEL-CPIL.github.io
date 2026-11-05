import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            if g <= 5:
                d = 3
            else:
                d = 3
                while True:
                    girth_est = d + 5 if d % 2 == 1 else d + 4
                    if girth_est >= g:
                        break
                    d += 1

        if q == 4:
            # Define GF(4) operations
            # 0, 1, 2, 3
            # 1 = primitive element
            # 1+1=0, 1+2=3, 1+3=2, 2+2=1, 2+3=1, 3+3=2
            # 1*1=2, 1*2=3, 1*3=1, 2*2=3, 2*3=2, 3*3=1
            add = [[0,1,3,2],[1,0,2,3],[3,2,0,1],[2,3,1,0]]
            mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            # sub is same as add for GF(4) (char 2)
            sub = add
        else:
            add = [[(i + j) % q for j in range(q)] for i in range(q)]
            sub = [[(i - j) % q for j in range(q)] for i in range(q)]
            mul = [[(i * j) % q for j in range(q)] for i in range(q)]

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        # Precompute tables
        mul_table = mul
        add_table = add
        sub_table = sub

        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        
        for vertex in order:
            if time.time() > deadline - 20:
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def check_girth(n, edges, min_girth):
        """
        Returns True if girth >= min_girth.
        Uses BFS for each edge to find shortest cycle.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check if regular and simple
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if len(set(adj[i])) != len(adj[i]):
                return False # Multi-edge
        
        # Find girth
        girth = float('inf')
        for u, v in edges:
            # BFS from u to v avoiding direct edge (u,v)
            # We can just BFS and stop when we find v, but ensure path doesn't use (u,v) directly?
            # Actually, shortest path from u to v in G - (u,v) + 1 is the cycle length.
            # BFS from u to v in the graph where edge (u,v) is removed.
            
            # Optimization: If we find a path of length L, cycle is L+1.
            # We stop if L+1 >= min_girth.
            
            # Standard BFS
            visited = [-1] * n
            visited[u] = 0
            queue = deque([u])
            found = False
            while queue:
                curr = queue.popleft()
                dist = visited[curr]
                if dist + 1 >= min_girth: 
                    # Any further path will be longer
                    break
                for neighbor in adj[curr]:
                    if neighbor == v and curr == u:
                        continue # Skip the edge we removed
                    if neighbor == u and curr == v:
                        continue # Skip the edge we removed
                    if visited[neighbor] == -1:
                        visited[neighbor] = dist + 1
                        if neighbor == v:
                            girth = min(girth, dist + 2) # path length dist+1 + 1 edge
                            found = True
                            break
                        queue.append(neighbor)
                if found:
                    break
            if found:
                if girth < min_girth:
                    return False
            else:
                # No cycle? Impossible in finite connected graph with edges, 
                # but if u and v are in different components (shouldn't happen), 
                # or if the graph is a tree (impossible for regular k>1)
                pass
                
        return girth >= min_girth

    best_n = None
    best_edges = None

    # Try d values
    # For k=4, g=10, d=4 might work (n=512). d=5 is n=2048.
    # Start from d=3.
    for d_try in range(3, 10):
        if time.time() > deadline - 30:
            break
        try:
            n_try, edges_try = luw_construct(k, g, d_try)
            if n_try < 10000: # Sanity
                # Check girth
                if check_girth(n_try, edges_try, g):
                    if best_n is None or n_try < best_n:
                        best_n = n_try
                        best_edges = edges_try
                    break # Found smallest d that works, larger d will be larger n
        except Exception:
            pass

    if best_n is not None:
        return best_n, best_edges

    # Fallback: Basic LUW
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw
