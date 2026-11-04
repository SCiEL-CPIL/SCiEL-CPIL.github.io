import time
import itertools
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Fallback: LUW Construction
    def luw_fallback(k, g):
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

    if k != 3 or g != 16:
        return luw_fallback(k, g)

    # Petersen graph: 10 vertices, 3-regular, girth 5
    # Vertices 0-9, edges:
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_petersen = 10
    
    # Build adjacency for Petersen
    petersen_adj = [[] for _ in range(n_petersen)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    # For voltage lift: assign a voltage (element of Z_m) to each directed edge
    # The lifted graph has n_petersen * m vertices
    # Vertex (u, i) connects to (v, (i + voltage(u,v)) % m) for each edge u-v
    
    # We need to check that the lifted graph has girth >= g
    # A cycle in the lift projects to a closed walk in the base
    # The girth of the lift is at least the girth of the base (5)
    # But we need girth 16, so we need m large enough
    
    # Key insight: for a voltage assignment over Z_m, 
    # a closed walk of length L in the base with net voltage s
    # becomes a cycle in the lift iff s ≡ 0 (mod m)
    # The shortest cycle in the lift has length >= g if:
    # - For every closed walk of length L < g in the base, the net voltage is not 0 mod m
    # - OR the projection has length >= g
    
    # Since Petersen has girth 5, closed walks of length 5,6,7,... exist
    # We need to ensure no closed walk of length < 16 has net voltage 0 mod m
    
    # Strategy: try different m values and voltage assignments
    # Start with small m, increase if needed
    
    def check_lift_girth(m, voltages, target_girth):
        """Check if the voltage lift has girth >= target_girth.
        voltages: dict mapping (u,v) -> voltage for directed edge u->v
        """
        n = n_petersen * m
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u in range(n_petersen):
            for v in petersen_adj[u]:
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + (i + voltages[(u, v)]) % m
                    adj[u_lift].append(v_lift)
        
        # Check girth using BFS from each vertex
        # For efficiency, we can use the fact that we only need to check
        # cycles up to length target_girth - 1
        
        for u in range(n):
            # BFS from u, track shortest path to each neighbor of u
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(u, 0)])
            visited[u] = 0
            
            while queue:
                curr, dist = queue.popleft()
                if dist >= target_girth:
                    break
                for v in adj[curr]:
                    if v == u:
                        # Self-loop, shouldn't happen
                        return False
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = curr
                        queue.append((v, dist + 1))
                    else:
                        # Found a cycle: u -> ... -> curr -> v -> ... -> u?
                        # Actually, we need to check if v is reachable from u
                        # and forms a cycle back to u
                        # The cycle length is dist + 1 if v is u, 
                        # or we need to trace back
                        # Simpler: just check if the path from u to curr plus edge curr->v
                        # creates a cycle that returns to u within target_girth
                        # This is complex; use a different approach
                        pass
        
        # Better approach: for each vertex u, do BFS and check if any neighbor
        # of u is reached at distance < target_girth - 1
        for u in range(n):
            neighbors_u = adj[u]
            if not neighbors_u:
                continue
            # BFS from u
            dist = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if dist[curr] >= target_girth - 1:
                    break
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        queue.append(v)
                    else:
                        # Check if v is a neighbor of u and the path u->...->curr->v->u
                        # forms a cycle of length dist[curr] + 2
                        # But this might not be a simple cycle
                        pass
            
            # Check if any neighbor of u has dist < target_girth - 1
            # Actually, if we reach a neighbor v of u at distance d, then
            # the cycle u -> ... -> v -> u has length d + 1
            # But we need d + 1 >= target_girth, so d >= target_girth - 1
            for v in neighbors_u:
                if v == u:
                    continue
                if dist[v] != -1 and dist[v] < target_girth - 1:
                    # Found a cycle of length dist[v] + 1 < target_girth
                    return False
        
        return True

    def get_voltages_from_assignment(m, assignment):
        """assignment: list of length 15, each element in range(m)
        Maps to directed edges in a fixed order
        """
        # Define a fixed ordering of directed edges
        # For each undirected edge (u,v), we have two directed edges
        # Let's define an ordering
        directed_edges = []
        for u in range(n_petersen):
            for v in sorted(petersen_adj[u]):
                directed_edges.append((u, v))
        
        voltages = {}
        for i, (u, v) in enumerate(directed_edges):
            voltages[(u, v)] = assignment[i] % m
            # For the reverse edge, voltage is -assignment[i] mod m
            voltages[(v, u)] = (-assignment[i]) % m
        return voltages

    # Try voltage lifts with different m
    # Theoretical lower bound: m >= g / girth_base = 16/5 = 3.2, so m >= 4
    # But in practice, we need m large enough to avoid short cycles
    
    best_result = None
    
    for m in range(4, 200):  # Try m from 4 to 199
        if time.time() > deadline - 5:
            break
        n_lift = n_petersen * m
        if n_lift > 936:
            break
        
        # Try random voltage assignments
        random.seed(42 + m)
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            assignment = [random.randint(0, m - 1) for _ in range(15)]
            voltages = get_voltages_from_assignment(m, assignment)
            
            # Quick check: verify girth
            if check_lift_girth(m, voltages, g):
                # Build the graph
                edges = []
                for u in range(n_petersen):
                    for v in petersen_adj[u]:
                        if u < v:  # Only add each undirected edge once
                            for i in range(m):
                                u_lift = u * m + i
                                v_lift = v * m + (i + voltages[(u, v)]) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                
                # Deduplicate and verify
                edges = list(set(edges))
                
                # Final verification: check degree and girth
                adj = [[] for _ in range(n_lift)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                valid = True
                for u in range(n_lift):
                    if len(adj[u]) != k:
                        valid = False
                        break
                
                if valid:
                    # Double-check girth with a more thorough method
                    girth_ok = True
                    for u in range(n_lift):
                        dist = [-1] * n_lift
                        dist[u] = 0
                        queue = deque([u])
                        while queue:
                            curr = queue.popleft()
                            if dist[curr] >= g:
                                break
                            for v in adj[curr]:
                                if dist[v] == -1:
                                    dist[v] = dist[curr] + 1
                                    queue.append(v)
                        for v in adj[u]:
                            if v != u and dist[v] != -1 and dist[v] < g - 1:
                                girth_ok = False
                                break
                        if not girth_ok:
                            break
                    
                    if girth_ok:
                        if best_result is None or n_lift < best_result[0]:
                            best_result = (n_lift, edges)
                            if n_lift <= 512:  # Moore bound, can't do better
                                return best_result
    
    if best_result is not None:
        return best_result
    
    return luw_fallback(k, g)
