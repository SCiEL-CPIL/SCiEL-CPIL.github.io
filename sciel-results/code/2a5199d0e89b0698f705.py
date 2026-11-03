import time
import random
from collections import deque
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
        """Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
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

    def nonbacktracking_girth(adj, g):
        """Return True if the graph has a cycle shorter than g."""
        n = len(adj)
        for s in range(n):
            for a in adj[s]:
                dist = {}
                dist[(a, s)] = 1
                q = deque([(a, s)])
                while q:
                    v, p = q.popleft()
                    d = dist[(v, p)]
                    if d >= g - 1:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        new_state = (w, v)
                        if new_state not in dist:
                            dist[new_state] = d + 1
                            if w == s and v != a:
                                if d + 1 < g:
                                    return True
                            q.append(new_state)
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def try_voltage_graph(k, g, deadline):
        if k != 4 or g != 10:
            return None
            
        # Base graph: K_{5,5} minus perfect matching
        # Vertices 0..4 (A), 5..9 (B)
        # Edges (i, j) where i in A, j in B, i != j-5
        base_edges = []
        for i in range(5):
            for j in range(5, 10):
                if i != j - 5:
                    base_edges.append((i, j))
        
        # There are 20 edges
        group_size = 32
        n_lift = 10 * group_size
        
        # Precompute short closed walks in base graph
        # We need to check cycles of length 4, 6, 8
        # A cycle in lift exists iff sum of voltages on corresponding closed walk in base is 0
        
        # To speed up, we can just check the lift directly for girth
        # But building the lift each time is expensive.
        # Instead, we check voltage sums.
        
        # Let's just do local search on the lift girth, but optimize the check.
        # Or, simpler: Random restarts with local search on voltage assignment.
        
        def get_energy(voltages):
            # Build adjacency of lift
            # n_lift = 320
            # adj is list of lists
            adj = [[] for _ in range(n_lift)]
            for idx, (u, v) in enumerate(base_edges):
                w = voltages[idx]
                for x in range(group_size):
                    u_id = u * group_size + x
                    v_id = v * group_size + (x + w) % group_size
                    adj[u_id].append(v_id)
                    adj[v_id].append(u_id)
            
            # Check girth < g
            # Use BFS from each vertex? Too slow for 320 vertices * 20 restarts?
            # 320 vertices, degree 4. BFS is fast.
            # 320 * 4 * depth. Depth 5. ~ 320 * 4 * 100 = 128k ops.
            # 20 restarts * 100 steps * 128k = 256M ops. Might be tight in Python.
            
            # Optimization: Only check for short cycles.
            # We can use the nonbacktracking check but it's O(n * k * g).
            
            # Alternative: Check voltage sums for all short closed walks in base.
            # Number of closed walks of length 4 in K_{5,5}-match:
            # It's manageable.
            
            # Let's stick to the lift check but optimize it.
            # We can skip vertices if we find a short cycle early.
            
            violations = 0
            for s in range(n_lift):
                # BFS
                visited = {}
                queue = deque([(adj[s][0], s, 1)])
                visited[(adj[s][0], s)] = 1
                # Note: adj[s] has 4 neighbors. We start from each?
                # Standard girth check:
                # For each start s, for each neighbor a, BFS from a with parent s.
                
                for a in adj[s]:
                    dist = {}
                    dist[(a, s)] = 1
                    q = deque([(a, s)])
                    found_short = False
                    while q:
                        v, p = q.popleft()
                        d = dist[(v, p)]
                        if d >= g - 1:
                            continue
                        for w in adj[v]:
                            if w == p:
                                continue
                            if w == s:
                                if d + 1 < g:
                                    violations += 1
                                    found_short = True
                                    break
                            new_state = (w, v)
                            if new_state not in dist:
                                dist[new_state] = d + 1
                                q.append(new_state)
                    if found_short:
                        break
            return violations

        best_voltages = None
        best_energy = float('inf')
        
        rng = random.Random(42)
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            voltages = [rng.randint(0, group_size - 1) for _ in range(len(base_edges))]
            energy = get_energy(voltages)
            
            if energy == 0:
                best_voltages = voltages
                break
                
            if energy < best_energy:
                best_energy = energy
                best_voltages = voltages[:]
            
            # Local search
            for step in range(1000):
                if time.time() > deadline - 10:
                    break
                if energy == 0:
                    best_voltages = voltages
                    break
                    
                idx = rng.randint(0, len(base_edges) - 1)
                old_v = voltages[idx]
                new_v = rng.randint(0, group_size - 1)
                if new_v == old_v:
                    continue
                voltages[idx] = new_v
                new_energy = get_energy(voltages)
                
                if new_energy < energy:
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_voltages = voltages[:]
                else:
                    # Revert
                    voltages[idx] = old_v
                    
        if best_voltages is None:
            return None
            
        # Build final edges
        edges = []
        for idx, (u, v) in enumerate(base_edges):
            w = best_voltages[idx]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + w) % group_size
                edges.append((u_id, v_id))
                
        return n_lift, edges

    # First, try the voltage graph construction for the specific (4,10) case
    if k == 4 and g == 10:
        result = try_voltage_graph(k, g, deadline)
        if result is not None:
            n_found, edges_found = result
            # Verify
            adj = build_adj(edges_found, n_found)
            if not nonbacktracking_girth(adj, g):
                return n_found, edges_found
            # If verification fails (should not happen if energy 0), continue to fallback
            # But energy 0 means no short cycles found.
            # Let's assume it's correct.

    # Fallback: LUW construction
    n_luw, edges_luw = luw_construction()
    return n_luw, edges_luw
