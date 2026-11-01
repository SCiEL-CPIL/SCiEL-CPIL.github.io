"""Voltage-graph lift approach for the (3,16)-cage problem."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g, minimizing n.
    
    Uses voltage-graph lifts of the Petersen graph over Z_m, then scans m
    upward from the Moore bound, certifying girth via non-backtracking BFS.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Moore bound for (3,16): 2 + 3*sum_{i=0}^{g/2-1} 2^i = 2 + 3*(2^8 - 1) = 2 + 3*255 = 767
    # But the actual lower bound is 512. Start from 512.
    moore = 512
    
    # Petersen graph: 10 vertices, 3-regular
    # Standard Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner 5-cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
    ]
    
    # For the Petersen graph, each vertex has degree 3.
    # We need to assign voltages (elements of Z_m) to each directed edge.
    # The voltage on edge (u,v) is a value; the inverse edge (v,u) gets -value.
    # A closed walk in the base graph lifts to a closed walk in the cover iff
    # the sum of voltages along the walk is 0 mod m.
    # The girth of the cover is at least g iff no non-trivial closed walk of
    # length < g in the base graph has voltage sum 0 mod m.
    
    # Strategy: assign voltages as powers of a primitive root mod m.
    # The Petersen graph has girth 5, so we need to check all closed walks
    # of length 5, 6, ..., g-1 = 15.
    
    # For a 3-regular graph, the number of closed walks of length L from a vertex
    # is bounded. We can enumerate them via non-backtracking walks.
    
    # Key insight: In the Petersen graph, the non-backtracking closed walks
    # of length L have specific voltage sums. If we assign voltages carefully,
    # we can ensure no short closed walk has sum 0 mod m.
    
    # Let's use a simpler approach: assign voltages based on the structure.
    # The Petersen graph can be viewed as a Cayley graph of Z_5 with generators {1, -1, 5}
    # where 5 represents the "spoke" connection.
    
    # Actually, let's use a different approach. The Petersen graph is the Kneser graph KG(5,2).
    # Vertices are 2-subsets of {0,1,2,3,4}, edges between disjoint pairs.
    
    # Let me re-index the Petersen graph using the standard labeling:
    # Vertices 0-4: outer cycle, vertices 5-9: inner star
    # Edge list as above.
    
    # For voltage assignment, I'll use the following scheme:
    # Assign each edge a voltage from {r^0, r^1, ..., r^{L-1}} where r is a primitive root mod m.
    # The key is that the voltage sum around any short closed walk should be non-zero mod m.
    
    # Let's try a specific voltage assignment pattern.
    # The Petersen graph has 15 edges. Assign voltages as consecutive powers of r.
    
    # For each m starting from moore, try to find a primitive root and voltage assignment.
    
    def find_primitive_root(m):
        """Find a primitive root modulo m, or return None."""
        if m <= 1:
            return None
        # Factor phi(m)
        phi = m
        n = m
        p = 2
        while p * p <= n:
            if n % p == 0:
                phi = phi // p * (p - 1)
                while n % p == 0:
                    n //= p
            p += 1
        if n > 1:
            phi = phi // n * (n - 1)
        
        # Find prime factors of phi
        factors = []
        n = phi
        p = 2
        while p * p <= n:
            if n % p == 0:
                factors.append(p)
                while n % p == 0:
                    n //= p
            p += 1
        if n > 1:
            factors.append(n)
        
        # Try candidates
        for g in range(2, m):
            if math.gcd(g, m) != 1:
                continue
            ok = True
            for p in factors:
                if pow(g, phi // p, m) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None
    
    def check_girth(n, edges, g_target):
        """Check if graph has girth >= g_target using non-backtracking BFS from each vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking BFS: state is (vertex, prev_vertex)
        # A non-backtracking walk of length L starting at s is a sequence
        # s, v1, v2, ..., vL where vi != v(i-2) for i >= 2.
        # A cycle of length L exists iff there's a non-backtracking walk of length L
        # from s back to s where the last step doesn't backtrack to the previous vertex.
        
        # For girth >= g, we need no cycle of length < g.
        # Check all lengths from 3 to g-1.
        
        # Use BFS in the non-backtracking graph.
        # State: (current_vertex, previous_vertex)
        # Start: for each vertex s, for each neighbor v of s, state (v, s) at distance 1.
        # We want to detect if we can return to s in exactly L steps without backtracking.
        
        # More efficient: for each starting vertex s, do BFS in the directed line graph
        # (non-backtracking walks) and check if s is reachable in L steps.
        
        for s in range(n):
            # BFS from s in the non-backtracking walk space
            # dist[(v, u)] = shortest non-backtracking walk length from s to v arriving from u
            # We start with walks of length 1: (neighbor, s)
            
            # visited[v][u] = distance
            # Use a dict: key (v, u), value = distance
            dist = {}
            queue = deque()
            
            for v in adj[s]:
                key = (v, s)
                dist[key] = 1
                queue.append((v, s, 1))
            
            while queue:
                v, u, d = queue.popleft()
                
                # If we're back at s and d >= 3, we found a cycle of length d
                # But we need d < g_target to be a violation
                if v == s and d >= 3 and d < g_target:
                    return False
                
                if d >= g_target:
                    continue
                
                for w in adj[v]:
                    if w == u:
                        continue  # no backtracking
                    key2 = (w, v)
                    if key2 not in dist:
                        dist[key2] = d + 1
                        queue.append((w, v, d + 1))
        
        return True
    
    # Scan m from moore upward
    for m in range(moore, 2000):
        if time.time() > deadline:
            break
        
        r = find_primitive_root(m)
        if r is None:
            continue
        
        # Assign voltages to the 15 edges of the Petersen graph
        # Use powers of r: r^0, r^1, ..., r^14
        # The assignment matters. Let's use a fixed pattern.
        
        # Edge voltages: assign edge i voltage r^i mod m
        # But we need to be careful about the direction.
        # For an undirected edge {u,v}, we assign voltage a to u->v and -a to v->u.
        
        # Let's try: for edge i in petersen_edges, voltage = r^(i) mod m
        # where i is 0..14
        
        # Build the lifted graph
        # Vertices: (base_vertex, lift_value) where lift_value in Z_m
        # Edge between (u, x) and (v, y) if {u,v} is an edge in base and y - x = voltage(u,v) mod m
        
        # For edge (u,v) with voltage a: (u, x) connects to (v, (x + a) % m)
        # and (v, x) connects to (u, (x - a) % m)
        
        # But wait - this gives a 3-regular graph on 10*m vertices.
        # We need to check if it's connected and has girth >= 16.
        
        n_lifted = 10 * m
        # This might be too large for BFS. Let's check girth first.
        
        # Actually, for m >= 512, n_lifted >= 5120, which is too large for the BFS
        # to complete in time. We need a smarter check.
        
        # Alternative: check the voltage sums of closed walks directly.
        # A closed walk in the base graph of length L lifts to a closed walk in the
        # cover iff the voltage sum is 0 mod m.
        # So we need: for every non-backtracking closed walk of length L < 16 in the
        # Petersen graph, the voltage sum is not 0 mod m.
        
        # Enumerate non-backtracking closed walks in the Petersen graph up to length 15.
        
        # Build adjacency with voltages
        # adj_volt[u] = list of (v, voltage) for edges from u to v
        adj_volt = [[] for _ in range(10)]
        for i, (u, v) in enumerate(petersen_edges):
            volt = pow(r, i, m)
            adj_volt[u].append((v, volt))
            adj_volt[v].append((u, (-volt) % m))
        
        # Check all non-backtracking closed walks of length 3 to 15
        # Starting from each vertex
        # State: (current, prev, length, voltage_sum)
        # We do BFS/DFS
        
        # For efficiency, use dynamic programming:
        # dp[L][v][u] = set of voltage sums for non-backtracking walks of length L from start to v arriving from u
        # But this is expensive. Let's just enumerate.
        
        # Actually, for the Petersen graph, the number of non-backtracking walks
        # of length L grows like 2^L (since after the first step, each vertex has
        # 2 non-backtracking choices). For L=15, that's about 32768 per starting vertex,
        # times 10 vertices = ~327K walks. That's manageable.
        
        valid = True
        for start in range(10):
            # BFS: state (current, prev)
            # Track voltage sum
            # visited to avoid re-exploring? No, different voltage sums matter.
            # Actually, we just need to check if any closed walk has sum 0 mod m.
            
            # Use a set of (current, prev, voltage_sum) at each level
            # But that could explode. Instead, just enumerate all non-backtracking walks.
            
            # Recursive enumeration
            def enumerate_walks(current, prev, length, volt_sum):
                if length >= 3 and current == start:
                    if volt_sum % m == 0:
                        return True  # found a violation
                    return False
                if length >= 15:
                    return False
                
                found = False
                for nxt, v in adj_volt[current]:
                    if nxt == prev:
                        continue
                    if enumerate_walks(nxt, current, length + 1, (volt_sum + v) % m):
                        return True
                return False
            
            # Start with first step
            for nxt, v in adj_volt[start]:
                if enumerate_walks(nxt, start, 1, v % m):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Build the lifted graph
            # Vertices: (base, lift) -> index = base * m + lift
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                volt = pow(r, i, m)
                for x in range(m):
                    y = (x + volt) % m
                    edges.append((u * m + x, v * m + y))
            
            # Verify degree and simplicity
            # Each vertex has degree 3 by construction
            # No loops or repeated edges by construction
            
            # Verify girth
            if check_girth(n_lifted, edges, g):
                return n_lifted, edges
            else:
                # Shouldn't happen if voltage check passed, but just in case
                continue
    
    # Fallback: return the original Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_vertex = (0, (None,) + (0,) * d)
    index, order, edges = {start_vertex: 0}, [start_vertex], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
