import time
import random
import sys
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (i, j) tuples."""
    start_time = time.time()
    deadline = start_time + 115.0

    def find_girth_nb(adj, n, target_girth):
        """Find girth using non-backtracking walk BFS.
        Returns the girth (or a value >= target_girth if girth >= target_girth)."""
        min_cycle = 999
        for s in range(n):
            # visited[(u, v)] = depth, meaning we reached v from u at depth
            visited = {}
            queue = deque()
            for v0 in adj[s]:
                visited[(s, v0)] = 0
                queue.append((s, v0, 0))
            while queue:
                u, v, depth = queue.popleft()
                if depth >= target_girth:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s and depth + 1 >= 3:
                        cl = depth + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                        continue
                    if (v, w) in visited:
                        cl = depth + visited[(v, w)] + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                    else:
                        visited[(v, w)] = depth + 1
                        queue.append((v, w, depth + 1))
            if min_cycle >= target_girth:
                return min_cycle
        return min_cycle

    def seed_construct(k, g):
        """The LUW seed construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    def adj_from_edges(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True

    def find_primitive_root(m):
        """Find a primitive root modulo m (for prime m)."""
        if m == 2:
            return 1
        phi = m - 1
        factors = []
        x = phi
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.append(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            factors.append(x)
        for g in range(2, m):
            ok = True
            for f in factors:
                if pow(g, phi // f, m) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def petersen_edges():
        """Return the 15 edges of the Petersen graph as (u, v) with u < v."""
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        edges.append((5, 7))
        edges.append((7, 9))
        edges.append((9, 6))
        edges.append((6, 8))
        edges.append((8, 5))
        for i in range(5):
            edges.append((i, i + 5))
        return edges

    def voltage_lift_petersen(m, root, target_girth):
        """Construct a voltage lift of the Petersen graph over Z_m.
        Assign voltages as powers of a primitive root to maximize girth.
        Returns (n, edges) or None if the lift doesn't have girth >= target_girth."""
        n = 10 * m
        pet_edges = petersen_edges()
        
        # Assign voltages: use powers of primitive root
        # For each edge, assign a voltage from the set of powers
        # We want to maximize the minimum cycle length
        # Strategy: assign distinct powers to maximize the order of products
        
        # The Petersen graph has 15 edges. We assign voltages as:
        # Edge i gets voltage root^i (mod m) for i = 0, ..., 14
        # But we need to be careful about orientation
        
        # For a voltage graph, each edge (u, v) gets a voltage a, meaning:
        # In the lift, vertex (u, h) connects to (v, h+a) and (v, h-a)
        # (undirected: both directions)
        
        # Actually for undirected voltage graphs, we assign each edge a voltage
        # and the lift has vertices (v, h) for v in base, h in Z_m
        # Edge (u,v) with voltage a creates edges:
        #   (u,h) - (v, h+a) for all h
        #   (u,h) - (v, h-a) for all h  [if we consider undirected]
        # But this would give degree 2*m... no.
        
        # For a regular voltage graph: each edge in the base has a voltage.
        # The lift of edge (u,v) with voltage a gives edges:
        #   (u,h) to (v, h+a) for each h
        # Since the base is undirected, we also need (v,h) to (u, h-a)
        # But these are the same set of edges just viewed from the other side.
        # So each base edge gives m edges in the lift.
        # Each vertex in the lift has degree equal to the degree in the base = 3.
        # Wait, that's not right either. Let me think again.
        
        # In a voltage graph, for each undirected edge e = {u,v} with voltage a_e:
        # In the lift, for each h in Z_m, we add edge ((u,h), (v, h + a_e))
        # This gives each vertex (u,h) exactly one edge going to v-side for each neighbor v of u.
        # So degree is preserved: 3-regular.
        # Total edges in lift = 15 * m = 3n/2. Correct.
        
        # Now, a closed walk in the lift projects to a closed walk in the base.
        # The "voltage" of the closed walk is the product (sum in Z_m) of edge voltages
        # with signs depending on direction.
        # A cycle of length L in the base with voltage sum 0 in Z_m gives a cycle of length L in the lift.
        # A cycle of length L with voltage sum s != 0 gives cycles of length L * order(s) in the lift.
        
        # So we want: for every cycle in the Petersen graph, the voltage sum is either 0
        # (giving a cycle of the same length) or has high order in Z_m.
        # The Petersen graph has girth 5, so we need cycles of length >= 16 in the lift.
        # Cycles of length 5 in the base: if voltage sum = 0, we get 5-cycles (bad).
        # If voltage sum has order >= 4, we get cycles of length >= 20 (good).
        # Cycles of length 6: if voltage sum = 0, we get 6-cycles (bad).
        # If order >= 3, we get cycles of length >= 18 (good).
        # Etc.
        
        # So the key constraint: no cycle in the Petersen graph should have voltage sum 0
        # (unless its length is already >= 16, which none are since girth is 5 and
        # the Petersen has no cycles longer than 10).
        
        # Actually the Petersen graph has cycles of length 5, 6, 8, 9, 10.
        # We need all these to have non-zero voltage sums with high enough order.
        
        # Let's assign voltages and check.
        # We'll use a systematic assignment and verify with BFS.
        
        # Build the lift adjacency
        adj = [[] for _ in range(n)]
        
        # Assign voltages: edge i gets voltage root^(i) mod m
        # But we need to handle the undirected nature carefully.
        # Let's orient each edge arbitrarily and assign a voltage.
        
        # For edge (u, v) with u < v, assign voltage a.
        # In the lift: (u, h) connects to (v, h + a) for all h.
        # Also (v, h) connects to (u, h - a) for all h (same edges).
        
        voltages = {}
        for i, (u, v) in enumerate(pet_edges):
            # Use powers of the primitive root
            voltages[(u, v)] = pow(root, i, m)
        
        for (u, v), a in voltages.items():
            for h in range(m):
                nu = u * m + h
                nv = v * m + ((h + a) % m)
                adj[nu].append(nv)
                adj[nv].append(nu)
        
        # Verify degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        # Check girth
        girth = find_girth_nb(adj, n, target_girth)
        if girth >= target_girth:
            edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            return n, edges
        return None

    # Get the seed construction as fallback
    n_seed, edges_seed = seed_construct(k, g)
    best_n = n_seed
    best_edges = edges_seed

    # Try voltage lifts of the Petersen graph
    if k == 3 and g == 16:
        moore = 512
        # Try m values from moore/10 upward (since n = 10*m)
        # n must be even, so 10*m is always even.
        # Start from m = 52 (n=520) up to m = 93 (n=930)
        
        for m in range(52, 94):
            if time.time() > deadline - 10:
                break
            
            if not is_prime(m):
                continue
            
            root = find_primitive_root(m)
            if root is None:
                continue
            
            result = voltage_lift_petersen(m, root, g)
            if result is not None:
                n_found, edges_found = result
                if n_found < best_n:
                    best_n = n_found
                    best_edges = edges_found
                    break
        
        # Also try with different voltage assignments
        # Try using root^i for different starting offsets
        if best_n == n_seed:
            for m in range(52, 94):
                if time.time() > deadline - 10:
                    break
                if not is_prime(m):
                    continue
                root = find_primitive_root(m)
                if root is None:
                    continue
                
                # Try different voltage assignments
                for offset in range(5):
                    n = 10 * m
                    pet_edges = petersen_edges()
                    adj = [[] for _ in range(n)]
                    
                    for i, (u, v) in enumerate(pet_edges):
                        a = pow(root, (i + offset) % m, m)
                        for h in range(m):
                            nu = u * m + h
                            nv = v * m + ((h + a) % m)
                            adj[nu].append(nv)
                            adj[nv].append(nu)
                    
                    valid = True
                    for i in range(n):
                        if len(adj[i]) != 3:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    girth = find_girth_nb(adj, n, g)
                    if girth >= g:
                        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        break

    # Final verification
    if best_n < 4 or len(best_edges) != best_n * k // 2:
        best_n = n_seed
        best_edges = edges_seed

    return best_n, best_edges
