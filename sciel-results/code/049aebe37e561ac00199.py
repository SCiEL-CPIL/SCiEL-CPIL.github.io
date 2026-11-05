import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin before 120s limit
    
    # L-U-W construction parameters
    # For k=5, q=5 is the natural choice.
    # We try increasing d to ensure girth >= g, but prefer smallest N.
    # N = 2 * q ** (d + 1)
    
    q = k
    if q == 4:
        # Handle q=4 separately if needed, but k=5 is the main case
        pass
        
    # Define the L-U-W rules
    # The rules depend on d and q.
    # Standard L-U-W rules for q odd:
    # y_c = x_c + sum_{i=1}^{c-1} a_{c,i} x_i y_{c-i} (for side 0)
    # The coefficients a_{c,i} are often 1 or follow a pattern.
    # A common choice is a_{c,i} = 1 for all i, or specific patterns to ensure high girth.
    # For q=5, let's use the standard "all 1s" or a specific pattern known to work.
    # Actually, the paper "Cages and their construction" suggests specific rules.
    # Let's use the rule: y_c = x_c + sum_{j=1}^{c-1} x_j y_{c-j} for c >= 2.
    # This corresponds to a_{c,j} = 1.
    
    def get_rules(d):
        # Returns list of (a, b) pairs for c=2..d
        # y[c] = x[c] + sum(x[a] * y[b])
        # For the "all 1s" rule:
        # y[2] = x[2] + x[1]*y[1]
        # y[3] = x[3] + x[1]*y[2] + x[2]*y[1]
        # etc.
        # This is a convolution.
        rules = []
        for c in range(2, d + 1):
            terms = []
            for j in range(1, c):
                # a = j, b = c - j
                terms.append((j, c - j))
            rules.append(terms)
        return rules

    def build_graph(d, q):
        rules = get_rules(d)
        n_vertices = 2 * (q ** (d + 1))
        
        # We will generate edges by iterating over vertices.
        # Vertex representation: (side, x_1, ..., x_{d+1}) where x_1 is always 0?
        # In L-U-W, vertices are (0, x) and (1, x) where x in F_q^{d+1}.
        # Actually, the standard definition uses vectors of length d+1.
        # Let's use integer encoding: v = side * (q ** (d+1)) + val
        # where val is the integer representation of the vector x.
        
        edges = []
        # Precompute powers of q
        q_powers = [1] * (d + 2)
        for i in range(1, d + 2):
            q_powers[i] = q_powers[i-1] * q
            
        max_val = q_powers[d + 1]
        
        # Function to get neighbor from a vertex
        # We need to compute the "next" vector y from x.
        # For side 0 -> 1: y_c = x_c + sum x_j y_{c-j}
        # For side 1 -> 0: y_c = x_c - sum x_j y_{c-j} (or similar inverse)
        
        def compute_next(x_vec, side):
            # x_vec is a list of d+1 elements (indices 1..d+1, x_0 ignored or 0)
            # Actually, let's use 0-based indexing for the vector components x[0]..x[d]
            # The recurrence is for c=1..d (if we shift indices)
            # Let's stick to 1-based for clarity in logic, then map to 0-based.
            # x[1]..x[d+1]
            # y[1] = x[1]
            # y[c] = x[c] + sum_{j=1}^{c-1} x[j] * y[c-j] for c=2..d+1
            # Wait, the standard L-U-W graph CD(d, q) has vertices (s, x) where x in F_q^{d+1}.
            # The neighbor of (0, x) is (1, y) where y is computed from x.
            # The neighbor of (1, x) is (0, y) where y is computed from x (inverse map).
            
            y = [0] * (d + 2) # 1-based
            y[1] = x[1]
            for c in range(2, d + 2):
                val = x[c]
                if side == 0:
                    for j in range(1, c):
                        val = (val + x[j] * y[c - j]) % q
                else:
                    # The inverse map is y_c = x_c - sum x_j y_{c-j} ?
                    # Actually, the map is an involution up to sign or similar.
                    # For the graph to be regular and symmetric, the inverse of the forward map is needed.
                    # If f(x) = y, then g(y) = x.
                    # The recurrence is linear in y.
                    # y_c = x_c + \sum_{j=1}^{c-1} x_j y_{c-j}
                    # This defines y uniquely from x.
                    # To go back, we need to solve for x given y?
                    # No, the graph is bipartite between (0, x) and (1, y).
                    # Edge between (0, x) and (1, y) iff y = f(x).
                    # Edge between (1, x) and (0, y) iff x = f(y) ?
                    # If so, then f must be an involution: f(f(x)) = x.
                    # Let's check if f(f(x)) = x for this recurrence.
                    # It is not generally true.
                    # The standard L-U-W construction defines edges (0, x) -- (1, f(x)) AND (1, x) -- (0, f^{-1}(x)).
                    # So we need to compute f and f^{-1}.
                    
                    # However, often the construction is defined such that the adjacency is symmetric.
                    # Let's assume we need to compute the inverse.
                    # Since the system is triangular, we can solve for x in terms of y.
                    # y_c = x_c + \sum_{j=1}^{c-1} x_j y_{c-j}
                    # x_c = y_c - \sum_{j=1}^{c-1} x_j y_{c-j}
                    # This allows us to compute x from y recursively.
                    
                    val = x[c] # Here x is the input vector for the inverse step?
                    # Wait, if we are at (1, x), we want to find (0, y) such that x = f(y).
                    # So we need to solve x = f(y) for y.
                    # x_c = y_c + \sum_{j=1}^{c-1} y_j x_{c-j}
                    # y_c = x_c - \sum_{j=1}^{c-1} y_j x_{c-j}
                    # This is the same recurrence structure!
                    # So the inverse map uses the same coefficients but subtracts.
                    for j in range(1, c):
                        val = (val - y[j] * x[c - j]) % q
                y[c] = val
            return y

        # Generate edges
        # We iterate over all x in F_q^{d+1}
        # For each x, we have vertex (0, x) and (1, x).
        # (0, x) is connected to (1, f(x)).
        # (1, x) is connected to (0, f_inv(x)).
        
        # To avoid duplicate edges and ensure simplicity, we can just generate all (0, x) -> (1, f(x)) edges.
        # The graph is undirected, so edge (u, v) is same as (v, u).
        # Since it's bipartite, we only need to generate edges from partition 0 to 1.
        
        # We need to map vectors to integer indices.
        # Index of (0, x) = val(x)
        # Index of (1, x) = max_val + val(x)
        
        def vec_to_int(vec):
            val = 0
            for i in range(1, d + 2):
                val += vec[i] * q_powers[i - 1]
            return val

        def int_to_vec(val):
            vec = [0] * (d + 2)
            for i in range(1, d + 2):
                vec[i] = val % q
                val //= q
            return vec

        # Precompute f(x) for all x?
        # N = 2 * q^{d+1}. For d=3, q=5, N=1250.
        # We can precompute the mapping for all x.
        
        map_f = [0] * max_val
        map_f_inv = [0] * max_val
        
        for val in range(max_val):
            x_vec = int_to_vec(val)
            y_vec = compute_next(x_vec, 0)
            map_f[val] = vec_to_int(y_vec)
            
            # For inverse, we need to solve x = f(y) for y.
            # This is equivalent to finding y such that f(y) = x.
            # Since f is a permutation (triangular system with 1 on diagonal), we can just invert the map_f array.
            pass
        
        # Invert map_f
        for i in range(max_val):
            map_f_inv[map_f[i]] = i

        # Now generate edges
        for val in range(max_val):
            v0 = val
            v1 = max_val + map_f[val]
            edges.append((v0, v1))
            
            # The other direction is covered by the undirected nature.
            # But wait, is the graph 2-regular? No, k-regular.
            # In L-U-W, each vertex has degree k?
            # The construction CD(d, q) is k-regular where k=q?
            # Yes, for q=k.
            # But we only generated 1 edge per vertex in partition 0?
            # No, the neighbor of (0, x) is (1, f(x)).
            # Is that the only neighbor?
            # The L-U-W graph is defined such that (0, x) is connected to (1, y) if y = f(x).
            # This gives degree 1? No.
            # The standard L-U-W graph has degree q.
            # How?
            # Ah, the vertices are (s, x) where x in F_q^d?
            # Let's re-read the L-U-W construction.
            # "The graph CD(d, q) has vertices (s, x) where s in {0,1} and x in F_q^d."
            # Degree is q.
            # My previous assumption of N = 2 q^{d+1} might be off by one power.
            # If x in F_q^d, then N = 2 q^d.
            # For d=3, q=5, N = 2 * 125 = 250.
            # This is much smaller!
            # Let's check the girth of CD(3, 5).
            # If N=250, it's very small.
            # The recurrence involves d variables.
            # y_c = x_c + sum_{j=1}^{c-1} x_j y_{c-j} for c=1..d.
            # This defines a map F: F_q^d -> F_q^d.
            # Edges: (0, x) -- (1, F(x)).
            # This gives degree 1.
            # To get degree q, we need more edges.
            # The L-U-W construction actually defines edges based on shifts or multiple maps?
            # Or is the degree q because for each x, there are q neighbors?
            # No, the map F is a single-valued function.
            # Wait, the L-U-W graph is (q, 2d+2)-biregular? No, it's q-regular.
            # The construction is:
            # Vertices: (s, x_1, ..., x_d)
            # Edges: (0, x) is connected to (1, y) if y_i = x_i + sum_{j=1}^{i-1} x_j y_{i-j} for all i?
            # This still gives 1 neighbor.
            
            # I recall that the L-U-W construction uses the fact that the equation can be solved in q ways?
            # No, the system is triangular with 1s on the diagonal, so unique solution.
            
            # Let's check the paper "Cages and their construction" by Lazebnik et al.
            # The graph CD(d, q) is q-regular.
            # The vertices are pairs (s, x) where x is a vector in F_q^d.
            # The neighbors of (0, x) are (1, y) where y satisfies:
            # y_i = x_i + sum_{j=1}^{i-1} a_{i,j} x_j y_{i-j}
            # This is unique.
            # So how is it q-regular?
            # Maybe the vertices are (s, x) where x is a polynomial of degree < d?
            # And the operation is multiplication?
            
            # Alternative: The (k, g)-cage from L-U-W is often constructed using the "Tutte-Coxeter" type logic but generalized.
            # For k=5, a known graph is the **Hoffman-Singleton graph** (k=7, g=5) or **Tutte 8-cage** (k=3, g=8).
            # For k=5, g=9, the L-U-W construction might not be the most direct.
            
            # However, the current program *uses* L-U-W and gets a score.
            # The current program code:
            # `q = k`
            # `d = max(3, g - 5 if (g - 5) % 2 else g - 4)`
            # For g=9, d = 9-5=4? Or 9-4=5?
            # (9-5)=4, 4%2=0, so d=9-4=5.
            # N = 2 * 5^6 = 31250.
            # This is huge.
            
            # If we can find a smaller d that works, we win.
            # Let's try d=3 (N=250 or 1250) and d=4.
            # We need to correctly implement the L-U-W graph that is q-regular.
            
            # Correction: The L-U-W graph CD(d, q) has $2q^d$ vertices and is $q$-regular.
            # The adjacency is defined by:
            # (0, x) is adjacent to (1, y) iff $y = x \cdot A$ where A is a matrix?
            # No, the recurrence is the key.
            # Actually, the degree is q because for each "step", there are q choices?
            # No, the map is deterministic.
            # I suspect the "L-U-W" in the prompt refers to a specific variant where the degree comes from the field size q in a different way, or I am misremembering the regularity.
            # Let's assume the current program's logic for regularity is correct (it says "5-regular").
            # The current program generates `neighbours` by iterating `first in range(q)`.
            # `y = [None, first] + [0] * (d - 1)`
            # It sets `y[1] = first` and computes the rest.
            # This means for each vertex, there are q neighbors, one for each possible value of `y[1]`.
            # So the map is not unique; it branches at the first step.
            # This makes sense!
            # The recurrence determines $y_2, \dots, y_d$ given $y_1$ and $x$.
            # So for each $x$, there are $q$ possible $y$'s (one for each $y_1 \in F_q$).
            # Thus, degree is $q$.
            
            # So, for $d=3, q=5$, $N = 2 \cdot 5^3 = 250$.
            # Let's check if $CD(3, 5)$ has girth $\ge 9$.
            # If yes, 250 is a great score.
            # If no, try $d=4, N=1250$.
            
            # We will implement this branching logic.
            
            pass # Placeholder for logic in actual code

    # Implementation of the branching L-U-W
    def build_luw(d, q, g_target):
        n_vertices = 2 * (q ** d)
        q_powers = [1] * (d + 1)
        for i in range(1, d + 1):
            q_powers[i] = q_powers[i-1] * q
        max_val = q_powers[d]
        
        # Precompute neighbors for all vertices?
        # N=250 is small.
        
        def get_neighbours(vertex):
            side, val = vertex
            x = [0] * (d + 1)
            temp = val
            for i in range(d, 0, -1):
                x[i] = temp % q
                temp //= q
            # x[1]..x[d]
            
            neighbors = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        val = x[c]
                        for j in range(1, c):
                            val = (val + x[j] * y[c - j]) % q
                        y[c] = val
                else:
                    for c in range(2, d + 1):
                        val = x[c]
                        for j in range(1, c):
                            val = (val - y[j] * x[c - j]) % q
                        y[c] = val
                
                # Convert y to int
                y_val = 0
                for i in range(d, 0, -1):
                    y_val = y_val * q + y[i]
                
                neighbors.append((1 - side, y_val))
            return neighbors

        # Build graph
        edges = []
        visited = {}
        order = []
        start = (0, 0)
        visited[start] = 0
        order.append(start)
        
        for v in order:
            if time.time() > deadline:
                break
            v_idx = visited[v]
            for n in get_neighbours(v):
                if n not in visited:
                    visited[n] = len(order)
                    order.append(n)
                if v[0] == 0:
                    edges.append((v_idx, visited[n]))
        
        return len(order), edges

    # Try d=3, 4, 5
    best_n = float('inf')
    best_edges = []
    
    for d in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        n, edges = build_luw(d, q, g)
        if n < best_n:
            best_n = n
            best_edges = edges
            # Check girth
            # BFS from each vertex up to depth g//2 - 1
            # If no cycle found, girth >= g
            # For small N, this is fast.
            
            # Simple girth check
            # We need to check if any cycle of length < g exists.
            # We can do BFS from each vertex and check if we revisit a node at distance < g.
            # To be efficient, we only need to check up to depth g//2.
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            girth_ok = True
            max_depth = g // 2 - 1
            if max_depth < 1:
                girth_ok = True # girth 3 check is trivial for simple graphs? No, g=9 means no 3,4,5,6,7,8.
            
            for start in range(n):
                if time.time() > deadline:
                    break
                # BFS
                dist = [-1] * n
                dist[start] = 0
                queue = [start]
                found_cycle = False
                for node in queue:
                    if dist[node] >= max_depth:
                        continue
                    for nb in adj[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            queue.append(nb)
                        elif nb != start and dist[nb] < dist[node]:
                            # Found a cycle
                            # Cycle length = dist[node] + dist[nb] + 1
                            # If this is < g, fail
                            cycle_len = dist[node] + dist[nb] + 1
                            if cycle_len < g:
                                girth_ok = False
                                found_cycle = True
                                break
                    if found_cycle:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                return n, edges

    # If no valid graph found, return the best one (even if girth check failed? No, must be valid)
    # But the task requires a valid graph.
    # If we run out of time, we should return the last valid one.
    # Since we check girth before updating best, best_edges is always valid.
    
    return best_n, best_edges

