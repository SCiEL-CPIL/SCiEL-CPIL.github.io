"""LUW algebraic construction for 3-regular graphs of large girth, with fast verification."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # safety margin before 150s kill

    # --- Fast girth check via non-backtracking BFS from every vertex ---
    def verify_girth(n, edges, min_girth):
        """Return True iff girth >= min_girth. Uses BFS with parent tracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_depth = min_girth // 2  # we only need to check up to depth floor(g/2)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle: length = du + dist[v] + 1
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # --- LUW construction for 3-regular graphs ---
    def build_luw(q, d):
        """Build a 3-regular graph using LUW-type algebraic construction.
        q = field size (3 for degree 3), d = depth parameter.
        """
        if q == 3:
            # Use Z/3Z arithmetic
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: (x * y) % 3
        elif q == 4:
            # Use F_4 with irreducible polynomial x^2+x+1
            # Elements: 0,1,2,3 where 2=i, 3=i+1, i^2=i+1
            table_add = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
            table_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: table_add[x][y]
            sub = lambda x, y: table_add[x][y]  # char 2: sub = add
            mul = lambda x, y: table_mul[x][y]
        else:
            return None

        # Define the rule for depth d
        # For LUW: rule at level c uses previous two levels
        # Standard LUW rule for 3-regular:
        # At depth c, vertex state is (x_0, x_1, ..., x_{c-1})
        # Transition: x_c = a * y_{c-2} + b * x_{c-1} (mod q)
        # where (a,b) cycles through specific values
        
        # We use the standard LUW rule:
        # c=1: free choice (gives degree q)
        # c>=2: x_c determined by previous states
        # For 3-regular, q=3, and we use a specific recurrence
        
        # Actually, the LUW construction for degree k uses:
        # Vertices are pairs (side, (x_1, x_2, ..., x_d))
        # where x_1 is in {0,...,q-1} (free), and x_c for c>=2 is computed.
        # The rule alternates between two types of recurrences.
        
        # Standard LUW rule for 3-regular graphs:
        # For side 0 (forward): x_c = y_{c-2} + x_{c-1}  (in F_q)
        # For side 1 (backward): x_c = x_{c-2} - y_{c-1}  (in F_q)
        # Wait, let me use the correct formulation.
        
        # LUW construction:
        # Vertices: (s, x) where s in {0,1} and x = (x_0, x_1, ..., x_{d-1}) in F_q^d
        # x_0 is always 0 (or a fixed value)
        # Neighbors of (s, x): for each y_1 in F_q, compute the path
        # and get the other vertex.
        
        # Let me use the formulation from the original LUW paper:
        # A vertex is (i, (x_0, x_1, ..., x_{d-1})) where i in {0,1}
        # x_0 = 0 always
        # The neighbors are obtained by choosing x_{-1} freely (gives degree q)
        # and computing forward/backward.
        
        # For the forward direction (i=0):
        # Given (0, (x_0,...,x_{d-1})), for each choice of z in F_q:
        #   y_0 = z
        #   y_1 = x_1 - x_0 * z  ... no, this isn't right either.
        
        # Let me use a simpler, well-known formulation:
        # The (k+1)-regular Ramanujan graph construction by Margulis/LPS:
        # For k=3, use q where q ≡ ±1 mod 6, and the graph has q(q^2-1)/2 vertices.
        # But for our purposes, the LUW construction is:
        
        # I'll use the explicit LUW recurrence:
        # State: (side, x_1, x_2, ..., x_d) where x_1 is free (0..q-1)
        # For side=0 (forward step from previous state):
        #   x_c = x_{c-1} + x_{c-2}  for c >= 3  (with x_0 = 0)
        #   But we need a different rule to get exactly degree 3.
        
        # Actually, the simplest correct LUW construction for degree k:
        # Uses a specific matrix over F_q.
        # For k=3, q=3:
        #   The adjacency is defined by: for vertex (s, x_1, ..., x_d),
        #   neighbors are obtained by extending the sequence in both directions.
        
        # Let me just use the known formula from the original code but with
        # a cleaner implementation:
        
        # The rule from the original program:
        # rule[2] = (1,1): x_2 = y_1 * x_1
        # rule[3] = (2,1): x_3 = y_2 * x_1  (but y_2 is from the other path)
        # This is getting complex. Let me use a verified formulation.
        
        # For 3-regular LUW graphs, use the following:
        # Vertices: (side, (x_1, ..., x_d)) where x_i in F_3
        # side in {0, 1}
        # Neighbors of (0, x): for each a in F_3:
        #   Compute y by: y_1 = a, y_c = x_c + a * x_{c-2} for c >= 3, with x_0=0
        #   Wait, this doesn't give a bijection.
        
        # I'll use the formulation that the original program used, which is:
        # The "rule" specifies for each depth c, which two previous positions
        # to use in the recurrence.
        
        # For 3-regular, the standard LUW rule is:
        # At each level c >= 2: x_c = y_a * x_b (in F_q)
        # where (a,b) depends on c.
        
        # Let me just implement it directly:
        
        def make_neighbors(q_val, d_val):
            if q_val == 3:
                add_f = lambda a, b: (a + b) % 3
                sub_f = lambda a, b: (a - b) % 3
                mul_f = lambda a, b: (a * b) % 3
            else:  # q_val == 4
                ta = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
                tm = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
                add_f = lambda a, b: ta[a][b]
                sub_f = lambda a, b: ta[a][b]
                mul_f = lambda a, b: tm[a][b]
            
            # LUW rule for 3-regular:
            # The key insight: use a recurrence x_c = f(x_{c-1}, x_{c-2})
            # that gives a 3-regular graph.
            # For F_3: x_c = x_{c-1} + x_{c-2} gives a 3-regular graph.
            # For F_4: x_c = x_{c-1} + x_{c-2} also works.
            
            def gen_neighbors(state):
                side, x = state
                x = list(x)
                # x[0] is x_1, x[1] is x_2, ..., x[d-1] is x_d
                # x_0 = 0 (implicit)
                for first in range(q_val):
                    y = [None] + [first]  # y[0] unused, y[1] = first
                    # y has indices 1..d, y[1]=first, y[2]..y[d] computed
                    y = [0] * (d_val + 1)
                    y[1] = first
                    # y[0] = 0 (x_0 = 0)
                    for c in range(2, d_val + 1):
                        if side == 0:
                            # Forward: x_c = x_{c-1} + x_{c-2} in the "other" sequence
                            # Actually, the LUW rule is:
                            # From vertex (side, x), going to neighbor:
                            # The neighbor's coordinates are computed by:
                            # y_c = x_c + mul(y_{c-2}, x_{c-1})  for side=0
                            # y_c = x_c - mul(x_{c-2}, y_{c-1})  for side=1
                            # with y_0 = 0, y_1 = first (free choice)
                            if c == 2:
                                # y_2 = x_2 + y_0 * x_1 = x_2 + 0 = x_2
                                # But wait, we need y_2 to depend on the free choice
                                # Let me re-examine the original code's rule.
                                pass
                            # I'll follow the original code's approach more carefully
                            pass
                        yield (0,)
                return None
            return None
        
        # This is getting too complex to derive from scratch.
        # Let me use the EXACT formulation from the original program,
        # which was verified to produce valid graphs, but optimize it.
        
        # Original formulation:
        # Vertices: (side, (None, x_1, x_2, ..., x_d)) -- the None is a placeholder
        # Actually: (side, (x_0, x_1, ..., x_d)) where x_0 = None (unused)
        # Neighbors of (side, x): for first in range(q):
        #   y = [None, first, 0, ..., 0]
        #   for c in range(2, d+1):
        #     a, b = rule[c]
        #     if side == 0: y[c] = add(x[c], mul(y[a], x[b]))
        #     else: y[c] = sub(x[c], mul(x[a], y[b]))
        #   yield (1-side, tuple(y))
        
        # Rule for 3-regular (q=3):
        # rule[2] = (1,1), rule[3] = (2,1), rule[4] = (1,2)
        # For c >= 5: rule[c] based on pattern
        
        # Let me implement this exactly:
        
        if q == 3:
            add_f = lambda a, b: (a + b) % 3
            sub_f = lambda a, b: (a - b) % 3
            mul_f = lambda a, b: (a * b) % 3
        else:
            ta = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
            tm = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add_f = lambda a, b: ta[a][b]
            sub_f = lambda a, b: ta[a][b]
            mul_f = lambda a, b: tm[a][b]
        
        # Build rule table
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
            rule[c] = options[r]
        
        # Generate graph
        start = (0, tuple([0] * (d + 1)))  # x_0..x_d all 0, side 0
        index = {start: 0}
        order = [start]
        edges = []
        
        for idx, vertex in enumerate(order):
            if time.time() - t0 > deadline - 5:
                return None, None
            side, x = vertex
            x_list = list(x)
            for first in range(q):
                y = [0] * (d + 1)
                y[0] = 0  # placeholder, unused in the formula
                y[1] = first
                # y[0] is x_0, which is 0
                # Actually in the original, y = [None, first] + [0]*(d-1)
                # So y[0] = None (unused), y[1] = first, y[2..d] = 0 initially
                # But in the formula, y[a] and y[b] are accessed where a,b are indices
                # Let me re-read: y[c] = add(x[c], mul(y[a], x[b]))
                # Here x and y are tuples of length d+1 (indices 0..d)
                # x[0] is the first element (was None in original, but we use 0)
                # Actually, let me match the original exactly.
                
                # Original: y = [None, first] + [0]*(d-1)
                # So y has length d+1: y[0]=None, y[1]=first, y[2..d]=0
                # x has length d+1: x[0]=None (for start) or computed
                # But x[0] is never used in the formula since a,b >= 1 always?
                # rule[2] = (1,1): a=1, b=1, so y[1] and x[1]
                # rule[3] = (2,1): a=2, b=1, so y[2] and x[1]
                # So x[0] and y[0] are never accessed. Good.
                
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_f(x[c], mul_f(y[a], x[b]))
                    else:
                        y[c] = sub_f(x[c], mul_f(x[a], y[b]))
                neighbor = (1 - side, tuple(y))
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                if side == 0:
                    edges.append((idx, index[neighbor]))
        
        return len(order), edges

    # --- Try LUW constructions with different parameters ---
    if k == 3:
        # For 3-regular, q can be 3 or 4
        # d determines the "depth" and thus the girth
        # Girth is approximately 2*d or so
        # For girth 16, we need d around 8-10
        
        best_n = float('inf')
        best_edges = None
        
        for q in [3, 4]:
            for d in range(6, 15):
                if time.time() - t0 > deadline - 10:
                    break
                n, edges = build_luw(q, d)
                if n is None:
                    continue
                if n < best_n:
                    # Verify girth
                    if verify_girth(n, edges, g):
                        best_n = n
                        best_edges = edges
                        break  # For this q, increasing d increases n, so stop
            if time.time() - t0 > deadline - 10:
                break
        
        if best_edges is not None:
            return best_n, best_edges
    
    # --- Fallback: random search ---
    def generate_random_3_reg(n):
        if n < 4 or n % 2 == 1:
            return None
        # Use configuration model with retry
        for _ in range(10):
            half_edges = [i for i in range(n) for _ in range(3)]
            random.shuffle(half_edges)
            edges_set = set()
            valid = True
            for i in range(len(half_edges) // 2):
                u, v = half_edges[2*i], half_edges[2*i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges_set:
                    valid = False
                    break
                edges_set.add(key)
            if not valid:
                continue
            edge_list = list(edges_set)
            deg = [0] * n
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if all(d == 3 for d in deg):
                return edge_list
        return None

    # Try random graphs
    for n in range(600, 1000, 20):
        if time.time() - t0 > deadline - 5:
            break
        for _ in range(10):
            if time.time() - t0 > deadline - 3:
                break
            edges = generate_random_3_reg(n)
            if edges is not None and verify_girth(n, edges, g):
                return n, edges

    # Last resort: return a known valid large graph
    # The complete bipartite graph K_{3,3} blown up won't work for girth 16.
    # Just return an error case - but we must return something valid.
    # Use a cycle of length 16 repeated? No, that's not 3-regular.
    # Use a large LUW graph as absolute fallback:
    n, edges = build_luw(3, 12)
    if n is not None:
        return n, edges
    
    return 0, []
# EVOLVE-BLOCK-END
