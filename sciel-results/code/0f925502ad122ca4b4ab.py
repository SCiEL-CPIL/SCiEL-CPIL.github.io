import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # Moore bound
    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
        else:
            moore_bound = (k - 1) * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:
        moore_bound = 1

    def get_girth(adj, n, limit_g):
        """
        Returns the girth of the graph.
        Uses BFS from each vertex, stopping if a cycle shorter than limit_g is found.
        Returns limit_g if no shorter cycle is found.
        """
        min_girth = limit_g
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= limit_g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            # We can potentially break early if we found a very short cycle
                            if min_girth < 6:
                                return min_girth
        return min_girth

    def build_cayley(p, g1, g2):
        """
        Builds a 3-regular Cayley graph on Z_p with generators {g1, -g1, g2, -g2}.
        Note: For 3-regular, we need 3 generators. 
        Standard Cayley graphs on abelian groups with generators S require |S| = degree.
        If we use {g1, -g1, g2}, it's not symmetric unless g1=g2 or something.
        Wait, a 3-regular graph on an abelian group must have a symmetric generating set S with |S|=3.
        Since the group is abelian, if g is in S, -g must be in S for the graph to be undirected?
        No, the graph is undirected if S = S^-1.
        If |S|=3, one element must be its own inverse (order 2) or the set is not symmetric?
        In Z_p (odd prime), no element has order 2.
        So we cannot have a 3-regular Cayley graph on Z_p with p odd.
        
        Correction: We can use a different group or a different construction.
        
        Alternative: Use the "Cage" construction via voltage graphs on a 3-regular base.
        The Petersen graph is the standard base.
        
        Let's try a different algebraic approach:
        Construct a graph on vertices (x, y) where x in Z_a, y in Z_b?
        
        Actually, let's look at the "Hoffman-Singleton" type or "Magma" cages.
        For k=3, g=16, known cages are around 900 vertices.
        
        Let's try a deterministic construction based on the **Clebsch graph** or similar?
        Clebsch is 5-regular.
        
        Let's go back to Voltage Lifts but make the search **deterministic and structured**.
        Instead of random voltages, we use **linear voltages** defined by a matrix.
        
        Base: Petersen Graph (10 vertices, 15 edges).
        Lift over Z_m.
        Vertices: (u, i) for u in 0..9, i in 0..m-1.
        Edge (u, i) - (v, (i + A[u,v]) % m).
        
        To ensure high girth, we want the "commutator" or "displacement" of any closed walk in the base to be non-zero modulo m, unless the walk is very long.
        
        A closed walk in the base of length L has a total voltage sum V.
        We want V != 0 mod m for all closed walks of length < g.
        
        The set of all closed walks in Petersen generates the free abelian group on the edges (with constraints).
        Actually, the voltage of a closed walk is a linear combination of the voltages of the edges.
        
        Let's define the voltage of edge e as a vector in Z_m^2 or just Z_m.
        If we use Z_m, the condition is that the sum of voltages on any closed walk of length < g is not 0 mod m.
        
        Let's try to construct voltages such that the graph is a "cover" with large girth.
        
        Strategy:
        1. Pick m (size of the lift).
        2. Define voltages for the 15 edges of Petersen.
        3. We can parameterize the voltages.
        
        Let's try a specific family:
        Let the edges be indexed.
        We can fix some voltages to 0 and others to powers of a base?
        
        Let's try **Randomized Local Search** on the voltage assignment.
        Start with a random assignment.
        Calculate the girth.
        If girth < g, find a short cycle.
        Perturb the voltages of edges in that cycle to break it.
        
        This is a "Simulated Annealing" or "Hill Climbing" on the voltage space.
        The state space is Z_m^15.
        
        Objective: Maximize girth.
        
        Steps:
        1. Initialize random voltages.
        2. Loop:
           a. Find shortest cycle (BFS).
           b. If length >= g, return success.
           c. Pick a random edge in the cycle.
           d. Change its voltage by a small random amount.
           e. If time runs out, break.
        
        This is structurally different from "try random seeds". It actively repairs short cycles.
        """
        pass

    # Since the previous thought about Cayley on Z_p failed for degree 3, 
    # we implement the Voltage Lift Local Search.
    
    # Petersen Graph Edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = len(pet_edges)
    
    # Map edge to index
    edge_idx = {}
    for i, e in enumerate(pet_edges):
        edge_idx[e] = i
        edge_idx[(e[1], e[0])] = i

    def build_and_check(m, voltages):
        """
        Builds the graph and returns (girth, adj, edges_list)
        """
        n = 10 * m
        adj = [[] for _ in range(n)]
        
        for (u, v) in pet_edges:
            vol = voltages[edge_idx[(u, v)]]
            for a in range(m):
                b = (a + vol) % m
                idx_u = u * m + a
                idx_v = v * m + b
                adj[idx_u].append(idx_v)
                adj[idx_v].append(idx_u)
        
        # Verify regularity and simplicity
        for i in range(n):
            if len(adj[i]) != 3:
                return -1, None, None
            if len(set(adj[i])) != 3:
                return -1, None, None
        
        # Check girth
        girth = get_girth(adj, n, g)
        
        if girth >= g:
            edges_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges_list.append((u, v))
            return girth, adj, edges_list
        else:
            return girth, adj, None

    def find_short_cycle(adj, n, max_len):
        """
        Finds a cycle of length < max_len.
        Returns the list of vertices in the cycle, or None.
        """
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found = None
            while queue:
                u = queue.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != start and parent[u] != v and parent[v] != u:
                        # Found a back edge
                        # Reconstruct cycle
                        cycle = [u, v]
                        # Walk back from u to start
                        curr = u
                        while curr != start:
                            curr = parent[curr]
                            if curr == v: break # Should not happen if we are careful, but just in case
                            cycle.append(curr)
                        # Walk back from v to start
                        curr = v
                        while curr != start:
                            curr = parent[curr]
                            if curr == u: break
                            cycle.append(curr)
                        # Remove start which is duplicated at the end of both paths?
                        # Actually, the cycle is u->...->start->...->v->u?
                        # No, BFS tree.
                        # Path start->...->u and start->...->v.
                        # Cycle is u->...->lca->...->v->u.
                        
                        # Simpler: Just return the edge (u,v) and the two paths?
                        # For perturbation, we just need the edges in the cycle.
                        
                        # Let's just return the edge index that closes the cycle
                        # We need to identify which edges are in the cycle to perturb them.
                        
                        # Reconstruct path start->u
                        path_u = []
                        curr = u
                        while curr != start:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_u.reverse()
                        path_u.append(start)
                        
                        # Reconstruct path start->v
                        path_v = []
                        curr = v
                        while curr != start:
                            path_v.append(curr)
                            curr = parent[curr]
                        path_v.reverse()
                        path_v.append(start)
                        
                        # Find LCA
                        # Since it's BFS, the LCA is where the paths diverge from start.
                        # Actually, we can just collect all edges in path_u and path_v.
                        # The cycle consists of edges in path_u + edges in path_v + edge (u,v).
                        
                        # To perturb, we can pick a random edge from the union of paths.
                        all_vertices = set(path_u) | set(path_v)
                        # Pick a random edge incident to a vertex in this set?
                        # Or just pick a random edge from the graph?
                        # Better: pick a random edge from the cycle.
                        
                        # Let's just return the edge (u,v) and the two paths.
                        # We will perturb a random edge from the cycle.
                        
                        # For simplicity in the loop, let's just return the edge index of (u,v)
                        # and assume we can perturb any edge in the cycle.
                        # We'll pick a random edge from the cycle edges.
                        
                        # Cycle edges:
                        # (start, path_u[0]), (path_u[0], path_u[1]), ..., (path_u[-1], u)
                        # (start, path_v[0]), ..., (path_v[-1], v)
                        # (u, v)
                        
                        cycle_edges = []
                        # Add edges from u to start
                        for i in range(len(path_u) - 1):
                            a, b = path_u[i], path_u[i+1]
                            if a < b:
                                cycle_edges.append(edge_idx.get((a, b)))
                            else:
                                cycle_edges.append(edge_idx.get((b, a)))
                        
                        # Add edges from v to start
                        for i in range(len(path_v) - 1):
                            a, b = path_v[i], path_v[i+1]
                            if a < b:
                                cycle_edges.append(edge_idx.get((a, b)))
                            else:
                                cycle_edges.append(edge_idx.get((b, a)))
                        
                        # Add closing edge
                        if u < v:
                            cycle_edges.append(edge_idx.get((u, v)))
                        else:
                            cycle_edges.append(edge_idx.get((v, u)))
                        
                        return cycle_edges
                    
            if found:
                return found
        return None

    best_n = float('inf')
    best_edges = None

    # Try different m values
    # Moore bound for g=16 is 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254.
    # Petersen lift has 10*m vertices.
    # We need 10*m >= 254 => m >= 26.
    # Known best is 936.
    # Let's try m from 30 to 100.
    
    rng = random.Random(42)
    
    for m in range(30, 100):
        if time.time() > deadline - 20:
            break
        
        n = 10 * m
        if n < moore_bound:
            continue
            
        # Initialize random voltages
        voltages = [rng.randint(0, m - 1) for _ in range(num_edges)]
        
        # Local Search
        for step in range(200):
            if time.time() > deadline - 10:
                break
                
            girth, adj, edges_list = build_and_check(m, voltages)
            
            if girth >= g:
                best_n = n
                best_edges = edges_list
                break
                
            # Find a short cycle
            cycle_edges = find_short_cycle(adj, n, g)
            
            if cycle_edges is None:
                # Should not happen if girth < g, but just in case
                # Randomly perturb
                idx = rng.randint(0, num_edges - 1)
                voltages[idx] = (voltages[idx] + rng.randint(1, m - 1)) % m
            else:
                # Perturb a random edge in the cycle
                # Filter out None indices (should not happen for Petersen edges)
                valid_edges = [e for e in cycle_edges if e is not None]
                if valid_edges:
                    idx = rng.choice(valid_edges)
                    voltages[idx] = (voltages[idx] + rng.randint(1, m - 1)) % m
                else:
                    idx = rng.randint(0, num_edges - 1)
                    voltages[idx] = (voltages[idx] + rng.randint(1, m - 1)) % m
        
        if best_edges is not None:
            break

    # If we found a solution, return it
    if best_edges is not None:
        return best_n, best_edges

    # Fallback: Return a large random graph or a known construction if time permits
    # For now, return the last best attempt or a simple large cycle (invalid girth but valid structure)
    # We must return a valid 3-regular graph.
    # Let's return a large Petersen lift with random voltages, even if girth is low,
    # as it is a valid 3-regular graph. The score will be bad (high n) but valid.
    
    # Actually, if we didn't find one, we should probably return the largest m we tried?
    # No, we want to minimize n.
    # If we failed to find girth g, we can't return it as a valid solution for the task contract?
    # "no cycle may be shorter than g"
    # So we MUST return a valid graph.
    # If we can't find one, we have to return a known valid one or fail.
    # The LUW fallback in the original code was a valid 3-regular graph.
    # Let's implement a simple fallback: A large cycle with chords? No, must be 3-regular.
    # A random 3-regular graph on 1000 vertices.
    
    # Let's use the LUW fallback from the original code, it's safe.
    def build_luw_fallback(d):
        q = 3
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
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

    # Try to find a valid one with a larger m if we have time
    for m in range(100, 150):
        if time.time() > deadline - 5:
            break
        n = 10 * m
        for seed in range(10):
            if time.time() > deadline - 2:
                break
            voltages = [random.Random(seed).randint(0, m - 1) for _ in range(num_edges)]
            girth, adj, edges_list = build_and_check(m, voltages)
            if girth >= g:
                return n, edges_list
                
    # Final Fallback
    best_n, best_edges = build_luw_fallback(11)
    return best_n, best_edges
