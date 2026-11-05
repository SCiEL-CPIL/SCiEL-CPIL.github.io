import time
import numpy as np
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (non-backtracking BFS from every vertex) ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Non-abelian voltage lift on the (3,6)-cage (20 vertices) ----------------
    def nonabelian_voltage_lift():
        if k != 3 or g != 16:
            return None

        # The 20-vertex (3,6)-cage: a 3-regular graph on 20 vertices with girth 6.
        # This is the Petersen graph's "double" or we can use the dodecahedron's dual structure.
        # Actually, the (3,6)-cage is the 20-vertex graph. Let me construct it properly.
        # The (3,6)-cage is the 20-vertex 3-regular graph with girth 6.
        # It can be constructed as follows: take the Petersen graph and... 
        # Actually, the smallest 3-regular graph with girth 6 has 20 vertices.
        # Let me use a known construction: it's the graph where vertices are (i, j) with i in Z_5, j in {0,1,2,3,4}
        # No, let me just hardcode the (3,6)-cage edges.
        
        # The (3,6)-cage (20 vertices) can be described as:
        # Vertices 0..19, edges forming a 3-regular graph with girth 6.
        # One standard representation:
        cage_edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5), (1, 6),
            (2, 7), (2, 8), (2, 9),
            (3, 10), (3, 11), (3, 12),
            (4, 13), (4, 14), (4, 15),
            (5, 16), (5, 17), (5, 18),
            (6, 19), (6, 0), (6, 1),  # wait, this creates triangles
        ]
        # Let me use a proper (3,6)-cage. The (3,6)-cage is actually the 20-vertex graph.
        # Let me construct it via the standard method: it's the Levi graph of the Petersen graph? No.
        # The (3,6)-cage is the 20-vertex 3-regular graph. Let me look it up mentally.
        # It's the graph obtained from the Petersen graph by replacing each vertex with an edge and connecting appropriately.
        # Actually, let me just use a known list. The (3,6)-cage has 20 vertices and 30 edges.
        
        # A concrete (3,6)-cage:
        # Vertices 0-19. Edges:
        # Layer 0 (5 vertices): 0,1,2,3,4 forming a 5-cycle: (0,1),(1,2),(2,3),(3,4),(4,0)
        # Layer 1 (5 vertices): 5,6,7,8,9 forming a 5-cycle: (5,6),(6,7),(7,8),(8,9),(9,5)
        # Layer 2 (5 vertices): 10,11,12,13,14 forming a 5-cycle: (10,11),(11,12),(12,13),(13,14),(14,10)
        # Layer 3 (5 vertices): 15,16,17,18,19 forming a 5-cycle: (15,16),(16,17),(17,18),(18,19),(19,15)
        # Connections between layers:
        # 0-5, 1-6, 2-7, 3-8, 4-9
        # 5-10, 6-11, 7-12, 8-13, 9-14
        # 10-15, 11-16, 12-17, 13-18, 14-19
        # 15-0, 16-1, 17-2, 18-3, 19-4
        # Wait, that's 4*5 + 5*4 = 20+20 = 40 edges, but we need 30 edges for 20 vertices of degree 3.
        # 20*3/2 = 30 edges. So I have too many.
        
        # Let me reconsider. The (3,6)-cage is the 20-vertex graph. Let me use a different construction.
        # Actually, I'll just use the Petersen graph as the base (10 vertices, girth 5) and accept that
        # we need a larger group to get girth 16. The Petersen approach gave ~770 vertices.
        # With the (3,6)-cage (girth 6), we need fewer constraints, so potentially smaller groups.
        
        # Let me just use the Petersen graph but with a non-abelian group.
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all non-backtracking closed walks of length [5, g-1]
        max_L = g - 1
        walk_vectors = set()
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length >= max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C = np.array(list(walk_vectors), dtype=np.int64)

        # Non-abelian group: use S_3 (symmetric group on 3 elements)
        # S_3 has 6 elements. We represent them as permutation matrices or just track group elements.
        # For a 3-regular graph, each edge gets a voltage in S_3 \ {e}.
        # S_3 elements: e, (12), (13), (23), (123), (132)
        # Non-identity elements: 5 of them.
        
        # Group multiplication table for S_3 (0=e, 1=(12), 2=(13), 3=(23), 4=(123), 5=(132))
        # We need to define multiplication. Let me use a simple representation.
        # Actually, for voltage graphs, the voltage on an edge in one direction is the inverse of the voltage in the other direction.
        # In S_3: e^{-1}=e, (12)^{-1}=(12), (13)^{-1}=(13), (23)^{-1}=(23), (123)^{-1}=(132), (132)^{-1}=(123)
        
        # Let me define S_3 multiplication. Elements as permutations of (1,2,3):
        # 0: identity (1)(2)(3)
        # 1: (12)
        # 2: (13)
        # 3: (23)
        # 4: (123) = (1->2, 2->3, 3->1)
        # 5: (132) = (1->3, 2->1, 3->2)
        
        # Multiplication table (row * column):
        # I'll compute it:
        # (12)*(12)=e, (12)*(13)=(132), (12)*(23)=(123), (12)*(123)=(13), (12)*(132)=(23)
        # (13)*(12)=(132), (13)*(13)=e, (13)*(23)=(123), (13)*(123)=(23), (13)*(132)=(12)
        # (23)*(12)=(123), (23)*(13)=(132), (23)*(23)=e, (23)*(123)=(13), (23)*(132)=(12)
        # (123)*(12)=(13), (123)*(13)=(23), (123)*(23)=(12), (123)*(123)=(132), (123)*(132)=e
        # (132)*(12)=(23), (132)*(13)=(12), (132)*(23)=(13), (132)*(123)=e, (132)*(132)=(123)
        
        mult_table = np.array([
            [0, 1, 2, 3, 4, 5],
            [1, 0, 5, 4, 2, 3],
            [2, 5, 0, 4, 3, 1],
            [3, 4, 4, 0, 2, 1],
            [4, 2, 3, 1, 5, 0],
            [5, 3, 1, 2, 0, 4]
        ], dtype=np.int64)
        
        # Inverses:
        inv = [0, 1, 2, 3, 5, 4]
        
        non_identity = [1, 2, 3, 4, 5]
        m = 6  # group size
        
        # For a voltage graph over a non-abelian group, the product along a walk is computed by
        # multiplying voltages in order, taking inverses when traversing edges in reverse.
        
        # This is more complex than abelian. Let me implement a solver that assigns voltages
        # to the 15 edges from non_identity elements, and checks that no non-backtracking closed
        # walk of length < g has product = identity.
        
        def product_along_walk(counts, voltages):
            # counts[i] is the net number of times edge i is traversed forward minus backward
            # But for non-abelian groups, order matters! So we can't just use counts.
            # We need to track the actual sequence of edges traversed.
            # This makes the problem much harder.
            # For now, let's just try random assignments and verify directly.
            return None

        # Given the complexity, let me just do random search with direct verification.
        # Assign random voltages from non_identity to each of the 15 edges, build the graph,
        # and check girth.
        
        def build_and_verify(voltages, m):
            n = n_base * m
            edges = []
            for u, v_base in pet_edges:
                e_idx = edge_index[(u, v_base)]
                vol = voltages[e_idx]
                for layer in range(m):
                    u_lift = u * m + layer
                    # In non-abelian, the voltage on edge (u,v) from u to v is vol,
                    # so v's layer is layer * vol (right multiplication) or vol * layer (left)
                    # Convention: if we label vertices by group elements, and edge (u,v) has voltage g,
                    # then the lift of (u, h) connects to (v, h*g)
                    v_lift = v_base * m + int(mult_table[layer, vol])
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    else:
                        edges.append((v_lift, u_lift))
            
            # Check for duplicate edges and self-loops
            edge_set = set()
            valid_edges = []
            for a, b in edges:
                if a == b:
                    return None
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    return None
                edge_set.add(key)
                valid_edges.append((a, b))
            
            # Check degree
            deg = [0] * n
            for a, b in valid_edges:
                deg[a] += 1
                deg[b] += 1
            if any(d != k for d in deg):
                return None
            
            gc = check_girth(n, valid_edges, g, deadline - 3)
            if gc >= g:
                return (n, valid_edges)
            return None

        # Try random voltage assignments
        rng = random.Random(42)
        best_result = None
        max_attempts = 200
        for attempt in range(max_attempts):
            if remaining() < 10:
                break
            voltages = [rng.choice(non_identity) for _ in range(15)]
            res = build_and_verify(voltages, m)
            if res is not None:
                return res
        
        return None

    # ---------------- Abelian voltage lift (fallback, same as before but optimized) ----------------
    def abelian_voltage_lift():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1
        walk_vectors = set()
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length >= max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C = np.array(list(walk_vectors), dtype=np.int64)
        constraint_nonzero_indices = [np.nonzero(C[i])[0] for i in range(C.shape[0])]

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        def solve_voltage(m, max_iters, seed):
            rng = random.Random(seed)
            v = np.random.RandomState(seed).randint(1, m, size=15).astype(np.int64)

            def cost(vec):
                residues = np.mod(C @ vec, m)
                return int(np.sum(residues == 0))

            best_v = v.copy()
            best_cost = cost(v)
            if best_cost == 0:
                return v, 0

            T = 1.0
            cooling = 0.9997
            min_T = 0.0005
            for it in range(max_iters):
                if remaining() < 4:
                    break
                if best_cost > 0:
                    residues = np.mod(C @ v, m)
                    violated_idx = np.nonzero(residues == 0)[0]
                    if len(violated_idx) == 0:
                        break
                    c_idx = violated_idx[rng.randrange(len(violated_idx))]
                    nz_indices = constraint_nonzero_indices[c_idx]
                    if len(nz_indices) == 0:
                        continue
                    j = nz_indices[rng.randrange(len(nz_indices))]
                else:
                    j = rng.randrange(15)
                old_val = v[j]
                v[j] = rng.randrange(1, m)
                new_cost = cost(v)
                delta = new_cost - best_cost
                if delta < 0 or (T > min_T and rng.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                T *= cooling
            return best_v, best_cost

        def build_and_verify(v, m):
            n = 10 * m
            edges = []
            for u, v_base in pet_edges:
                e_idx = edge_index[(u, v_base)]
                vol = int(v[e_idx])
                for layer in range(m):
                    u_lift = u * m + layer
                    v_lift = v_base * m + (layer + vol) % m
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    else:
                        edges.append((v_lift, u_lift))
            gc = check_girth(n, edges, g, deadline - 3)
            if gc >= g:
                return (n, edges)
            return None

        m = 77
        while m < 200 and remaining() > 20:
            if is_prime(m):
                for restart in range(8):
                    if remaining() < 10:
                        break
                    v, cost = solve_voltage(m, max_iters=5000, seed=restart * 1000 + m)
                    if cost == 0:
                        res = build_and_verify(v, m)
                        if res is not None:
                            return res
            m += 1
        return None

    # ---------------- Main ----------------
    best = None

    # Try non-abelian first
    if remaining() > 5:
        r1 = nonabelian_voltage_lift()
        if r1 is not None:
            best = r1

    # Fallback to abelian
    if remaining() > 10:
        r2 = abelian_voltage_lift()
        if r2 is not None:
            if best is None or r2[0] < best[0]:
                best = r2

    if best is not None:
        return best
    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
