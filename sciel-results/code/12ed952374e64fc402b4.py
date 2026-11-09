"""Construct 91 edges on 31 vertices minimizing 4-cycles.
Uses polarity graph of PG(2,5) for a C4-free 90-edge core,
then adds one edge and performs incremental local search (edge swaps) to minimize total C4s.
Optimized with delta-based C4 counting in the swap loop.
"""
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    n = 31
    q = 5
    
    # Build points of PG(2,5)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    pt = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    pt = (0, 1, (z * inv_y) % q)
                else:
                    pt = (0, 0, 1)
                if pt not in seen:
                    seen.add(pt)
                    points.append(pt)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: p perp q iff x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    def is_adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1 * x2 + y1 * y2 - z1 * z2) % q == 0
    
    # Build core edge list
    core_edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_adjacent(i, j):
                core_edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    if len(core_edges) != 90:
        return _greedy_construct()
    
    # Find best edge to add: minimize new C4s created
    best_edge = None
    best_new_c4s = float('inf')
    candidate_edges = []
    
    for u in range(n):
        if time.time() > deadline - 80:
            break
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            new_c4s = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        new_c4s += 1
            if new_c4s < best_new_c4s:
                best_new_c4s = new_c4s
                best_edge = [u, v]
            if new_c4s <= best_new_c4s + 2:
                candidate_edges.append((new_c4s, u, v))
    
    if best_edge is None:
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break
    
    u, v = best_edge
    adj[u].add(v)
    adj[v].add(u)
    current_edges = core_edges + [best_edge]
    
    # Incremental C4 counter using common-neighbor counts
    # cn[i][j] = number of common neighbors of i and j
    # Total C4s = sum_{i<j} C(cn[i][j], 2) / 2

    def build_cn_matrix(adj_sets):
        cn = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                common = adj_sets[i] & adj_sets[j]
                cn[i][j] = cn[j][i] = len(common)
        return cn

    def total_c4_from_cn(cn):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                c = cn[i][j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    def delta_c4_add_edge(cn, u, v, adj_sets):
        """Compute change in total C4s when adding edge (u,v) to current graph."""
        # Adding edge (u,v): for each pair (a,b) where a in N(u), b in N(v),
        # if a==b it's a triangle (not C4). For C4s, we need a != b.
        # New C4s formed: for each w in N(u) and z in N(v) with w!=z,
        # if w-z is an edge... no wait.
        # A new C4 is u-a-v-b-u where a in N(u)\{v}, b in N(v)\{u}, a!=b, and a-b is an edge.
        # Actually: adding edge (u,v) creates new C4s of the form u-x-v-y-u
        # where x in N(u), y in N(v), x!=y, and x-y is an edge.
        # But also we need x!=v, y!=u (already satisfied since no self-loops).
        # Also x!=y to avoid triangles being counted.
        # Wait, more carefully: a C4 containing the new edge (u,v) is u-a-v-b-u
        # where a is a neighbor of u (a!=v), b is a neighbor of v (b!=u),
        # a!=b, and (a,b) is an edge.
        # But this counts each C4 once (the new edge is unique to it).
        # However, we also need to account for changes in C4s NOT containing (u,v).
        # Actually, adding an edge only creates new C4s that contain that edge.
        # It doesn't destroy any existing C4s.
        # So delta = number of C4s containing the new edge (u,v).
        # A C4 containing (u,v): u-a-v-b-u where a in N(u)\{v}, b in N(v)\{u}, a!=b, edge(a,b) exists.
        # But we also need to check: is this a valid 4-cycle? u-a-v-b-u with all distinct vertices.
        # a!=u (no self-loop), b!=v (no self-loop), a!=v (excluded), b!=u (excluded), a!=b (excluded).
        # So yes, just count pairs (a,b) with a in N(u)\{v}, b in N(v)\{u}, a!=b, and a-b is edge.
        count = 0
        for a in adj_sets[u]:
            if a == v:
                continue
            for b in adj_sets[v]:
                if b == u or b == a:
                    continue
                if b in adj_sets[a]:
                    count += 1
        # Each C4 is counted once (it contains the new edge exactly once in this enumeration)
        # Wait, no. For a fixed new edge (u,v), a C4 u-a-v-b-u is counted once.
        # But actually we should double-check: the C4 is {u,v,a,b} with edges (u,a),(a,v),(v,b),(b,u).
        # In our enumeration: a is in N(u), b is in N(v), a!=b, and edge(a,b) exists.
        # This gives exactly one count per C4 containing edge (u,v).
        # So delta = count.
        return count

    def delta_c4_remove_edge(cn, u, v, adj_sets):
        """Compute change in total C4s when removing edge (u,v) from current graph.
        This equals -delta_c4_add_edge if we reverse the operation."""
        # Removing edge (u,v) destroys C4s that contain (u,v).
        # Same count as delta_c4_add_edge would give for adding it back.
        count = 0
        for a in adj_sets[u]:
            if a == v:
                continue
            for b in adj_sets[v]:
                if b == u or b == a:
                    continue
                if b in adj_sets[a]:
                    count += 1
        return -count

    cn = build_cn_matrix(adj)
    current_c4s = total_c4_from_cn(cn)

    edge_set = set()
    for e in current_edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
    edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]

    def update_cn_remove_edge(cn, u, v, adj_sets):
        for a in adj_sets[u]:
            if a == v:
                continue
            cn[a][v] -= 1
            cn[v][a] -= 1
        for b in adj_sets[v]:
            if b == u:
                continue
            cn[u][b] -= 1
            cn[b][u] -= 1

    def update_cn_add_edge(cn, u, v, adj_sets):
        for a in adj_sets[u]:
            if a == v:
                continue
            cn[a][v] += 1
            cn[v][a] += 1
        for b in adj_sets[v]:
            if b == u:
                continue
            cn[u][b] += 1
            cn[b][u] += 1

    no_improve_count = 0
    max_no_improve = 5
    best_c4s_ever = current_c4s
    best_edges_ever = list(current_edges)
    best_adj_ever = [set(s) for s in adj]

    while time.time() < deadline - 15:
        if no_improve_count >= max_no_improve:
            break

        # Rebuild non_edges and edge_list
        edge_set = set()
        for e in current_edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]
        non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]

        best_local_swap = None
        best_local_c4s = current_c4s

        for r_e in edge_list:
            if time.time() > deadline - 15:
                break
            r_u, r_v = r_e

            # Compute C4s after removing r_e using delta
            delta_remove = delta_c4_remove_edge(cn, r_u, r_v, adj)
            c4_after_remove = current_c4s + delta_remove

            # Temporarily remove r_e from adj and cn
            adj[r_u].discard(r_v)
            adj[r_v].discard(r_u)
            update_cn_remove_edge(cn, r_u, r_v, adj)

            for a_e in non_edges:
                if time.time() > deadline - 15:
                    break
                a_u, a_v = a_e
                if a_e == r_e:
                    continue

                # Compute C4s after adding a_e using delta
                delta_add = delta_c4_add_edge(cn, a_u, a_v, adj)
                new_c4s = c4_after_remove + delta_add

                if new_c4s < best_local_c4s:
                    best_local_c4s = new_c4s
                    best_local_swap = (r_e, a_e)
                    if new_c4s == 0:
                        break

            # Restore r_e
            adj[r_u].add(r_v)
            adj[r_v].add(r_u)
            update_cn_add_edge(cn, r_u, r_v, adj)

            if best_local_c4s == 0:
                break

        if best_local_swap is not None:
            r_e, a_e = best_local_swap
            r_u, r_v = r_e
            a_u, a_v = a_e

            adj[r_u].discard(r_v)
            adj[r_v].discard(r_u)
            update_cn_remove_edge(cn, r_u, r_v, adj)

            adj[a_u].add(a_v)
            adj[a_v].add(a_u)
            update_cn_add_edge(cn, a_u, a_v, adj)

            current_edges = [e for e in current_edges
                            if (min(e[0],e[1]), max(e[0],e[1])) != r_e]
            current_edges.append([a_u, a_v])

            current_c4s = best_local_c4s
            no_improve_count = 0
            if current_c4s < best_c4s_ever:
                best_c4s_ever = current_c4s
                best_edges_ever = list(current_edges)
                best_adj_ever = [set(s) for s in adj]
        else:
            no_improve_count += 1

    # If we improved, use the best found; otherwise keep current
    if best_c4s_ever < current_c4s:
        current_edges = best_edges_ever
        current_c4s = best_c4s_ever
        adj = best_adj_ever

    # Verify
    assert len(current_edges) == 91
    final_edge_set = set(tuple(sorted(e)) for e in current_edges)
    assert len(final_edge_set) == 91

    return current_edges

def _greedy_construct():
    """Greedy construction: add edges one by one, each time choosing the edge
    that adds the fewest new C4s."""
    import random
    random.seed(42)
    
    t0 = time.time()
    dl = t0 + 105.0
    n = 31
    target = 91
    
    adj = [set() for _ in range(n)]
    edges = []
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    def count_new_c4s(u, v):
        cnt = 0
        for a in adj[u]:
            for b in adj[v]:
                if b in adj[a]:
                    cnt += 1
        return cnt
    
    remaining = list(all_edges)
    
    for _ in range(target):
        if time.time() > dl:
            break
        best_e = None
        best_cnt = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            cnt = count_new_c4s(u, v)
            if cnt < best_cnt:
                best_cnt = cnt
                best_e = e
            if cnt == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    while len(edges) < target and remaining:
        e = remaining.pop()
        u, v = e
        if v not in adj[u]:
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
    
    return edges[:target]
