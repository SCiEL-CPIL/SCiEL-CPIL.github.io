"""Construct 91 edges on 31 vertices minimizing 4-cycles.
Uses polarity graph of PG(2,5) for a C4-free 90-edge core,
then adds one edge and performs incremental local search (edge swaps) to minimize total C4s.
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
    
    cn = build_cn_matrix(adj)
    current_c4s = total_c4_from_cn(cn)
    
    edge_set = set()
    for e in current_edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
    edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]
    
    def update_cn_remove_edge(cn, u, v, adj_sets):
        """Update cn matrix when removing edge (u,v)."""
        # For each neighbor a of u, cn[a][v] decreases by 1 (if a != v)
        # For each neighbor b of v, cn[u][b] decreases by 1 (if b != u)
        # Also cn[u][v] changes: it was counting common neighbors including the edge itself? No.
        # cn[u][v] = |N(u) ∩ N(v)|. After removing edge (u,v), N(u) loses v, N(v) loses u.
        # The common neighbors don't change because u∉N(v) was already true (no self-loops) 
        # Wait, u IS in N(v) before removal. So cn[u][v] = |N(u) ∩ N(v)|.
        # Before: N(u) contains v, N(v) contains u. 
        # N(u) ∩ N(v) includes any w that is neighbor of both.
        # After removing (u,v): N(u) loses v, N(v) loses u.
        # w is in new N(u) ∩ N(w) iff w was in old N(u)∩N(v) and w != u and w != v.
        # Since u∉N(u) and v∉N(v), the only potential issue is if u was in N(v) (yes) 
        # and v was in N(u) (yes), but u∉N(u) so u∉N(u)∩N(v), similarly v∉N(u)∩N(v).
        # So cn[u][v] doesn't change!
        # But cn[a][v] for a in N(u)\{v}: a was common neighbor of... wait.
        # cn[a][v] = |N(a) ∩ N(v)|. Removing edge (u,v) removes u from N(v).
        # So if u was in N(a), then cn[a][v] decreases by 1.
        # Similarly cn[u][b] for b in N(v)\{u}: if b in N(u), cn[u][b] decreases by 1.
        
        # Remove u from N(v)'s perspective for cn updates
        for a in adj_sets[u]:
            if a == v:
                continue
            # cn[a][v]: u was a common neighbor of a and v iff u in N(a) and u in N(v)
            # u in N(v) is true (we're removing it), u in N(a) is true since a in N(u) and undirected
            # So u was a common neighbor of a and v. After removal, it's not.
            cn[a][v] -= 1
            cn[v][a] -= 1
        
        for b in adj_sets[v]:
            if b == u:
                continue
            # cn[u][b]: v was a common neighbor of u and b iff v in N(u) and v in N(b)
            # v in N(u) is true, v in N(b) is true since b in N(v) and undirected
            cn[u][b] -= 1
            cn[b][u] -= 1
    
    def update_cn_add_edge(cn, u, v, adj_sets):
        """Update cn matrix when adding edge (u,v)."""
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
    
    best_c4s = current_c4s
    best_swap = None
    no_improve_count = 0
    max_no_improve = 3
    
    while time.time() < deadline - 15:
        if no_improve_count >= max_no_improve:
            break
        
        # Rebuild non_edges and edge_list
        edge_set = set()
        for e in current_edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]
        non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
        
        found_improvement = False
        best_local_swap = None
        best_local_c4s = current_c4s
        
        for r_e in edge_list:
            if time.time() > deadline - 15:
                break
            r_u, r_v = r_e
            
            # Remove edge r_e
            adj[r_u].discard(r_v)
            adj[r_v].discard(r_u)
            update_cn_remove_edge(cn, r_u, r_v, adj)
            c4_after_remove = total_c4_from_cn(cn)
            
            for a_e in non_edges:
                if time.time() > deadline - 15:
                    break
                a_u, a_v = a_e
                if a_e == r_e:
                    continue
                
                # Add edge a_e
                adj[a_u].add(a_v)
                adj[a_v].add(a_u)
                update_cn_add_edge(cn, a_u, a_v, adj)
                new_c4s = total_c4_from_cn(cn)
                
                # Undo add
                adj[a_u].discard(a_v)
                adj[a_v].discard(a_u)
                update_cn_remove_edge(cn, a_u, a_v, adj)
                
                if new_c4s < best_local_c4s:
                    best_local_c4s = new_c4s
                    best_local_swap = (r_e, a_e)
                    if new_c4s == 0:
                        break
            
            # Undo remove
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
            found_improvement = True
            no_improve_count = 0
        else:
            no_improve_count += 1
    
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
