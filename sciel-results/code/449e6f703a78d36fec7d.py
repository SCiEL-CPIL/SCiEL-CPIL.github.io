import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size_fast(adj):
        """
        Fast exact computation of the largest induced triangle-free subgraph.
        Uses the fact that a set S is triangle-free iff for every triangle (a,b,c),
        S does not contain all of {a,b,c}.
        We use branch and bound with vertex ordering by degree in the conflict graph.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # For each vertex, precompute which triangles it belongs to
        tri_for_vertex = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v].append(idx)
        
        # Order vertices by number of triangles they participate in (descending)
        # This gives better pruning
        order = sorted(range(n), key=lambda v: len(tri_for_vertex[v]), reverse=True)
        
        # Precompute for each vertex in the order, the triangles it participates in
        # and for each such triangle, the other two vertices
        vertex_tri_info = []
        for v in order:
            info = []
            for t_idx in tri_for_vertex[v]:
                t = triangles[t_idx]
                others = t & ~(1 << v)
                # Find the two other vertices
                other_list = []
                for u in range(n):
                    if (others >> u) & 1:
                        other_list.append(u)
                info.append((t_idx, other_list[0], other_list[1]))
            vertex_tri_info.append(info)
        
        best_found = 0
        
        def dfs(pos, mask, size):
            nonlocal best_found
            if size + (n - pos) <= best_found:
                return
            if pos == n:
                if size > best_found:
                    best_found = size
                return
            
            v = order[pos]
            
            # Option 1: Don't include v
            dfs(pos + 1, mask, size)
            
            # Option 2: Include v
            # Check if adding v creates a triangle
            valid = True
            for t_idx, u, w in vertex_tri_info[pos]:
                # Check if both u and w are in mask
                if (mask >> u) & 1 and (mask >> w) & 1:
                    valid = False
                    break
            
            if valid:
                new_mask = mask | (1 << v)
                dfs(pos + 1, new_mask, size + 1)
        
        dfs(0, 0, 0)
        return best_found

    def remove_k4(adj):
        for _ in range(200):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    
    candidates = []
    
    # 1. K_{6,6,6} complete 3-partite graph
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)
    
    # 2. Three 6-cliques with inter-clique edges (no intra-clique edges)
    # This is K4-free if no three vertices from different cliques form a triangle
    # with a fourth. Actually, let's think more carefully.
    # If we have 3 cliques of size 6, and we add edges between them,
    # a K4 could form as: 2 from clique A, 1 from B, 1 from C, if all 6 cross-edges exist.
    # To be safe, let's use a sparse inter-clique pattern.
    # Actually, the simplest K4-free dense graph is the Turan graph T(18,3) = K_{6,6,6} which we already have.
    # Let's try: 3 groups of 6, within each group it's a clique, between groups use a bipartite-like pattern.
    # But cliques of size 6 contain K4s internally! So that won't work.
    
    # Instead: 6 groups of 3 vertices, make each group a triangle (K3), 
    # and connect groups with edges that don't create K4s.
    # Or: use the complement of a perfect matching on 18 vertices? No, that's dense.
    
    # Let's try: partition into 3 sets of 6. Within each set, no edges (independent set).
    # Between sets, add edges. This is the K_{6,6,6} case.
    
    # Alternative: Use a random regular-ish graph
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        density = 0.5 + random.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)
    
    # Evaluate candidates
    best_adj = None
    best_score = 18
    
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size_fast(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size_fast(best_adj)
        
    # Local Search with simulated annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score
    
    temperature = 1.0
    cooling = 0.995
    
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        if best_score <= 5:
            break
            
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        
        # Flip edge
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        new_score = max_triangle_free_size_fast(adj)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            adj = old_adj
            
        temperature *= cooling
        
    return edges_from_adj(best_adj)
