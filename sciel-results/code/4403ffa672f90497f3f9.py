import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 10  # safety margin before the 150 s wall-clock kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        # Optimized K4 check using bit operations
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    # Check if any two vertices in common are adjacent
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[c] & m) != 0:
                            return False
        return True

    def max_tri_free(adj_mask):
        """Exact size of the largest triangle-free induced subgraph (branch & bound)."""
        # Find triangles
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:
                            tris.append((a, b, c))
                        m &= m - 1
        
        if not tris:
            return N
            
        best = [0]
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        
        # Precompute: for each vertex, bitmask of triangles it participates in
        n_tris = len(tri_masks)
        vertex_tri_mask = [0] * N
        for i, tm in enumerate(tri_masks):
            m = tm
            while m:
                bit = m & -m
                v = bit.bit_length() - 1
                vertex_tri_mask[v] |= (1 << i)
                m ^= bit
        
        # Compute triangle adjacency: two triangles conflict if they share a vertex
        # tri_adj[i] = bitmask of triangles that conflict with triangle i
        tri_adj = [0] * n_tris
        for i in range(n_tris):
            ti = tri_masks[i]
            for j in range(i+1, n_tris):
                tj = tri_masks[j]
                if ti & tj:  # share at least one vertex
                    tri_adj[i] |= (1 << j)
                    tri_adj[j] |= (1 << i)
        
        # Greedy upper bound: max independent set in the conflict graph of triangles
        # A set of vertices is triangle-free iff the set of triangles it contains is independent
        # Upper bound on triangle-free set size = N - min_vertex_cover_of_triangles
        # But we use: if we have k triangles all present, we need to hit each with at least 1 vertex
        # So max triangle-free <= N - (min vertices to hit all triangles in avail)
        # Greedy: repeatedly pick vertex that hits most remaining triangles
        def greedy_upper_bound(avail):
            # Get triangles fully contained in avail
            active_tris = []
            for i, tm in enumerate(tri_masks):
                if (tm & avail) == tm:
                    active_tris.append(i)
            
            if not active_tris:
                return avail.bit_count()
            
            # Greedy set cover: pick vertices to hit all active triangles
            remaining = set(active_tris)
            hits = 0
            while remaining:
                # Find vertex in avail that hits most remaining triangles
                best_v = -1
                best_count = 0
                m = avail
                while m:
                    bit = m & -m
                    v = bit.bit_length() - 1
                    m ^= bit
                    # Count how many remaining triangles contain v
                    cnt = 0
                    for t in remaining:
                        if (tri_masks[t] >> v) & 1:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v < 0:
                    break
                hits += 1
                # Remove all triangles containing best_v
                new_remaining = set()
                for t in remaining:
                    if not ((tri_masks[t] >> best_v) & 1):
                        new_remaining.add(t)
                remaining = new_remaining
            
            return avail.bit_count() - hits
        
        # Memoization for B&B
        memo = {}
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return cnt
            
            # Check memo
            if avail in memo:
                return memo[avail]
            
            # Upper bound: if greedy bound <= best, prune
            ub = greedy_upper_bound(avail)
            if ub <= best[0]:
                memo[avail] = ub
                return ub
            
            # Find a triangle in avail
            tri = None
            tri_idx = -1
            for i, tm in enumerate(tri_masks):
                if (tm & avail) == tm:
                    tri = tm
                    tri_idx = i
                    break
                    
            if tri is None:
                memo[avail] = cnt
                best[0] = max(best[0], cnt)
                return cnt
            
            # Extract the three vertices
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            rest = tri ^ a_bit
            b_bit = rest & -rest
            b = b_bit.bit_length() - 1
            c_bit = tri ^ a_bit ^ b_bit
            c = c_bit.bit_length() - 1
            
            # Branch: for each vertex in the triangle, exclude it
            # Order: try excluding vertex that appears in most triangles first (more pruning)
            da = vertex_tri_mask[a].bit_count()
            db = vertex_tri_mask[b].bit_count()
            dc = vertex_tri_mask[c].bit_count()
            
            verts = sorted([(da, a), (db, b), (dc, c)], reverse=True)
            
            result = 0
            for _, v in verts:
                sub = bnb(avail & ~(1 << v))
                result = max(result, sub)
                if cnt - 1 <= best[0]:
                    break
            
            memo[avail] = result
            return result

        bnb(FULL)
        return best[0]

    def make_turan_3():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def make_paley_17_plus_isolated():
        """Paley(17) graph on vertices 0-16, vertex 17 isolated."""
        adj_mask = [0] * N
        # Paley(17): connect i,j if (j-i) mod 17 is a quadratic residue mod 17
        # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
        residues = set([1, 2, 4, 8, 9, 13, 15, 16])
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj_mask

    def make_structured_cyclic():
        adj_mask = [0] * N
        offsets = [2, 3, 5, 7, 11, 13]
        for i in range(N):
            for off in offsets:
                j = (i + off) % N
                if j > i:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def flip(adj_mask, u, v):
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)

    best_score = N + 1
    best_mask = None

    # Add Paley(17) + isolated as a strong seed
    seeds = []
    seeds.append(make_paley_17_plus_isolated())
    seeds.append(make_turan_3())
    seeds.append(make_structured_cyclic())
    
    for d in [0.4, 0.5, 0.6, 0.7, 0.8]:
        adj = [0]*N
        for i in range(N):
            for j in range(i+1, N):
                if rng.random() < d:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_k4_free(adj):
            seeds.append(adj)

    # Evaluate seeds first
    for seed in seeds:
        if time.time() >= deadline:
            break
        if not is_k4_free(seed):
            continue
        cur = max_tri_free(seed)
        if cur < best_score:
            best_score = cur
            best_mask = seed[:]

    # Local search from best seed
    if best_mask is not None:
        adj = best_mask[:]
        cur = best_score
        
        # Simulated Annealing with 1-opt and 2-opt moves
        temperature = 2.0
        cooling = 0.99
        steps = 0
        max_steps = 5000
        no_improve = 0
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            
            # Choose move type: 80% 1-opt, 20% 2-opt
            if rng.random() < 0.8:
                # 1-opt: flip one edge
                u, v = rng.sample(range(N), 2)
                flip(adj, u, v)
                
                if is_k4_free(adj):
                    new_score = max_tri_free(adj)
                    delta = new_score - cur
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.01)):
                        cur = new_score
                        if cur < best_score:
                            best_score = cur
                            best_mask = adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        flip(adj, u, v)
                        no_improve += 1
                else:
                    flip(adj, u, v)
                    no_improve += 1
            else:
                # 2-opt: flip two edges (u,v) and (w,x) where all four vertices distinct
                try:
                    u, v, w, x = rng.sample(range(N), 4)
                    flip(adj, u, v)
                    flip(adj, w, x)
                    
                    if is_k4_free(adj):
                        new_score = max_tri_free(adj)
                        delta = new_score - cur
                        
                        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.01)):
                            cur = new_score
                            if cur < best_score:
                                best_score = cur
                                best_mask = adj[:]
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            flip(adj, w, x)
                            flip(adj, u, v)
                            no_improve += 1
                    else:
                        flip(adj, w, x)
                        flip(adj, u, v)
                        no_improve += 1
                except ValueError:
                    # Not enough distinct vertices, fall back to 1-opt
                    u, v = rng.sample(range(N), 2)
                    flip(adj, u, v)
                    if is_k4_free(adj):
                        new_score = max_tri_free(adj)
                        delta = new_score - cur
                        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.01)):
                            cur = new_score
                            if cur < best_score:
                                best_score = cur
                                best_mask = adj[:]
                                no_improve = 0
                        else:
                            flip(adj, u, v)
                    else:
                        flip(adj, u, v)
                    no_improve += 1
            
            temperature *= cooling
            if temperature < 0.1:
                break
                
        if cur < best_score:
            best_score = cur
            best_mask = adj[:]

    # Final polish: try all single-edge flips and 2-opt moves from best solution
    if best_mask is not None and time.time() < deadline:
        adj = best_mask[:]
        cur = best_score
        
        # 1-opt polish
        for i in range(N):
            for j in range(i+1, N):
                if time.time() >= deadline:
                    break
                flip(adj, i, j)
                if is_k4_free(adj):
                    new_score = max_tri_free(adj)
                    if new_score < cur:
                        cur = new_score
                        best_score = cur
                        best_mask = adj[:]
                flip(adj, i, j)
        
        # 2-opt polish: try swapping pairs of edges
        if time.time() < deadline:
            for i in range(N):
                for j in range(i+1, N):
                    if time.time() >= deadline:
                        break
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        for l in range(k+1, N):
                            if l == i or l == j or l == k:
                                continue
                            if time.time() >= deadline:
                                break
                            flip(adj, i, j)
                            flip(adj, k, l)
                            if is_k4_free(adj):
                                new_score = max_tri_free(adj)
                                if new_score < cur:
                                    cur = new_score
                                    best_score = cur
                                    best_mask = adj[:]
                            flip(adj, k, l)
                            flip(adj, i, j)

    if best_mask is None:
        best_mask = make_paley_17_plus_isolated()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
