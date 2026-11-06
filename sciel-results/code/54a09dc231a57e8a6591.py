import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 12  # safety margin before the 150s wall-clock kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[c] & m) != 0:
                            return False
        return True

    def max_tri_free(adj_mask):
        """Exact size of the largest triangle-free induced subgraph (branch & bound)."""
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if c > b:
                            tris.append((a, b, c))
        
        if not tris:
            return N
            
        best = [0]
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        
        # Precompute vertex to triangle membership
        vertex_to_tris = [[] for _ in range(N)]
        for i, tm in enumerate(tri_masks):
            m = tm
            while m:
                bit = m & -m
                v = bit.bit_length() - 1
                vertex_to_tris[v].append(i)
                m ^= bit
        
        # Sort triangles: prioritize those where removing any one vertex breaks the most triangles
        # This maximizes the "constraint propagation" effect
        def tri_priority(tm):
            a = (tm & -tm).bit_length() - 1
            rest = tm ^ (1 << a)
            b = (rest & -rest).bit_length() - 1
            c = (tm ^ (1 << a) ^ (1 << b)).bit_length() - 1
            # Score = sum of triangle counts of the three vertices (higher = more central)
            # Also add a bonus if the three vertices are all mutually well-connected
            mutual = 0
            if (adj_mask[a] >> b) & 1: mutual += 1
            if (adj_mask[a] >> c) & 1: mutual += 1
            if (adj_mask[b] >> c) & 1: mutual += 1
            score = len(vertex_to_tris[a]) + len(vertex_to_tris[b]) + len(vertex_to_tris[c])
            # Prefer triangles whose vertices have high degree in the whole graph
            score += (adj_mask[a].bit_count() + adj_mask[b].bit_count() + adj_mask[c].bit_count()) // 3
            return -score
        
        tri_masks.sort(key=tri_priority)
        
        # Rebuild vertex_to_tris after sorting
        vertex_to_tris = [[] for _ in range(N)]
        for i, tm in enumerate(tri_masks):
            m = tm
            while m:
                bit = m & -m
                v = bit.bit_length() - 1
                vertex_to_tris[v].append(i)
                m ^= bit
        
        # Precompute for each triangle, the list of other triangles it conflicts with
        # (share at least 2 vertices) - helps with pruning
        tri_conflicts = [[] for _ in range(len(tri_masks))]
        for i in range(len(tri_masks)):
            for j in range(i+1, len(tri_masks)):
                shared = tri_masks[i] & tri_masks[j]
                if shared.bit_count() >= 2:
                    tri_conflicts[i].append(j)
                    tri_conflicts[j].append(i)
        
        def bnb(avail, depth=0):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            
            # Find a triangle in avail
            tri = None
            for tm in tri_masks:
                if (tm & avail) == tm:
                    tri = tm
                    break
                    
            if tri is None:
                best[0] = cnt
                return
                
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            rest = tri ^ a_bit
            b_bit = rest & -rest
            b = b_bit.bit_length() - 1
            c_bit = tri ^ a_bit ^ b_bit
            c = c_bit.bit_length() - 1
            
            # Heuristic: remove vertex with highest degree in avail subgraph
            da = (avail & adj_mask[a]).bit_count()
            db = (avail & adj_mask[b]).bit_count()
            dc = (avail & adj_mask[c]).bit_count()
            
            verts = sorted([(da, a), (db, b), (dc, c)], reverse=True)
            
            for _, v in verts:
                new_avail = avail & ~(1 << v)
                # Prune: if removing v doesn't break any triangle, we still need to recurse
                # but if cnt-1 <= best[0], we can stop
                bnb(new_avail, depth + 1)
                if cnt - 1 <= best[0]:
                    return

        bnb(FULL)
        return best[0]

    def make_paley_17_plus_isolated():
        adj_mask = [0] * N
        residues = set([1, 2, 4, 8, 9, 13, 15, 16])
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def make_paley_17_connected():
        """Paley(17) on vertices 0-16, vertex 17 connected to some vertices."""
        adj_mask = make_paley_17_plus_isolated()
        for v in [0, 3, 7, 11]:
            adj_mask[17] |= (1 << v)
            adj_mask[v] |= (1 << 17)
        return adj_mask

    def make_turan_3():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def make_strongly_regular_seed():
        """
        Build a graph from a strongly regular-like construction.
        Use the Clebsch graph complement or similar high-symmetry structure.
        Here we use a circulant graph with carefully chosen offsets to approximate
        strongly regular properties (uniform triangle density).
        """
        # Try multiple circulant constructions with different offset sets
        # that aim for uniform triangle counts
        candidates = [
            [1, 2, 3, 7, 8, 9],      # symmetric, moderate density
            [1, 2, 4, 7, 8, 10],     
            [1, 3, 5, 7, 9, 11],     # odd offsets only
            [2, 3, 5, 7, 8, 10],
            [1, 2, 3, 4, 8, 9],
            [1, 3, 4, 6, 8, 10],
        ]
        results = []
        for offsets in candidates:
            adj = [0] * N
            for i in range(N):
                for off in offsets:
                    j = (i + off) % N
                    if j > i:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if is_k4_free(adj):
                results.append(adj)
        return results

    def make_random_regular_seed(degree, trials=5):
        """Generate random regular graphs and check K4-free."""
        results = []
        for _ in range(trials):
            adj = [0] * N
            # Generate random degree-regular graph
            for i in range(N):
                neighbors = set()
                attempts = 0
                while len(neighbors) < degree and attempts < 100:
                    j = rng.randint(0, N-1)
                    if j != i and j not in neighbors:
                        neighbors.add(j)
                    attempts += 1
                for j in neighbors:
                    adj[i] |= (1 << j)
            # Symmetrize
            adj_sym = [0] * N
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1 or (adj[j] >> i) & 1:
                        adj_sym[i] |= (1 << j)
                        adj_sym[j] |= (1 << i)
            if is_k4_free(adj_sym):
                results.append(adj_sym)
        return results

    def flip(adj_mask, u, v):
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)

    def flip2(adj_mask, u1, v1, u2, v2):
        adj_mask[u1] ^= (1 << v1)
        adj_mask[v1] ^= (1 << u1)
        adj_mask[u2] ^= (1 << v2)
        adj_mask[v2] ^= (1 << u2)

    best_score = N + 1
    best_mask = None

    # Generate diverse seeds
    seeds = []
    seeds.append(make_paley_17_plus_isolated())
    seeds.append(make_paley_17_connected())
    seeds.append(make_turan_3())
    
    # Strongly regular / high-symmetry seeds
    seeds.extend(make_strongly_regular_seed())
    
    # Random regular seeds
    for deg in [5, 6, 7, 8]:
        seeds.extend(make_random_regular_seed(deg, trials=3))
    
    # Try different cyclic constructions
    for offsets in [[2,3,5,7,11,13], [2,4,6,8,10,14], [1,4,6,9,11,16], [2,5,7,10,12,15],
                    [1,2,5,6,7,10], [1,3,4,8,9,11]]:
        adj = [0]*N
        for i in range(N):
            for off in offsets:
                j = (i + off) % N
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_k4_free(adj):
            seeds.append(adj)
    
    # Random K4-free seeds
    for d in [0.30, 0.40, 0.50, 0.60, 0.70]:
        for _ in range(4):
            adj = [0]*N
            for i in range(N):
                for j in range(i+1, N):
                    if rng.random() < d:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if is_k4_free(adj):
                seeds.append(adj)
    
    # Evaluate all seeds
    for seed in seeds:
        if time.time() >= deadline:
            break
        if not is_k4_free(seed):
            continue
        cur = max_tri_free(seed)
        if cur < best_score:
            best_score = cur
            best_mask = seed[:]

    # Local search with 1-edge and 2-edge flips
    if best_mask is not None:
        adj = best_mask[:]
        cur = best_score
        temperature = 1.5
        cooling = 0.997
        no_improve = 0
        max_no_improve = 300
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Choose move type: 60% single flip, 40% double flip
            if rng.random() < 0.6:
                u, v = rng.sample(range(N), 2)
                flip(adj, u, v)
                move_type = 1
            else:
                u1, v1, u2, v2 = rng.sample(range(N), 4)
                flip2(adj, u1, v1, u2, v2)
                move_type = 2
            
            if is_k4_free(adj):
                new_score = max_tri_free(adj)
                delta = new_score - cur
                
                if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.05)):
                    cur = new_score
                    if cur < best_score:
                        best_score = cur
                        best_mask = adj[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    if move_type == 1:
                        flip(adj, u, v)
                    else:
                        flip2(adj, u1, v1, u2, v2)
                    no_improve += 1
            else:
                if move_type == 1:
                    flip(adj, u, v)
                else:
                    flip2(adj, u1, v1, u2, v2)
                no_improve += 1
                
            temperature *= cooling
            if temperature < 0.1:
                break

    # Final polish: try all single-edge flips from best solution
    if best_mask is not None and time.time() < deadline:
        adj = best_mask[:]
        cur = best_score
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
    
    # Second polish pass if time remains
    if best_mask is not None and time.time() < deadline - 5:
        adj = best_mask[:]
        cur = best_score
        for i in range(N):
            for j in range(i+1, N):
                if time.time() >= deadline - 2:
                    break
                flip(adj, i, j)
                if is_k4_free(adj):
                    new_score = max_tri_free(adj)
                    if new_score < cur:
                        cur = new_score
                        best_score = cur
                        best_mask = adj[:]
                flip(adj, i, j)

    if best_mask is None:
        best_mask = make_paley_17_plus_isolated()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
