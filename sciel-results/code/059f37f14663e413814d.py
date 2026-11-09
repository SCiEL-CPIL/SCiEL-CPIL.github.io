import random
import time
import math

N = 18
ALLMASK = (1 << N) - 1

def is_k4_free(adj):
    """Fast K4-free check using bitmasks."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def largest_triangle_free(adj, deadline, limit=None):
    """
    Exact largest triangle-free induced subgraph via branch-and-bound.
    Uses triangle enumeration and independent set in conflict graph.
    """
    # Enumerate all triangles
    tri_edges = []
    for u in range(N):
        nbrs = adj[u] & ~((1 << (u + 1)) - 1)
        while nbrs:
            lsb = nbrs & -nbrs
            v = lsb.bit_length() - 1
            nbrs ^= lsb
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            while common:
                lsb2 = common & -common
                w = lsb2.bit_length() - 1
                common ^= lsb2
                tri_edges.append((u, v, w))
    m = len(tri_edges)
    if m == 0:
        return N

    # Conflict: two triangles share a vertex
    conf = [0] * m
    for i in range(m):
        u, v, w = tri_edges[i]
        mask_i = (1 << u) | (1 << v) | (1 << w)
        for j in range(i + 1, m):
            u2, v2, w2 = tri_edges[j]
            if mask_i & ((1 << u2) | (1 << v2) | (1 << w2)):
                conf[i] |= (1 << j)
                conf[j] |= (1 << i)

    # Order by degree for better pruning
    order = sorted(range(m), key=lambda x: conf[x].bit_count())
    pos = [0] * m
    for idx, t in enumerate(order):
        pos[t] = idx
    pconf_pos = [0] * m
    for i in range(m):
        cm = conf[i]
        pm = 0
        while cm:
            lsb = cm & -cm
            j = lsb.bit_length() - 1
            pm |= (1 << pos[j])
            cm ^= lsb
        pconf_pos[pos[i]] = pm
    tri_vertex = [(1 << u) | (1 << v) | (1 << w) for (u, v, w) in (tri_edges[t] for t in order)]

    best = 0

    def branch(avail, chosen):
        nonlocal best
        if time.time() > deadline:
            return
        ub = chosen + avail.bit_count()
        if ub <= best:
            return
        if avail == 0:
            if chosen > best:
                best = chosen
                if limit is not None and best >= limit:
                    return
            return
        # Pick vertex with max degree in avail
        av = avail
        best_v = -1
        best_deg = -1
        while av:
            lsb = av & -av
            v = lsb.bit_length() - 1
            deg = (pconf_pos[v] & avail).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            av ^= lsb
        v = best_v
        # Include
        new_avail = avail & ~pconf_pos[v] & ~(1 << v)
        branch(new_avail, chosen + 1)
        # Exclude
        branch(avail & ~(1 << v), chosen)

    branch(ALLMASK, 0)
    return best

def greedy_build_k4_free(deadline, seed):
    """
    Build a K4-free graph by adding edges one at a time,
    prioritizing edges that create many triangles (to minimize max TF subgraph).
    Uses a greedy strategy: at each step, add the edge that maximizes
    triangle creation while maintaining K4-freedom.
    """
    rng = random.Random(seed)
    adj = [0] * N
    
    # Start with a random sparse graph
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Try to add edges greedily
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_edges)
    
    for (i, j) in all_edges:
        if time.time() > deadline:
            break
        if adj[i] & (1 << j):
            continue
        # Check if adding edge (i,j) creates a K4
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        if not is_k4_free(adj):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
    
    # Now try to add more edges with a different order
    rng.shuffle(all_edges)
    for (i, j) in all_edges:
        if time.time() > deadline:
            break
        if adj[i] & (1 << j):
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        if not is_k4_free(adj):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
    
    return adj

def local_search_minimize_tf(adj, deadline):
    """
    Local search: flip edges to minimize the largest triangle-free induced subgraph.
    Uses a fast heuristic score instead of exact computation for speed.
    """
    def heuristic_score(adj):
        """
        Fast heuristic: count triangles and use a proxy for max TF subgraph.
        More triangles -> smaller max TF subgraph (roughly).
        Also penalize large independent sets.
        """
        # Count triangles
        tri_count = 0
        for u in range(N):
            nbrs = adj[u] & ~((1 << (u + 1)) - 1)
            while nbrs:
                lsb = nbrs & -nbrs
                v = lsb.bit_length() - 1
                nbrs ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                tri_count += common.bit_count()
        
        # Estimate max independent set (proxy for max TF subgraph)
        # Use a simple greedy: repeatedly pick vertex with min degree
        remaining = ALLMASK
        independent = 0
        while remaining:
            # Find vertex with min degree in remaining
            best_v = -1
            best_deg = N + 1
            rm = remaining
            while rm:
                lsb = rm & -rm
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                rm ^= lsb
            independent += 1
            remaining &= ~((1 << best_v) | adj[best_v])
        
        # Score: we want to MINIMIZE this
        # More triangles is better (smaller TF), larger independent set is worse
        return independent - tri_count * 0.1
    
    current_adj = adj[:]
    current_score = heuristic_score(current_adj)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        i, j = random.sample(range(N), 2)
        current_adj[i] ^= (1 << j)
        current_adj[j] ^= (1 << i)
        
        if not is_k4_free(current_adj):
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            continue
        
        new_score = heuristic_score(current_adj)
        if new_score < current_score:
            current_score = new_score
        else:
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
    
    return current_adj

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def adj_to_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_l = N + 1
    best_adj = None

    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]

    random.seed(42)

    # Phase 1: Generate diverse K4-free graphs
    candidates = []
    
    # Turan T(18,3)
    adj_turan = [0] * N
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts3[i]:
                for v in parts3[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    candidates.append(adj_turan)

    # Greedy builds with different seeds
    for seed in range(30):
        if time.time() > deadline - 60:
            break
        adj = greedy_build_k4_free(deadline - 60, seed)
        if is_k4_free(adj):
            candidates.append(adj)
    
    # Random dense repaired
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.55:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Repair K4s
        for _ in range(3000):
            if is_k4_free(adj):
                break
            # Find a K4 and remove an edge
            found = False
            for a in range(N):
                for b in range(a + 1, N):
                    if not (adj[a] >> b) & 1:
                        continue
                    common_ab = adj[a] & adj[b]
                    common_ab &= ~((1 << (b + 1)) - 1)
                    while common_ab:
                        lsb = common_ab & -common_ab
                        c = lsb.bit_length() - 1
                        common_abc = adj[a] & adj[b] & adj[c]
                        common_abc &= ~((1 << (c + 1)) - 1)
                        if common_abc:
                            lsb2 = common_abc & -common_abc
                            d = lsb2.bit_length() - 1
                            # Remove edge (a,d)
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                        common_ab ^= lsb
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        if is_k4_free(adj):
            candidates.append(adj)

    # Phase 2: Evaluate candidates with exact method
    for adj in candidates:
        if time.time() > deadline - 15:
            break
        l = largest_triangle_free(adj, deadline)
        update_best(adj, l)

    # Phase 3: Local search from best
    if best_adj is not None and time.time() < deadline - 5:
        current = best_adj[:]
        while time.time() < deadline - 1:
            # Local search step
            improved = local_search_minimize_tf(current, deadline - 1)
            if not is_k4_free(improved):
                break
            l = largest_triangle_free(improved, deadline)
            if l < best_l:
                update_best(improved, l)
                current = improved
            else:
                # Try a random perturbation
                i, j = random.sample(range(N), 2)
                current[i] ^= (1 << j)
                current[j] ^= (1 << i)
                if not is_k4_free(current):
                    current[i] ^= (1 << j)
                    current[j] ^= (1 << i)
                    continue
                l = largest_triangle_free(current, deadline)
                if l < best_l:
                    update_best(current, l)

    if best_adj is None:
        return adj_to_edges(candidates[0])

    return adj_to_edges(best_adj)
