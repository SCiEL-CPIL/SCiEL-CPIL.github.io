"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _edges_to_adj(edge_set, n=18):
    adj = [0] * n
    for (u, v) in edge_set:
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_adj(adj, n=18):
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            # any edge inside common -> K4
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    if adj[w] & common & ((1 << n) - 1):
                        # check if any two vertices in common are adjacent
                        c2 = common & ((1 << n) - 1)
                        # quick: for each x in c2, if adj[x] & c2 -> edge
                        # but we already know w in c2; need another neighbor in c2
                        if adj[w] & (c2 & ~(1 << w)):
                            return False
    return True

def _is_k4_free(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    return _is_k4_free_adj(adj, n)

def _max_triangle_free_subgraph_size(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    full = (1 << n) - 1

    # Precompute triangle masks: for each vertex, mask of vertices that form a triangle with it
    tri_with = [0] * n
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    tri_with[u] |= (1 << v) | (1 << w)
                    tri_with[v] |= (1 << u) | (1 << w)
                    tri_with[w] |= (1 << u) | (1 << v)

    # Greedy upper bound: repeatedly remove a vertex that is in the most remaining triangles
    # This gives an upper bound on the min vertex cover of triangles, hence lower bound on max TF size? No.
    # Actually: min vertex cover of triangles >= greedy removal count? Not necessarily.
    # We'll use greedy to get a candidate triangle-free set (vertices not removed), which is a lower bound on max TF size.
    # For upper bound, we use branch and bound.

    # Lower bound: greedy triangle-free set
    remaining = full
    removed_count = 0
    while True:
        # count triangles per vertex within remaining
        best_v = -1
        best_c = 0
        for v in range(n):
            if not (remaining >> v) & 1:
                continue
            # triangles containing v within remaining
            c = 0
            tv = tri_with[v] & remaining & ~(1 << v)
            # for each pair in tv, if edge between them -> triangle
            # cheaper: for each w in tv, if adj[w] & tv -> triangle with v,w,x
            # but that double counts; just count distinct triangles
            # Since n=18, do direct
            for w in range(v + 1, n):
                if not (tv >> w) & 1:
                    continue
                # x must be in tv, x > w, and w-x edge
                xmask = tv & adj[w] & ((1 << n) - 1) & ~(1 << v) & ~(1 << w)
                # count bits
                c += bin(xmask).count("1")
            if c > best_c:
                best_c = c
                best_v = v
        if best_c == 0:
            break
        remaining &= ~(1 << best_v)
        removed_count += 1

    lower_bound = bin(remaining).count("1")

    # If lower bound is already small, we can try to verify it's optimal via B&B
    # Branch and bound: find if there exists a triangle-free set of size > lower_bound
    # Equivalently, find min vertex cover of triangles; if min cover >= n - (lower_bound+1), then max TF <= lower_bound

    # B&B for minimum vertex cover of triangles (exact)
    # State: remaining triangles (as bitsets of vertices), current cover size
    # Upper bound on cover: greedy cover size (n - lower_bound)
    greedy_cover_size = n - lower_bound

    # Collect all triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    triangles.append((1 << u) | (1 << v) | (1 << w))

    if not triangles:
        return n

    # If greedy cover is 0, max TF = n
    if greedy_cover_size == 0:
        return n

    # B&B: can we cover all triangles with < greedy_cover_size vertices?
    # If yes, max TF > lower_bound. We want the maximum TF, so we want minimum cover.
    # We'll do B&B to find exact min cover, but with time limit.

    best_cover = greedy_cover_size
    deadline = time.time() + 2.0  # 2 second budget per evaluation

    def _bnb(remaining_tris, cover_mask, cover_size, depth):
        nonlocal best_cover
        if time.time() > deadline:
            return
        if not remaining_tris:
            if cover_size < best_cover:
                best_cover = cover_size
            return
        # Upper bound: even if we add 0 more, cover_size >= best_cover?
        if cover_size >= best_cover:
            return
        # Lower bound: need at least 1 more
        # Pick a triangle with fewest uncovered vertices
        tri = remaining_tris[0]
        # Branch on each vertex in tri not in cover
        for v in range(n):
            if (tri >> v) & 1 and not (cover_mask >> v) & 1:
                new_cover = cover_mask | (1 << v)
                new_rem = [t for t in remaining_tris if not (t & (1 << v))]
                _bnb(new_rem, new_cover, cover_size + 1, depth + 1)
                if time.time() > deadline:
                    return

    # Sort triangles by size (all size 3) but maybe by frequency for better pruning
    # Just use as is
    _bnb(triangles, 0, 0, 0)

    # The exact min cover might not be found if time runs out; best_cover is an upper bound on min cover
    # So max TF >= n - best_cover
    # We return the best lower bound we have, which is at least lower_bound
    # Actually, if B&B didn't prove optimality, we can only say max TF >= n - best_cover
    # But we want to return a value for scoring; we'll return max(lower_bound, n - best_cover)
    # Since best_cover <= greedy_cover_size, n - best_cover >= lower_bound
    return max(lower_bound, n - best_cover)

def _random_greedy_k4_free(n=18, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    adj = [0] * n
    edge_set = set()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        common = adj[u] & adj[v]
        # Check if adding u-v creates K4: need an edge within common
        # i.e., exists w,x in common with w-x edge
        ok = True
        cm = common
        while cm:
            lsb = cm & (-cm)
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~(1 << w)):
                ok = False
                break
            cm ^= lsb
        if ok:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
    
    return list(edge_set)

def _local_search(edges, n=18, max_iters=500, seed=None, deadline=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _edges_to_adj(edge_set, n)
    current_score = _max_triangle_free_subgraph_size(edge_set, n)
    best_edges = list(edge_set)
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try a random edge swap: remove one edge, add one
        if len(edge_set) > 0:
            rem_edge = rng.choice(list(edge_set))
        else:
            break
        
        cand = rng.choice(all_edges)
        
        new_set = edge_set - {rem_edge} | {cand}
        new_adj = _edges_to_adj(new_set, n)
        
        if not _is_k4_free_adj(new_adj, n):
            continue
        
        new_score = _max_triangle_free_subgraph_size(new_set, n)
        
        if new_score < current_score:
            current_score = new_score
            edge_set = new_set
            adj = new_adj
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
    
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 18
    
    # Start with the known good construction: complete tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    base_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    base_score = _max_triangle_free_subgraph_size(base_set, n)
    
    best_edges = base_edges
    best_score = base_score
    
    # Try random constructions + local search
    for trial in range(80):
        if time.time() > deadline:
            break
        
        # Build random K4-free graph
        edges = _random_greedy_k4_free(n, seed=trial)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Check it's valid
        if not _is_k4_free(edge_set, n):
            continue
        
        score = _max_triangle_free_subgraph_size(edge_set, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search with tighter deadline
        ls_deadline = min(deadline, start_time + 142)
        ls_edges, ls_score = _local_search(edges, n, max_iters=150, seed=trial, deadline=ls_deadline)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
        
        # Also try local search on the base
        if trial == 0:
            ls_base, ls_base_score = _local_search(base_edges, n, max_iters=250, seed=999, deadline=ls_deadline)
            if ls_base_score < best_score:
                best_score = ls_base_score
                best_edges = ls_base
    
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
