import random
import time
import itertools

DEADLINE = 145.0  # safety margin before 150s wall-clock kill


def is_k4_free(adj, n):
    """Check no 4-clique exists using bitsets."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            for c in range(b + 1, n):
                if (common >> c) & 1:
                    common2 = common & adj[c]
                    for d in range(c + 1, n):
                        if (common2 >> d) & 1:
                            return False
    return True


def max_triangle_free_size(adj, n):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    all_mask = (1 << n) - 1

    # Precompute: for each vertex, the set of vertices that form a triangle with it
    # tri_conflict[v] = bitmask of u such that {v,u} + some w forms triangle
    # Actually: if we include v and u (adjacent), we cannot include any common neighbor
    # For non-adjacent v,u, no restriction from that pair

    # Better: when building a triangle-free set, if we add v, we must remove all
    # vertices that form a triangle with two already-included vertices.
    # Simplest exact: branch on vertices, maintain candidate set.

    # Use the standard approach:
    # State: candidate set C (vertices that can still be added), current set S
    # Triangle-free means: for any v in S, no two neighbors of v are both in S.
    # Equivalent: S is an independent set in the "triangle graph" where
    # two vertices are connected if they share a common neighbor.

    # Build triangle graph: T[v] = set of u such that v and u have a common neighbor
    # (i.e., adding both v and u to a triangle-free set would create a triangle
    #  with that common neighbor, IF that common neighbor is also in the set)
    # Wait, that's not right. Triangle-free means no three vertices form a triangle.
    # So if v, u, w are all in S and all three edges exist, that's a triangle.
    # The constraint is: S contains no triangle.

    # Branch and bound: pick a vertex, include or exclude.
    # If we include v, we must exclude all pairs (u,w) where u~w and u~v and w~v.
    # That's expensive. Alternative:
    # If we include v, then for every neighbor u of v, we cannot include any
    # common neighbor of v and u.

    # Let's use a different formulation:
    # A set S is triangle-free iff for every v in S, S ∩ N(v) is an independent set.
    # 
    # Branch and bound with bitsets:
    # candidates = bitmask of vertices not yet decided
    # current_size = size of S so far
    # For each v we might add, we need to check it doesn't form a triangle
    # with two existing members.

    # Simpler exact approach for n=18:
    # Iterate over all subsets? 2^18 = 262144. That's feasible!

    # Check all subsets of size s from high to low
    # Actually, just check all subsets and find the largest triangle-free one.
    # 2^18 = 262144 subsets. For each, check if triangle-free.
    # A subset is triangle-free if it contains no K3.

    # Precompute for each triple whether it's a triangle
    # Then for each subset, check no triangle is contained.

    # Even better: use the complement. A subset S is triangle-free iff
    # the induced subgraph on S has no K3.

    # For n=18, iterate all 2^18 subsets, check triangle-free.
    # Precompute triangle masks: for each pair (u,v) with edge, 
    # common_neighbors[u][v] = set of w such that triangle u,v,w exists.

    # For a subset S, it contains a triangle iff there exist u,v,w in S
    # with all three edges. 

    # Fast check: for each v in S, check if S ∩ N(v) contains an edge.
    # Precompute for each v and each subset of N(v) whether it contains an edge.
    # N(v) has at most 17 elements, so 2^17 = 131072 per v. Total 18 * 131072 checks.
    # That's feasible but let's think of something simpler.

    # Actually for n=18, we can just do:
    # For each subset S (represented as bitmask), check if triangle-free.
    # Triangle-free check: for each v in S, the neighbors of v in S must be
    # an independent set (no edge between any two).
    
    # Precompute: for each v, and for each possible neighbor-subset mask,
    # whether that subset is independent in the graph.
    # But that's 18 * 2^17 which is a lot of precomputation.

    # Alternative: just iterate subsets and check triangles directly.
    # For subset S, check all triples in S. Worst case C(18,3) = 816 per subset.
    # 262144 * 816 ≈ 214 million. Too slow in Python.

    # Better: precompute for each pair (u,v) that are adjacent, the set of
    # common neighbors. Then subset S has a triangle iff there exist
    # adjacent u,v in S with a common neighbor also in S.
    # For each adjacent pair (u,v), common[u][v] is a bitmask.
    # S has triangle iff exists adjacent (u,v) in S with (common[u][v] & S) != 0.

    # Precompute adj_pairs: list of (u, v, common_mask) for all edges (u,v)
    adj_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                cm = adj[u] & adj[v]
                adj_pairs.append((u, v, cm))

    # Now check each subset
    # For subset mask S:
    #   for each (u,v,cm) in adj_pairs:
    #     if (S & (1<<u)) and (S & (1<<v)) and (S & cm):
    #       return False  # has triangle
    #   return True

    # Optimize: only check pairs where both u,v are in S.
    # Precompute for each vertex u, the list of (v, cm) where v>u and edge exists.

    # Let's just do it directly. For n=18, 2^18 = 262144 subsets.
    # Average number of edges ~ 81 (for K6,6,6 it's 108).
    # So ~108 checks per subset on average. 262144 * 108 ≈ 28M. Should be OK in ~10-20s.

    # But we want the MAXIMUM size. We can go from high to low.
    # For s from 18 down to 1:
    #   check if any subset of size s is triangle-free.
    #   if yes, return s.

    # Number of subsets of size s: C(18,s). Sum over all s is 2^18.
    # But going from 18 down, we stop early.

    # Precompute: for each vertex u, list of (v, common_mask) for v > u with edge
    v_edges = [[] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                cm = adj[u] & adj[v]
                v_edges[u].append((v, cm))
                v_edges[v].append((u, cm))

    def is_triangle_free(mask):
        # Check if subset 'mask' is triangle-free
        for u in range(n):
            if not (mask >> u) & 1:
                continue
            for v, cm in v_edges[u]:
                if (mask >> v) & 1:
                    if mask & cm:
                        return False
        return True

    # Search from large to small
    for s in range(n, 0, -1):
        # Iterate all subsets of size s
        # Use combinations
        for combo in itertools.combinations(range(n), s):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_triangle_free(mask):
                return s
    return 0


def max_triangle_free_size_fast(adj, n):
    """Faster version using precomputed triangle masks."""
    # For each pair of vertices (u,v), define tri_mask[u][v] = 
    # set of w such that {u,v,w} is a triangle.
    # A subset S is triangle-free iff for all u,v in S: S ∩ tri_mask[u][v] = empty
    # (when u,v are adjacent; if not adjacent, tri_mask is empty anyway)

    # Precompute
    tri = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                cm = adj[u] & adj[v]
                tri[u][v] = cm
                tri[v][u] = cm

    # For each vertex u, precompute the "forbidden" mask given u is in S:
    # If u is in S, then for any v in S ∩ N(u), we need S ∩ (N(u) ∩ N(v)) = empty
    # i.e., S ∩ tri[u][v] = empty for all v in S ∩ N(u)

    # Branch and bound:
    # candidates = set of vertices still available
    # We build S incrementally. When we add u to S, we must remove from candidates
    # all vertices v such that there exists w in S with {u,v,w} triangle.
    # That is: for each v in candidates, if there's w in S with tri[u][v] & (1<<w), 
    # then v cannot be added.
    # Simpler: when adding u, remove from candidates all v where 
    # (tri[u][v] & current_S_mask) != 0.

    best = [0]
    
    def bb(candidates, current_mask, size):
        if size > best[0]:
            best[0] = size
        if candidates == 0:
            return
        # Upper bound: size + popcount(candidates)
        if size + bin(candidates).count('1') <= best[0]:
            return
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidate subgraph
        u = -1
        max_deg = -1
        for i in range(n):
            if (candidates >> i) & 1:
                deg = bin(candidates & adj[i]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    u = i
        if u == -1:
            return
        
        # Branch 1: include u
        new_candidates = candidates & ~(1 << u)
        # Remove vertices that would form triangle with u and some vertex in current_mask
        for v in range(n):
            if (new_candidates >> v) & 1:
                if tri[u][v] & current_mask:
                    new_candidates &= ~(1 << v)
        bb(new_candidates, current_mask | (1 << u), size + 1)
        
        # Branch 2: exclude u
        bb(candidates & ~(1 << u), current_mask, size)

    bb((1 << n) - 1, 0, 0)
    return best[0]


def construct():
    start_time = time.time()
    n = 18
    best_graph = None
    best_score = 18  # worst case

    # Initial: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    # Build adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    score = max_triangle_free_size_fast(adj, n)
    if score < best_score:
        best_score = score
        best_graph = [list(e) for e in edges]
    
    # Randomized search: try to find better graphs
    random.seed(42)
    
    def random_k4_free():
        """Generate a random K4-free graph."""
        adj = [0] * n
        # Start with random edges, then remove any K4
        # Strategy: assign edges randomly with probability p, then fix K4s
        p = 0.4
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Remove K4s: while there's a K4, remove a random edge from it
        changed = True
        while changed:
            changed = False
            for a in range(n):
                if changed:
                    break
                for b in range(a + 1, n):
                    if changed:
                        break
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            common2 = common & adj[c]
                            for d in range(c + 1, n):
                                if (common2 >> d) & 1:
                                    # Found K4: a,b,c,d. Remove random edge.
                                    k4_edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                                    rem = random.choice(k4_edges)
                                    adj[rem[0]] &= ~(1 << rem[1])
                                    adj[rem[1]] &= ~(1 << rem[0])
                                    changed = True
                                    break
                            if changed:
                                break
                    if changed:
                        break
        return adj

    def local_improve(adj):
        """Try to reduce max triangle-free size by flipping edges."""
        current_score = max_triangle_free_size_fast(adj, n)
        improved = True
        while improved:
            improved = False
            # Try flipping each edge
            for u in range(n):
                for v in range(u + 1, n):
                    # Try adding edge
                    if not (adj[u] >> v) & 1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_k4_free(adj, n):
                            new_score = max_triangle_free_size_fast(adj, n)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                    # Try removing edge
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        new_score = max_triangle_free_size_fast(adj, n)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj, current_score

    # Try multiple random starts with local improvement
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        adj = random_k4_free()
        adj, score = local_improve(adj)
        if score < best_score:
            best_score = score
            best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
        
        # Also try: start from K6,6,6 and perturb
        if attempts % 3 == 0:
            adj2 = [0] * n
            for u, v in edges:
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
            # Randomly flip some edges
            for _ in range(random.randint(1, 5)):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if (adj2[u] >> v) & 1:
                        adj2[u] &= ~(1 << v)
                        adj2[v] &= ~(1 << u)
                    else:
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
            # Fix K4s
            changed = True
            while changed:
                changed = False
                for a in range(n):
                    if changed:
                        break
                    for b in range(a + 1, n):
                        if changed:
                            break
                        if not (adj2[a] >> b) & 1:
                            continue
                        common = adj2[a] & adj2[b]
                        for c in range(b + 1, n):
                            if (common >> c) & 1:
                                common2 = common & adj2[c]
                                for d in range(c + 1, n):
                                    if (common2 >> d) & 1:
                                        k4_edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                                        rem = random.choice(k4_edges)
                                        adj2[rem[0]] &= ~(1 << rem[1])
                                        adj2[rem[1]] &= ~(1 << rem[0])
                                        changed = True
                                        break
                                if changed:
                                    break
                        if changed:
                            break
            adj2, score2 = local_improve(adj2)
            if score2 < best_score:
                best_score = score2
                best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj2[u] >> v) & 1]

    # Final: try structured constructions
    # Turan graph T(18,3) = K6,6,6 already done.
    # Try other Turan-like: T(18,4) = K5,5,4,4 (but that's not K4-free necessarily)
    # Actually T(n,r) is K_{r+1}-free, so T(18,3) is K4-free.
    # T(18,4) would be K5-free but not necessarily K4-free.
    
    # Try: take K6,6,6 and add some edges within parts (carefully to avoid K4)
    if time.time() - start_time < DEADLINE:
        adj3 = [0] * n
        for u, v in edges:
            adj3[u] |= (1 << v)
            adj3[v] |= (1 << u)
        # Add edges within parts, but check K4-freedom
        for part_idx, part in enumerate(parts):
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, v = part[i], part[j]
                    # Try adding edge u-v
                    adj3[u] |= (1 << v)
                    adj3[v] |= (1 << u)
                    if not is_k4_free(adj3, n):
                        adj3[u] &= ~(1 << v)
                        adj3[v] &= ~(1 << u)
        
        score3 = max_triangle_free_size_fast(adj3, n)
        if score3 < best_score:
            best_score = score3
            best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj3[u] >> v) & 1]
    
    # Try more structured: blow-up of small K4-free graphs
    # C5 blown up: 18 = 5*3 + 3, so parts of size 3,3,3,3,3,3 (6 parts)
    # C5 is triangle-free, so blow-up is triangle-free... not what we want.
    # We want K4-free with small triangle-free induced subgraphs.
    # 
    # Try: complete multipartite with parts of unequal size
    # K5,5,4,4: is this K4-free? Yes, multipartite graphs are K_{k+1}-free
    # where k is number of parts. 4 parts → K5-free, but could have K4.
    # Actually a complete 4-partite graph can have K4 (one from each part).
    # So it's NOT K4-free in general.
    # Complete 3-partite: K4-free. That's what we have.
    
    # Try: 3-partite with unequal parts: K7,6,5
    if time.time() - start_time < DEADLINE:
        parts2 = [list(range(0, 7)), list(range(7, 13)), list(range(13, 18))]
        adj4 = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts2[i]:
                    for b in parts2[j]:
                        adj4[a] |= (1 << b)
                        adj4[b] |= (1 << a)
        score4 = max_triangle_free_size_fast(adj4, n)
        if score4 < best_score:
            best_score = score4
            best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj4[u] >> v) & 1]
    
    # Try K8,5,5
    if time.time() - start_time < DEADLINE:
        parts3 = [list(range(0, 8)), list(range(8, 13)), list(range(13, 18))]
        adj5 = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts3[i]:
                    for b in parts3[j]:
                        adj5[a] |= (1 << b)
                        adj5[b] |= (1 << a)
        score5 = max_triangle_free_size_fast(adj5, n)
        if score5 < best_score:
            best_score = score5
            best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj5[u] >> v) & 1]
    
    # Try K9,5,4
    if time.time() - start_time < DEADLINE:
        parts4 = [list(range(0, 9)), list(range(9, 14)), list(range(14, 18))]
        adj6 = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts4[i]:
                    for b in parts4[j]:
                        adj6[a] |= (1 << b)
                        adj6[b] |= (1 << a)
        score6 = max_triangle_free_size_fast(adj6, n)
        if score6 < best_score:
            best_score = score6
            best_graph = [[u, v] for u in range(n) for v in range(u + 1, n) if (adj6[u] >> v) & 1]

    return best_graph
