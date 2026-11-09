import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def verify_c4_free(n, edges):
        """Verify no C4 exists in the graph."""
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        # Check for C4: for each pair of vertices, count common neighbors
        # A C4 exists iff two vertices share >= 2 common neighbors
        # AND those two vertices are not adjacent (otherwise it's a triangle+C4)
        # Actually, C4 = 4-cycle: v1-v2-v3-v4-v1
        # This means v1 and v3 share v2 and v4 as common neighbors
        # So: for any pair (u,v) with u!=v, if |N(u) ∩ N(v)| >= 2, we need to check
        # if there exist two common neighbors a,b such that a-b is an edge (forming C4 u-a-v-b-u)
        # OR if u-v is NOT an edge and |N(u) ∩ N(v)| >= 2, then u-a-v-b-u is a C4
        # Wait, that's not quite right. Let me think again.
        # C4: vertices a,b,c,d with edges ab, bc, cd, da.
        # a and c are opposite, b and d are opposite.
        # a and c share b and d as common neighbors.
        # b and d share a and c as common neighbors.
        # So: if any pair of vertices (x,y) has >= 2 common neighbors, AND x,y are NOT adjacent,
        # then we have a C4 (x - n1 - y - n2 - x).
        # If x,y ARE adjacent, then x-n1-y-n2-x is still a C4 if n1 != n2.
        # Actually no: if x-y is an edge, and x shares n1,n2 with y, then x-n1-y is a triangle,
        # and x-n2-y is another triangle. The cycle x-n1-y-n2-x requires edge n1-n2 or not?
        # x-n1-y-n2-x: edges are x-n1, n1-y, y-n2, n2-x. This is a 4-cycle iff n1 != n2 and n1-n2 is NOT required.
        # Wait, the cycle is x -> n1 -> y -> n2 -> x. The edges are (x,n1), (n1,y), (y,n2), (n2,x).
        # For this to be a simple 4-cycle, we need x, n1, y, n2 all distinct.
        # x and y are distinct (given). n1 and n2 are distinct (they're two different common neighbors).
        # n1 != x (n1 is a neighbor of x, and graphs are simple, so n1 != x).
        # n1 != y (n1 is a neighbor of y, so n1 != y).
        # Similarly n2 != x, n2 != y.
        # So yes, if |N(x) ∩ N(y)| >= 2, there IS a C4.
        # Therefore: the graph is C4-free iff for ALL pairs (x,y), |N(x) ∩ N(y)| <= 1.
        
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
        return True

    def build_greedy(n_target, seed, strategy='ordered'):
        """Build a C4-free graph greedily, trying to achieve min degree >= n_target - 39."""
        rng = random.Random(seed)
        target_deg = n_target - 39
        if target_deg < 0:
            target_deg = 0
        
        adj = np.zeros((n_target, n_target), dtype=bool)
        degs = np.zeros(n_target, dtype=np.int32)
        
        # Precompute: for each pair (u,v), track common neighbor count
        common = np.zeros((n_target, n_target), dtype=np.int32)
        
        def can_add(u, v):
            if u == v or adj[u][v]:
                return False
            # Adding edge (u,v) creates a C4 iff there exists w such that
            # adj[u][w] and adj[v][w] (i.e., common[u][v] >= 1)
            # Because then u-w-v-u would be... wait, that's a triangle.
            # C4: u-w1-v-w2-u. We need w1 != w2, both adjacent to u and v.
            # So if common[u][v] >= 2, adding (u,v) creates C4.
            # If common[u][v] == 1, say w is the only common neighbor,
            # then u-w-v is a path of length 2. Adding u-v makes triangle u-w-v.
            # No C4 is created.
            # Wait, I need to re-examine.
            # Current graph is C4-free. We add edge (u,v).
            # A new C4 must use the new edge (u,v).
            # So the C4 is u - v - x - y - u, where (v,x), (x,y), (y,u) are existing edges.
            # This means x is a neighbor of v, y is a neighbor of u, and (x,y) is an edge.
            # Also x != y, x != u, x != v, y != u, y != v.
            # Equivalently: there exist x in N(v), y in N(u) with x != y and (x,y) is an edge.
            # Another way: the C4 is u - a - v - b - u where a in N(u)∩N(v)... no.
            # Let me think differently.
            # After adding (u,v), a C4 involving (u,v) has the form:
            # u - v - x - y - u
            # Edges: (u,v) new, (v,x) old, (x,y) old, (y,u) old.
            # So x is a neighbor of v (in old graph), y is a neighbor of u (in old graph),
            # and (x,y) is an edge in old graph.
            # Also we need u, v, x, y all distinct.
            # x != u: x is neighbor of v, and if x=u then (v,u) is an edge, but we're adding it, so in old graph (v,u) is not an edge. So x != u.
            # x != v: simple graph.
            # y != u: simple graph.
            # y != v: y is neighbor of u, if y=v then (u,v) is an edge, but in old graph it's not. So y != v.
            # x != y: if x=y, then we have u-v-x-u which is a triangle, not a 4-cycle.
            # So the condition is: there exist x in N(v) \ {u}, y in N(u) \ {v} with x != y and adj[x][y] = True.
            # 
            # Alternative characterization:
            # The new C4 uses edge (u,v) and three old edges.
            # The three old edges form a path of length 3 from u to v: u - y - x - v.
            # So we need a path of length 3 from u to v in the old graph.
            # Path u-y-x-v: (u,y) old edge, (y,x) old edge, (x,v) old edge.
            # So: there exist y in N(u), x in N(v) with y != x and (y,x) is an old edge.
            # 
            # Now, is this equivalent to common[u][v] >= 2?
            # common[u][v] = |N(u) ∩ N(v)|.
            # If w in N(u) ∩ N(v), then (u,w) and (w,v) are edges.
            # For a C4 u-v-x-w-u, we need (v,x) and (x,w) edges.
            # This is NOT the same as just having common neighbors.
            #
            # Let me reconsider. The standard fact:
            # A graph is C4-free iff for every pair of vertices u,v, |N(u) ∩ N(v)| <= 1.
            # This is a well-known characterization.
            # Proof: If |N(u) ∩ N(v)| >= 2, say a,b in N(u)∩N(v), then u-a-v-b-u is a 4-cycle.
            # Conversely, if there's a 4-cycle u-a-v-b-u, then a,b in N(u)∩N(v).
            # So YES, the characterization holds.
            #
            # Therefore, after adding (u,v), the graph is C4-free iff for all pairs (x,y),
            # |N(x) ∩ N(y)| <= 1 in the NEW graph.
            # The only pairs whose common neighbor count changes are:
            # - (u,v): common[u][v] in new graph = common[u][v] in old graph (since u not in N(u), v not in N(v))
            #   Wait, N(u) gains v, N(v) gains u.
            #   common[u][v] = |N(u) ∩ N(v)|.
            #   In new graph: N(u) = old N(u) ∪ {v}, N(v) = old N(v) ∪ {u}.
            #   N(u) ∩ N(v) = (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u})
            #   = (old N(u) ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v} ∩ old N(v)) ∪ ({v} ∩ {u})
            #   = (old N(u) ∩ old N(v)) ∪ ∅ ∪ ∅ ∪ ∅
            #   = old N(u) ∩ old N(v)
            #   So common[u][v] is unchanged.
            # - For w not in {u,v}: N(u) gains v, so if w is adjacent to v in old graph, then v becomes a common neighbor of u and w.
            #   Similarly, N(v) gains u, so if w is adjacent to u in old graph, then u becomes a common neighbor of v and w.
            #   So common[u][w] increases by 1 if adj[v][w] was true.
            #   common[v][w] increases by 1 if adj[u][w] was true.
            #   common[w][u] = common[u][w], common[w][v] = common[v][w].
            #
            # So to check if adding (u,v) preserves C4-freeness:
            # 1. common[u][v] must be <= 1 (unchanged, so just check current value).
            #    Actually, we need common[u][v] <= 1 in the new graph, which equals old common[u][v].
            #    So check common[u][v] <= 1.
            # 2. For all w not in {u,v}:
            #    - common[u][w] + adj[v][w] <= 1
            #    - common[v][w] + adj[u][w] <= 1
            #
            # This is the correct condition.
            if common[u][v] > 1:
                return False
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[v][w] and common[u][w] >= 1:
                    return False
                if adj[u][w] and common[v][w] >= 1:
                    return False
            return True

        def add_edge(u, v):
            adj[u][v] = adj[v][u] = True
            degs[u] += 1
            degs[v] += 1
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
                if adj[u][w]:
                    common[v][w] += 1
                    common[w][v] += 1

        # Greedy construction: try to add edges in a smart order
        # Strategy: for each vertex, try to connect it to as many other vertices as possible
        # while maintaining C4-freeness.
        
        # Initialize with no edges
        # Try to add edges in rounds
        max_rounds = 200
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            # Sort vertices by current degree (ascending) - boost low-degree vertices first
            verts = sorted(range(n_target), key=lambda x: degs[x])
            
            for u in verts:
                if degs[u] >= target_deg:
                    continue
                # Find candidates for u
                candidates = [v for v in range(n_target) if v != u and not adj[u][v]]
                # Sort candidates by degree (descending) - connect to high-degree vertices
                candidates.sort(key=lambda x: -degs[x])
                
                for v in candidates:
                    if can_add(u, v):
                        add_edge(u, v)
                        improved = True
                        if degs[u] >= target_deg:
                            break
            
            if not improved:
                break
        
        # Check if we achieved the target
        if np.min(degs) >= target_deg:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if adj[i][j]:
                        edges.append([i, j])
            # Final verification
            if verify_c4_free(n_target, edges):
                return {'n': n_target, 'edges': edges}
            else:
                return None
        return None

    def build_greedy_v2(n_target, seed):
        """Alternative greedy: build adjacency row by row with backtracking-lite approach."""
        rng = random.Random(seed)
        target_deg = n_target - 39
        if target_deg < 0:
            target_deg = 0
        
        adj = np.zeros((n_target, n_target), dtype=bool)
        degs = np.zeros(n_target, dtype=np.int32)
        common = np.zeros((n_target, n_target), dtype=np.int32)
        
        def can_add(u, v):
            if u == v or adj[u][v]:
                return False
            if common[u][v] > 1:
                return False
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[v][w] and common[u][w] >= 1:
                    return False
                if adj[u][w] and common[v][w] >= 1:
                    return False
            return True

        def add_edge(u, v):
            adj[u][v] = adj[v][u] = True
            degs[u] += 1
            degs[v] += 1
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
                if adj[u][w]:
                    common[v][w] += 1
                    common[w][v] += 1

        # Process vertices in a fixed order, adding edges to maximize degree
        # Use a "conflict-free" ordering: process vertices and for each, add edges to all possible
        # other vertices that haven't been fully processed yet
        
        # Try multiple orderings
        orderings = []
        orderings.append(list(range(n_target)))
        # Reverse
        orderings.append(list(range(n_target - 1, -1, -1)))
        # Random permutations
        for _ in range(20):
            perm = list(range(n_target))
            rng.shuffle(perm)
            orderings.append(perm)
        
        best_adj = None
        best_min_deg = -1
        
        for order in orderings:
            if time.time() > deadline:
                break
            
            adj_test = np.zeros((n_target, n_target), dtype=bool)
            degs_test = np.zeros(n_target, dtype=np.int32)
            common_test = np.zeros((n_target, n_target), dtype=np.int32)
            
            def can_add_test(u, v):
                if u == v or adj_test[u][v]:
                    return False
                if common_test[u][v] > 1:
                    return False
                for w in range(n_target):
                    if w == u or w == v:
                        continue
                    if adj_test[v][w] and common_test[u][w] >= 1:
                        return False
                    if adj_test[u][w] and common_test[v][w] >= 1:
                        return False
                return True

            def add_edge_test(u, v):
                adj_test[u][v] = adj_test[v][u] = True
                degs_test[u] += 1
                degs_test[v] += 1
                for w in range(n_target):
                    if w == u or w == v:
                        continue
                    if adj_test[v][w]:
                        common_test[u][w] += 1
                        common_test[w][u] += 1
                    if adj_test[u][w]:
                        common_test[v][w] += 1
                        common_test[w][v] += 1

            # Process in order: for each vertex, add edges to all later vertices if possible
            for i in range(n_target):
                u = order[i]
                # Try to add edge from u to all other vertices
                # Prefer vertices that are "later" in the order (not yet fully processed)
                for j in range(n_target):
                    if j == i:
                        continue
                    v = order[j]
                    if not adj_test[u][v]:
                        if can_add_test(u, v):
                            add_edge_test(u, v)
            
            min_deg = int(np.min(degs_test))
            if min_deg > best_min_deg:
                best_min_deg = min_deg
                best_adj = adj_test.copy()
            
            if best_min_deg >= target_deg:
                break
        
        if best_adj is not None and best_min_deg >= target_deg:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if best_adj[i][j]:
                        edges.append([i, j])
            if verify_c4_free(n_target, edges):
                return {'n': n_target, 'edges': edges}
        return None

    def build_from_complement(n_target, seed):
        """Build the complement graph (which should have no K_{1,39}) and derive the C4-free graph.
        The complement H has n vertices, and we need H to have no vertex of degree >= 39.
        Equivalently, every vertex in H has degree <= 38.
        The original graph G = complement of H must be C4-free.
        G is C4-free iff H has no two vertices with >= 2 common non-neighbors in H... 
        This is getting complex. Let's just build G directly."""
        return None

    best_result = None
    
    # Try n=47 first, then 46, then 45
    for n in [47, 46, 45]:
        if time.time() > deadline - 10:
            break
        
        # Try greedy v1 with multiple seeds
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            res = build_greedy(n, seed)
            if res:
                best_result = res
                break
        
        if best_result:
            break
        
        # Try greedy v2
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            res = build_greedy_v2(n, seed)
            if res:
                best_result = res
                break
        
        if best_result:
            break
    
    if not best_result:
        # Fallback: try a simpler approach - complete graph minus a matching or something
        # For n=39, min degree >= 0, so empty graph works
        # But let's try to do better
        # Try n=46 with a different strategy: start with complete graph and remove edges
        for n in [46, 45, 44]:
            if time.time() > deadline - 5:
                break
            # Start with complete graph, remove edges to make it C4-free
            # This is hard. Instead, try a structured graph.
            # Use the "polarity" idea but more carefully.
            # For q=6 (not prime power, but let's try): skip
            # For q=5: n=25+5+1=31, min degree 6. Too small.
            # For q=7: n=57, min degree 8. Need to take subgraph of size 46-47.
            
            # Let's try: take the q=7 polarity graph and find a large induced subgraph
            # with high minimum degree.
            q = 7
            points = []
            for x in range(q):
                for y in range(q):
                    points.append((x, y, 1))
            for x in range(q):
                points.append((x, 1, 0))
            points.append((1, 0, 0))
            
            n_pts = len(points)
            adj_full = np.zeros((n_pts, n_pts), dtype=bool)
            
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    p1 = points[i]
                    p2 = points[j]
                    dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                    if dot == 0:
                        adj_full[i][j] = adj_full[j][i] = True
            
            # Find a large induced subgraph with high min degree
            # Greedy: start with all vertices, remove the one with lowest degree in the subgraph
            # until we reach size n, tracking min degree.
            # Actually, we want to KEEP n vertices with max min degree.
            # Greedy: iteratively remove the vertex whose removal causes least damage to min degree.
            
            rng = random.Random(42)
            keep = set(range(n_pts))
            
            # We want to end up with n vertices
            while len(keep) > n:
                # For each vertex in keep, compute its degree within keep
                # Remove the vertex with highest degree (removing it hurts least for min degree of others)
                # Actually, we want to maximize min degree of the remaining set.
                # Heuristic: remove the vertex with the highest degree in the current subgraph.
                # This tends to keep the "core" of the graph.
                best_remove = None
                best_score = -1
                for v in keep:
                    deg_v = sum(1 for u in keep if u != v and adj_full[v][u])
                    # Score: how much would min degree drop if we remove v?
                    # Approximate: if v has low degree, removing it might not help.
                    # If v has high degree, removing it removes many edges.
                    # We want to remove the vertex that is "least critical" for maintaining high degrees.
                    # Heuristic: remove the vertex with the highest degree (it's connected to many,
                    # so its removal reduces degrees of many, but the remaining vertices still have
                    # decent connections among themselves).
                    # Alternative: remove the vertex with the lowest degree.
                    # Let's try both strategies.
                    score = deg_v  # Remove highest degree
                    if score > best_score:
                        best_score = score
                        best_remove = v
                
                if best_remove is not None:
                    keep.remove(best_remove)
            
            # Check min degree of the induced subgraph
            keep_list = sorted(keep)
            idx_map = {old: new for new, old in enumerate(keep_list)}
            adj_sub = np.zeros((n, n), dtype=bool)
            for old_i, new_i in idx_map.items():
                for old_j, new_j in idx_map.items():
                    if adj_full[old_i][old_j]:
                        adj_sub[new_i][new_j] = True
            
            degs_sub = np.array([int(np.sum(adj_sub[i])) for i in range(n)])
            min_deg = int(np.min(degs_sub))
            
            if min_deg >= n - 39:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_sub[i][j]:
                            edges.append([i, j])
                if verify_c4_free(n, edges):
                    best_result = {'n': n, 'edges': edges}
                    break
    
    if not best_result:
        # Absolute fallback
        best_result = {'n': 39, 'edges': []}
    
    return best_result
