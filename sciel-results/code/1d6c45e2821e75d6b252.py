import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """Construct a K4-free graph on 18 vertices minimizing the size of the
    largest triangle-free induced subgraph."""
    n = 18
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Vertices 0..17
    all_vertices = list(range(n))

    # --- Helper: check K4-free using bitmask representation ---
    # adj[i] is a bitmask of neighbors of vertex i
    def is_k4_free(adj):
        # For each pair of adjacent vertices, check if they share a common neighbor
        # that is also adjacent to both -> forms K4 with the pair
        for i in range(n):
            ai = adj[i]
            # For each neighbor j of i where j > i
            j_mask = ai & (1 << j) if False else 0
            # Iterate neighbors
            neighbors = []
            m = ai
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                neighbors.append(j)
                m ^= lsb
            for j in neighbors:
                if j <= i:
                    continue
                # Common neighbors of i and j
                common = ai & adj[j]
                # Check if any common neighbor k is adjacent to both i and j
                # (it is by definition), so we just need to check that
                # k is also adjacent to i and j, which is already true.
                # A K4 is {i, j, k, l} where all pairs are edges.
                # Actually, for K4 we need 4 vertices all mutually adjacent.
                # So pick i, j (edge), then common neighbors of i and j.
                # If two common neighbors k, l are also adjacent to each other,
                # then {i,j,k,l} is a K4.
                # Simpler: for each edge (i,j), if there exist k,l in common(i,j)
                # with edge (k,l), then K4 exists.
                # Even simpler: just check all 4-subsets? Too slow.
                # Use: for each triple (i,j,k) forming a triangle, check if there's
                # a 4th vertex adjacent to all three.
                pass
        return True

    # Better K4-free check: iterate over all 4-tuples? 18C4 = 3060, fast enough
    def check_k4_free(adj_list):
        # adj_list[i] is a set of neighbors
        for a in range(n):
            for b in range(a + 1, n):
                if b not in adj_list[a]:
                    continue
                for c in range(b + 1, n):
                    if c not in adj_list[a] or c not in adj_list[b]:
                        continue
                    for d in range(c + 1, n):
                        if (d in adj_list[a] and d in adj_list[b]
                                and d in adj_list[c]):
                            return False
        return True

    # --- Helper: find largest triangle-free induced subgraph size ---
    # Triangle-free induced subgraph = independent set in the "triangle graph"
    # where two vertices are connected if they form an edge AND there exists a
    # third vertex adjacent to both (i.e., they are in a triangle).
    # Wait: a triangle-free induced subgraph means no three vertices in the subset
    # form a triangle in the original graph.
    # So we need: max |S| such that for all {u,v,w} subset S, the induced subgraph
    # on {u,v,w} has no triangle.
    # This is equivalent to: S is an independent set in the "3-uniform hypergraph"
    # of triangles? No, it's a constraint satisfaction.
    
    # For n=18, we can use branch and bound to find the largest triangle-free
    # induced subgraph.
    # Alternative: a set S is triangle-free induced iff for every vertex v in S,
    # the neighbors of v within S form an independent set in G[S].
    # This is hard. Let's use a simpler approach:
    # A subset S is triangle-free if it contains no triangle from G.
    # So we want the largest subset that avoids all triangles of G.
    # This is the "maximum subset avoiding a set of 3-cliques" problem.
    
    def largest_triangle_free(adj_mask):
        """adj_mask[i] is a bitmask of neighbors of i.
        Return the size of the largest triangle-free induced subgraph."""
        # Precompute all triangles as bitmasks
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj_mask[b] & (1 << a)):
                    continue
                for c in range(b + 1, n):
                    if (adj_mask[c] & (1 << a)) and (adj_mask[c] & (1 << b)):
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        
        if not triangles:
            return n
        
        # Branch and bound: try to include vertices, maintaining that no triangle
        # is fully included.
        best = [0]
        
        def dfs(remaining, current_size):
            if current_size + remaining.bit_count() <= best[0]:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Choose a vertex to branch on
            # Pick vertex with highest degree in remaining for better pruning
            # For speed, just pick the lowest set bit
            lsb = remaining & (-remaining)
            v = lsb.bit_length() - 1
            
            # Branch 1: include v
            # If we include v, we must exclude any vertex that forms a triangle
            # with v and another included vertex. But we don't track included set
            # easily. Instead, when we include v, we remove all vertices u such
            # that there exists w already in the "forced excluded" set... 
            # This is getting complex. Let me use a different approach.
            pass
        
        # Simpler: use the fact that a triangle-free induced subgraph is an
        # independent set in the graph H where H has an edge (u,v) if u and v
        # are in some triangle together with a third vertex... no that's not right.
        
        # Actually: S is triangle-free iff S contains no triangle from G.
        # So S is an independent set in the 3-uniform hypergraph of triangles.
        # For n=18, we can use a simple branch and bound:
        # State: (set of candidate vertices, set of already chosen)
        # But 2^18 = 262144 is manageable with pruning.
        
        # Let me use a cleaner B&B:
        # candidates: bitmask of vertices that can still be added
        # chosen: bitmask of vertices already in S
        # Constraint: no triangle fully contained in chosen
        
        # Precompute for each vertex v, the set of pairs (u,w) such that {v,u,w}
        # is a triangle. When we add v, we must ensure that for no (u,w) with
        # u,w in chosen is {v,u,w} a triangle.
        
        # For speed, precompute tri_mask[v] = bitmask of vertices w such that
        # there exists u < v, w < v... this is getting complicated.
        
        # Alternative simple approach: just check all subsets? 2^18 is too many.
        # But with B&B it's fine.
        
        best_size = [0]
        
        # Precompute: for each pair (i,j), the set of k such that {i,j,k} is a triangle
        pair_tri = [[set() for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                if adj_mask[b] & (1 << a):
                    for c in range(b + 1, n):
                        if (adj_mask[c] & (1 << a)) and (adj_mask[c] & (1 << b)):
                            pair_tri[a][b].add(c)
                            pair_tri[b][a].add(c)
        
        def dfs(cands, chosen_mask, chosen_set):
            if chosen_mask.bit_count() + cands.bit_count() <= best_size[0]:
                return
            if not cands:
                if chosen_mask.bit_count() > best_size[0]:
                    best_size[0] = chosen_mask.bit_count()
                return
            
            lsb = cands & (-cands)
            v = lsb.bit_length() - 1
            
            # Branch 1: include v
            # Check if v can be added: for every u in chosen_set, 
            # pair_tri[u][v] must not contain any w in chosen_set
            can_add = True
            for u in chosen_set:
                if u in pair_tri[v]:
                    # Check if any w in chosen_set is in pair_tri[u][v]
                    for w in chosen_set:
                        if w != u and w != v and w in pair_tri[u][v]:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                new_chosen_mask = chosen_mask | lsb
                new_chosen_set = set(chosen_set)
                new_chosen_set.add(v)
                # New candidates: cands without v
                new_cands = cands ^ lsb
                dfs(new_cands, new_chosen_mask, new_chosen_set)
            
            # Branch 2: exclude v
            dfs(cands ^ lsb, chosen_mask, chosen_set)
        
        dfs((1 << n) - 1, 0, set())
        return best_size[0]

    # --- Initial solution: K6,6,6 ---
    # Complete tripartite graph: edges between different parts
    def initial_graph():
        parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
        adj_mask = [0] * n
        adj_list = [set() for _ in range(n)]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
                        adj_list[a].add(b)
                        adj_list[b].add(a)
        return adj_mask, adj_list

    adj_mask, adj_list = initial_graph()
    initial_score = largest_triangle_free(adj_mask)
    best_score = initial_score
    best_adj_mask = list(adj_mask)
    
    # For K6,6,6: two parts (12 vertices) induce a complete bipartite graph K6,6
    # which is triangle-free. So score should be 12.
    # Let's verify: in K6,6,6, take all vertices from part 0 and part 1.
    # The induced subgraph is K6,6, which is triangle-free. Size 12.
    # Can we get 13? If we add one vertex from part 2, it's connected to all
    # in part 0 and part 1, so with any edge in part 0-part 1, we get a triangle.
    # So 12 is correct for K6,6,6.
    
    # --- Local search: flip edges to reduce score ---
    random.seed(42)
    
    def score_graph(adj_m):
        """Return (score, is_k4_free) for a graph given as adj masks."""
        # Check K4-free
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj_m[b] & (1 << a)):
                    continue
                for c in range(b + 1, n):
                    if not (adj_m[c] & (1 << a)) or not (adj_m[c] & (1 << b)):
                        continue
                    for d in range(c + 1, n):
                        if (adj_m[d] & (1 << a)) and (adj_m[d] & (1 << b)) and (adj_m[d] & (1 << c)):
                            return (0, False)
        # Compute triangle-free induced subgraph size
        return (largest_triangle_free(adj_m), True)

    def try_flip(adj_m, u, v):
        """Try flipping edge (u,v). Return (new_adj_m, new_score) or None if K4 created."""
        new_adj = list(adj_m)
        bit_u = 1 << u
        bit_v = 1 << v
        
        if adj_m[u] & bit_v:
            # Remove edge
            new_adj[u] = adj_m[u] ^ bit_v
            new_adj[v] = adj_m[v] ^ bit_u
        else:
            # Add edge
            new_adj[u] = adj_m[u] | bit_v
            new_adj[v] = adj_m[v] | bit_u
        
        # Check K4-free
        for a in range(n):
            for b in range(a + 1, n):
                if not (new_adj[b] & (1 << a)):
                    continue
                for c in range(b + 1, n):
                    if not (new_adj[c] & (1 << a)) or not (new_adj[c] & (1 << b)):
                        continue
                    for d in range(c + 1, n):
                        if (new_adj[d] & (1 << a)) and (new_adj[d] & (1 << b)) and (new_adj[d] & (1 << c)):
                            return None
        
        sc, _ = score_graph(new_adj)
        return (new_adj, sc)

    # Local search
    current_adj = list(adj_mask)
    current_score = initial_score
    best_adj = list(adj_mask)
    best_score = initial_score
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all possible edge flips
        for u in range(n):
            if time.time() > deadline - 5:
                break
            for v in range(u + 1, n):
                if time.time() > deadline - 5:
                    break
                result = try_flip(current_adj, u, v)
                if result is None:
                    continue
                new_adj, new_score = result
                if new_score < current_score:
                    current_adj = new_adj
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = list(current_adj)
                        break
            if time.time() > deadline - 5:
                break
    
    # If we found a better score, use it
    if best_score < initial_score:
        adj_mask = best_adj
    
    # Build edge list from best_adj_mask
    adj_final = best_adj if best_score < initial_score else adj_mask
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_final[u] & (1 << v):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
