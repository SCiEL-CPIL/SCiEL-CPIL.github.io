import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def get_edges_from_adj(adj):
        n = len(adj)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def is_c4_free(adj):
        n = len(adj)
        # Convert to bitmasks for speed
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
        
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = masks[u] & masks[v]
                # Check if popcount > 1
                if common & (common - 1) != 0:
                    return False
        return True

    def has_independent_set_exact(adj, k):
        """
        Returns True if there is an independent set of size k.
        Uses a highly optimized branch and bound on the complement graph (finding clique of size k).
        Uses Python integers as bitsets for speed.
        """
        n = len(adj)
        
        # Build complement graph adjacency as bitmasks
        # comp_mask[i] is an integer where bit j is 1 if i and j are NOT adjacent in adj
        # and i != j
        comp_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and not adj[i][j]:
                    mask |= (1 << j)
            comp_masks[i] = mask

        # Precompute degrees in complement for ordering
        # deg[i] = popcount(comp_masks[i])
        degs = [bin(m).count('1') for m in comp_masks]
        
        # We want to find a clique of size k in the complement graph.
        # Vertices are 0..n-1.
        
        # Greedy coloring bound for the induced subgraph on candidates_mask
        def greedy_color_bound(candidates_mask):
            """
            Returns an upper bound on the clique size in the subgraph induced by candidates_mask.
            Uses a simple greedy coloring heuristic.
            """
            if candidates_mask == 0:
                return 0
            
            # Get list of vertices
            verts = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            
            # Order by degree in the induced subgraph (descending)
            # To speed up, we can precompute degrees or just calculate on the fly.
            # For n=43, calculating on the fly is fast enough.
            verts.sort(key=lambda v: bin(comp_masks[v] & candidates_mask).count('1'), reverse=True)
            
            colors = {}
            next_color = 1
            for v in verts:
                # Find available colors
                used_colors = set()
                # Check neighbors of v in the candidate set
                neigh_mask = comp_masks[v] & candidates_mask
                temp_neigh = neigh_mask
                while temp_neigh:
                    lsb = temp_neigh & -temp_neigh
                    n_idx = lsb.bit_length() - 1
                    if n_idx in colors:
                        used_colors.add(colors[n_idx])
                    temp_neigh ^= lsb
                
                # Assign smallest available color
                c = 1
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > next_color:
                    next_color = c
            
            return next_color

        def backtrack(candidates_mask, clique_size):
            if clique_size >= k:
                return True
            if candidates_mask == 0:
                return False
            
            # Bound 1: Simple size bound
            if clique_size + bin(candidates_mask).count('1') < k:
                return False
            
            # Bound 2: Greedy coloring bound
            color_bound = greedy_color_bound(candidates_mask)
            if clique_size + color_bound < k:
                return False
            
            # Pick a vertex v from candidates
            # Heuristic: pick the vertex with highest degree in the candidate subgraph
            verts = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            
            # Sort by degree in candidate subgraph
            verts.sort(key=lambda v: bin(comp_masks[v] & candidates_mask).count('1'), reverse=True)
            
            v = verts[0]
            v_bit = 1 << v
            
            # Branch 1: Include v
            # New candidates: neighbors of v in comp, intersected with current candidates
            new_mask = candidates_mask & comp_masks[v]
            new_mask &= ~v_bit # v is not its own neighbor
            
            if backtrack(new_mask, clique_size + 1):
                return True
            
            # Branch 2: Exclude v
            # Remove v from candidates
            candidates_mask_excl = candidates_mask & ~v_bit
            if backtrack(candidates_mask_excl, clique_size):
                return True
            
            return False

        initial_mask = (1 << n) - 1
        return backtrack(initial_mask, 0)

    def construct_polarity_graph(q):
        """
        Constructs the C4-free graph from the orthogonal polarity on PG(2, q).
        Number of vertices: q^2 + q + 1.
        Two points are adjacent if they are orthogonal w.r.t the quadratic form x^2 + y^2 + z^2 = 0.
        """
        # Points in PG(2, q) represented as (x, y, z) with z=1 (affine) + points at infinity
        points = []
        # Affine part: x, y in 0..q-1
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        # Points at infinity: (x, 1, 0) for x in 0..q-1, and (1, 0, 0)
        for m in range(q):
            points.append((m, 1, 0))
        points.append((1, 0, 0))
        
        n = len(points)
        adj = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            x1, y1, z1 = points[i]
            for j in range(i + 1, n):
                x2, y2, z2 = points[j]
                # Orthogonality: x1*x2 + y1*y2 + z1*z2 == 0 mod q
                dot = (x1 * x2 + y1 * y2 + z1 * z2) % q
                if dot == 0:
                    adj[i][j] = True
                    adj[j][i] = True
        
        return adj

    # Strategy:
    # 1. Construct the graph for q=17 (17^2+17+1 = 307 vertices).
    # 2. Verify it is C4-free.
    # 3. Take an induced subgraph of size 43.
    # 4. Verify the subgraph has no independent set of size 11 using the optimized exact search.
    # 5. If valid, return it.
    
    q = 17
    adj_full = construct_polarity_graph(q)
    
    # Verify C4-free on the full graph (or just the subgraph? Full is safer for theory, subgraph is faster)
    # Since n=307, checking C4-free on full graph might be slow in Python loops.
    # Let's check C4-free on the subgraph directly.
    
    n_target = 43
    # Select first 43 vertices. 
    # To be safe against any structural quirks, we can try a few different subsets if the first fails alpha check.
    # But polarity graphs are highly symmetric.
    
    adj_sub = adj_full[:n_target, :n_target].copy()
    
    # Check C4-free
    if not is_c4_free(adj_sub):
        # This should never happen for a polarity graph, but just in case.
        # Fall back to random search or smaller n.
        pass
    else:
        # Check independence number
        # We need to ensure alpha < 11.
        # The independence number of the full graph PG(2, q) with this polarity is known to be small (approx sqrt(q)).
        # For q=17, sqrt(17) ~ 4.1. So alpha is likely <= 5.
        # Let's run the exact check to be 100% sure.
        
        if not has_independent_set_exact(adj_sub, 11):
            edges = get_edges_from_adj(adj_sub)
            return {'n': 43, 'edges': edges}
    
    # Fallback: Try q=13 (183 vertices)
    q = 13
    adj_full = construct_polarity_graph(q)
    adj_sub = adj_full[:n_target, :n_target].copy()
    if is_c4_free(adj_sub):
        if not has_independent_set_exact(adj_sub, 11):
            edges = get_edges_from_adj(adj_sub)
            return {'n': 43, 'edges': edges}
            
    # Fallback: Try q=11 (133 vertices)
    q = 11
    adj_full = construct_polarity_graph(q)
    adj_sub = adj_full[:n_target, :n_target].copy()
    if is_c4_free(adj_sub):
        if not has_independent_set_exact(adj_sub, 11):
            edges = get_edges_from_adj(adj_sub)
            return {'n': 43, 'edges': edges}

    # If all else fails, return a known valid small graph (e.g., 30 vertices)
    # Construct 10 disjoint triangles
    n = 30
    adj = np.zeros((n, n), dtype=bool)
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a][b] = adj[b][a] = True
        adj[a][c] = adj[c][a] = True
        adj[b][c] = adj[c][b] = True
        
    edges = get_edges_from_adj(adj)
    return {'n': n, 'edges': edges}

# To ensure the function is callable as construct()
if __name__ == "__main__":
    # Optional: quick self-test
    res = construct()
    print(f"Constructed graph with n={res['n']}")
