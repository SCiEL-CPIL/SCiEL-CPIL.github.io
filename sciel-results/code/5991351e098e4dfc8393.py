import time
import random
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    results = []
    deadline = time.time() + 140.0  # safety margin before 150s kill

    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        # Build adjacency list for G
        adj_g = set()
        adj_g_set = set()
        for u, v in edges:
            adj_g_set.add((u, v))
            adj_g_set.add((v, u))
            
        # Map edges to indices
        edge_list = edges
        edge_index = {e: i for i, e in enumerate(edge_list)}
        
        # Build H = L(G)^2 conflict graph
        # H[i] is the set of indices of edges that conflict with edge_list[i]
        h_adj = [set() for _ in range(m)]
        
        for i, e1 in enumerate(edge_list):
            u1, v1 = e1
            for j, e2 in enumerate(edge_list):
                if i == j:
                    continue
                u2, v2 = e2
                # Check share vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
                    continue
                # Check connected by edge in G
                # Pairs to check: (u1,u2), (u1,v2), (v1,u2), (v1,v2)
                if (u1, u2) in adj_g_set or (u1, v2) in adj_g_set or (v1, u2) in adj_g_set or (v1, v2) in adj_g_set:
                    h_adj[i].add(j)
                    h_adj[j].add(i)

        # We want to find a coloring of H with minimum colors.
        # This is equivalent to partitioning vertices of H into independent sets.
        # An independent set in H is an induced matching in G.
        
        # Strategy:
        # 1. Get an initial coloring using DSATUR.
        # 2. Try to merge colors (reduce count) using local search.
        
        def dsatur():
            color = [-1] * m
            sat = [set() for _ in range(m)]
            uncolored = set(range(m))
            deg_h = [len(h_adj[i]) for i in range(m)]
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    s = len(sat[v])
                    d = deg_h[v]
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_v = v
                        best_sat = s
                        best_deg = d
                        
                v = best_v
                uncolored.remove(v)
                
                c = 0
                while c in sat[v]:
                    c += 1
                color[v] = c
                
                for u in h_adj[v]:
                    if u in uncolored:
                        sat[u].add(c)
                        
            return color

        # Initial coloring
        col = dsatur()
        num_colors = max(col) + 1
        
        # Local search to merge colors
        # We represent the coloring as a list of sets of vertices for each color class.
        # We try to merge two color classes if their union is an independent set in H.
        # i.e., for all u in C_i and v in C_j, (u,v) is NOT in H.
        
        def is_valid_merge(c1, c2):
            # Check if union of color class c1 and c2 is independent in H
            # i.e., no edge in H between any u in c1 and v in c2
            for u in c1:
                for v in c2:
                    if v in h_adj[u]:
                        return False
            return True

        def merge_colors(col, c1, c2):
            # Merge c2 into c1
            for i in range(m):
                if col[i] == c2:
                    col[i] = c1
            # Remove c2 from usage? We just track active colors.
            # We'll keep the array but ignore unused colors.
            pass

        # To speed up, we can maintain the sets of vertices for each color.
        color_sets = [set() for _ in range(num_colors)]
        for i in range(m):
            color_sets[col[i]].add(i)
        
        # Iterative improvement: try to merge any two color classes.
        # Repeat until no more merges are possible or time runs out.
        
        # We need to be careful with the time limit.
        # m is small (<= 240).
        # Number of colors is small (<= 20-45).
        # Checking merge is O(|C1| * |C2| * avg_deg) or just O(|C1| * |C2|) if we check adjacency sets.
        
        # Let's use a simpler heuristic:
        # Repeatedly try to merge the two smallest color classes? Or random pairs?
        # Let's try random pairs to escape local minima.
        
        rng = random.Random(42)
        
        # We'll keep a list of active colors
        active_colors = list(range(num_colors))
        
        # To make merging faster, we can precompute conflicts between color classes?
        # No, just check on the fly.
        
        # Let's limit the number of attempts to avoid infinite loops or excessive time.
        max_iters = 10000
        iter_count = 0
        
        while len(active_colors) > 1 and iter_count < max_iters and time.time() < deadline - 5.0:
            # Pick two random active colors
            if len(active_colors) < 2:
                break
            c1, c2 = rng.sample(active_colors, 2)
            
            if is_valid_merge(color_sets[c1], color_sets[c2]):
                # Merge c2 into c1
                for v in color_sets[c2]:
                    col[v] = c1
                    color_sets[c1].add(v)
                color_sets[c2] = set()
                active_colors.remove(c2)
                iter_count = 0 # reset counter on success
            else:
                iter_count += 1
                # If stuck for too long, maybe restart with a different seed or order?
                # For now, just continue.
                if iter_count > 500:
                    # Try to swap vertices to enable a merge? 
                    # This is complex. Let's just break and try a different initial coloring if possible.
                    # Or just stop.
                    break
        
        # Recalculate min color index and compress
        min_c = min(col)
        max_c = max(col)
        # The actual number of colors used is the number of active colors.
        # But the output requires a list of colors.
        # We should probably re-label colors to 0..K-1 to be clean, though not strictly required if verifier just checks classes.
        # "colours follow the sorted edge order" -> list of ints.
        # "Minimize the total number of colours used" -> likely max(col)+1 or distinct count.
        # Let's compress to 0..K-1.
        
        unique_colors = sorted(set(col))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        final_col = [color_map[c] for c in col]
        
        results.append(final_col)

    return results
