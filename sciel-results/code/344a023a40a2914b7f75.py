import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
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
        # C4-free iff for all u,v, |N(u) \cap N(v)| <= 1
        # Using matrix multiplication or bitsets
        # adj is bool matrix
        # common neighbors count is sum of (adj[u] & adj[v])
        # We can do this efficiently with numpy
        # For n=57, O(n^3) is 185k ops, very fast.
        for u in range(n):
            for v in range(u + 1, n):
                if np.sum(adj[u] & adj[v]) > 1:
                    return False
        return True

    def has_independent_set_exact(adj, k):
        """
        Returns True if there is an independent set of size k.
        Uses branch and bound on the complement graph (finding clique of size k).
        Optimized for small k and moderate n.
        """
        n = len(adj)
        comp = ~adj
        np.fill_diagonal(comp, False)
        
        # Precompute degrees in comp for initial ordering
        # We use a simple recursive search with bitmasks for candidates
        # To be faster, we use a list of vertices and a mask.
        
        # Tomita-style coloring bound is complex to implement correctly in Python quickly.
        # Simple bound: if len(candidates) + len(clique) < k, fail.
        # Heuristic: pick vertex with max degree in remaining candidates.
        
        # We'll use a recursive function that returns True/False.
        # To avoid recursion depth issues and overhead, we use a stack?
        # Recursion depth is k=11, so it's fine.
        
        # State: candidates (list of vertex indices), clique_size
        # We pass a list of candidates.
        
        def backtrack(candidates, clique_size):
            if clique_size >= k:
                return True
            if not candidates:
                return False
            if clique_size + len(candidates) < k:
                return False
            
            # Pick a vertex v from candidates
            # Heuristic: max degree in the subgraph induced by candidates
            # deg[v] = sum(comp[v][c] for c in candidates)
            
            # Optimization: If we just pick the first one, it might be slow.
            # Let's sort candidates by degree in the remaining graph?
            # This is O(M^2) where M is size of candidates.
            # M starts at 57, decreases.
            
            # Calculate degrees
            # comp is (n,n). candidates is list of indices.
            # We can use matrix multiplication: comp[candidates][:, candidates].sum(axis=1)
            # But slicing is expensive.
            
            # Simpler: just pick the vertex with highest index? Or lowest?
            # Random?
            # Let's try picking the vertex with the highest degree in the original comp graph?
            # That's a static heuristic.
            
            # Better dynamic heuristic:
            # For each v in candidates, count neighbors in candidates.
            # This is expensive in Python.
            
            # Alternative: Use a greedy coloring bound on the candidates.
            # Color the subgraph induced by candidates.
            # The number of colors is an upper bound on the clique size.
            # If clique_size + num_colors < k, fail.
            
            # Implementing greedy coloring in Python:
            # 1. Order candidates by degree (descending).
            # 2. Assign smallest available color.
            
            # Let's implement a simple version of this.
            
            # Get degrees in the induced subgraph
            # To avoid O(M^2) Python loops, we can use numpy if we keep a mask.
            # Let's represent candidates as a numpy bool array mask.
            
            # Revised backtrack using mask
            # This is cleaner.
            
            pass # Placeholder, we'll implement below

        # Let's implement the mask-based version directly for speed.
        # comp is numpy bool array.
        
        def check_clique(candidates_mask, clique_size):
            if clique_size >= k:
                return True
            if not np.any(candidates_mask):
                return False
            if clique_size + np.sum(candidates_mask) < k:
                return False
                
            # Greedy coloring bound
            # We need to color the subgraph induced by candidates_mask.
            # Colors are assigned to vertices.
            # A vertex v can have color c if no neighbor in candidates has color c.
            
            # This is a bit heavy. Let's try a simpler bound first:
            # Just branch on the first available vertex?
            # If that's too slow, we'll see.
            
            # Let's pick a vertex v.
            # To prune effectively, we should pick a vertex that constrains the search most.
            # High degree vertices are good.
            
            # Get list of candidates
            cand_list = np.where(candidates_mask)[0]
            
            # If we can't afford O(M^2) for degree calculation every step, 
            # we might be slow.
            
            # Let's try a different approach: 
            # Since we expect alpha to be small (e.g., 8), the search tree might be small.
            # Let's just pick the first candidate in the list (deterministic) or random?
            # Random might help avoid worst-case structures.
            
            # Let's pick the vertex with the highest degree in the original graph?
            # Precomputed degrees.
            # deg_orig = comp.sum(axis=1)
            
            # For now, let's just pick the first one from the sorted list (by index).
            # This is deterministic and fast.
            
            v = cand_list[0]
            
            # Option 1: Include v
            # New candidates: neighbors of v in comp, intersected with current candidates
            # Also, we can remove v from candidates.
            # To avoid symmetry, we can restrict to vertices > v? 
            # No, standard branch and bound doesn't need ordering if we just recurse on subsets.
            # But we must ensure we don't count the same clique multiple times?
            # The problem is decision (exists or not), so symmetry doesn't matter for correctness, 
            # but it affects performance if we explore equivalent branches.
            # However, since we just need to find ONE clique, we can be aggressive.
            
            new_mask = candidates_mask & comp[v]
            # Remove v from new_mask (it's not a neighbor of itself in simple graph, but just in case)
            new_mask[v] = False
            
            if check_clique(new_mask, clique_size + 1):
                return True
            
            # Option 2: Exclude v
            # Remove v from candidates
            candidates_mask[v] = False
            if check_clique(candidates_mask, clique_size):
                return True
            candidates_mask[v] = True # Restore
            
            return False
        
        initial_mask = np.ones(n, dtype=bool)
        return check_clique(initial_mask, 0)

    # Strategy: Use Polarity Graph of PG(2, 7)
    # This graph has 57 vertices.
    # It is C4-free.
    # We check if its independence number is < 11.
    # If so, any 43-vertex subgraph is valid.
    
    def construct_polarity_graph_q7():
        # PG(2, 7) has 57 points.
        # Points are non-zero vectors in GF(7)^3 up to scaling.
        # We can represent them as (x, y, z) with z=1 (affine plane, 49 points) + 7 points at infinity.
        # Total 56 + 1? No, 7^2 + 7 + 1 = 57.
        
        # Let's list all points.
        points = []
        # Affine part: x in 0..6, y in 0..6, z=1
        for x in range(7):
            for y in range(7):
                points.append((x, y, 1))
        # Points at infinity: x in 0..6, y=1, z=0 (slopes) + (1, 0, 0) (vertical)
        for m in range(7):
            points.append((m, 1, 0))
        points.append((1, 0, 0))
        
        n = len(points)
        # Map points to indices 0..56
        # Define polarity.
        # Standard polarity: (x, y, z) is orthogonal to (x', y', z') if x*x' + y*y' + z*z' = 0?
        # This is the Hermitian polarity? No, that's for unitary.
        # For projective planes, a polarity is a bijection between points and lines.
        # A simple polarity on PG(2, q) for q odd is given by the quadratic form x^2 + y^2 + z^2 = 0?
        # Wait, the graph defined by orthogonality in a non-degenerate quadratic form is C4-free.
        # Let's use the form Q(x,y,z) = x^2 + y^2 + z^2.
        # Two points p, q are adjacent if p^T A q = 0?
        # If we use the identity matrix, A=I.
        # p is adjacent to q if p.x*q.x + p.y*q.y + p.z*q.z == 0 mod 7.
        
        adj = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            x1, y1, z1 = points[i]
            for j in range(i + 1, n):
                x2, y2, z2 = points[j]
                # Check orthogonality
                dot = (x1 * x2 + y1 * y2 + z1 * z2) % 7
                if dot == 0:
                    adj[i][j] = True
                    adj[j][i] = True
        
        return adj

    # Generate the graph
    adj_57 = construct_polarity_graph_q7()
    
    # Verify C4-free (should be true by theory, but good to check)
    if is_c4_free(adj_57):
        # Check independence number
        # If alpha < 11, we can pick 43 vertices.
        # Let's check if there is an independent set of size 11.
        
        # To speed up the check, we can try to find a large independent set greedily first.
        # If greedy finds 11, we know alpha >= 11.
        # If greedy finds < 11, we need exact check.
        
        # Let's run the exact check.
        # If it returns False, alpha < 11.
        
        has_ind_11 = has_independent_set_exact(adj_57, 11)
        
        if not has_ind_11:
            # Alpha < 11.
            # We can take any 43 vertices.
            # To be safe and maximize n, we take n=43.
            # The induced subgraph on 43 vertices will also have alpha < 11.
            
            n_target = 43
            # Pick the first 43 vertices (indices 0 to 42)
            # Or random 43? First 43 is fine.
            
            adj_43 = adj_57[:n_target, :n_target].copy()
            
            # Re-verify C4-free on the subgraph (it is, as induced subgraph of C4-free graph)
            # Re-verify alpha < 11 on the subgraph (it is, as alpha is monotone)
            
            edges = get_edges_from_adj(adj_43)
            
            return {'n': 43, 'edges': edges}
        else:
            # Alpha >= 11.
            # We need to find a subgraph of size n < 43 with alpha < 11.
            # Or we fall back to the greedy construction for smaller n.
            # Let's try to remove vertices to reduce alpha.
            # This is hard.
            # Let's just use the fallback greedy method for n=42, 41...
            
            pass

    # Fallback: Greedy construction for n=42 down to 39
    # We reuse the greedy logic from the original program but with the faster exact check.
    
    for n in range(42, 38, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            # Greedy C4-free construction
            random.seed(seed)
            adj = np.zeros((n, n), dtype=bool)
            order = list(range(n))
            random.shuffle(order)
            
            for v in order:
                candidates = [u for u in range(n) if u != v]
                random.shuffle(candidates)
                
                for u in candidates:
                    if adj[v][u]:
                        continue
                    
                    # Check if adding (v,u) creates C4
                    # Condition: |N(v) \cap N(u)| < 2
                    common = np.sum(adj[v] & adj[u])
                    if common < 2:
                        # Add edge
                        adj[v][u] = True
                        adj[u][v] = True
                        # Check global C4-freeness? 
                        # The local check |N(v) \cap N(u)| < 2 is necessary but not sufficient 
                        # for global C4-freeness if we don't check other pairs?
                        # Actually, adding an edge (v,u) can only create C4s involving v or u.
                        # A C4 involving v and u as opposite vertices is prevented by common < 2.
                        # A C4 involving v and u as adjacent vertices?
                        # v-u-w-x-v. 
                        # This requires u-w, w-x, x-v.
                        # This C4 exists if there is a path of length 3 from u to v.
                        # The condition |N(v) \cap N(w)| <= 1 for all w ensures no C4.
                        # Adding (v,u) changes N(v) and N(u).
                        # We must ensure that for all w, |N(v) \cap N(w)| <= 1 and |N(u) \cap N(w)| <= 1.
                        # This is expensive.
                        # Let's just add it and check is_c4_free(adj) at the end?
                        # Or check locally?
                        # Let's just add it. We will verify at the end.
                        pass

            # Verify C4-free
            if not is_c4_free(adj):
                continue
                
            # Check Alpha
            # Fast greedy check
            verts = list(range(n))
            random.shuffle(verts)
            independent = []
            for v in verts:
                if all(not adj[v][u] for u in independent):
                    independent.append(v)
                    if len(independent) >= 11:
                        break
            if len(independent) >= 11:
                continue
                
            # Exact check
            if not has_independent_set_exact(adj, 11):
                edges = get_edges_from_adj(adj)
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                if n == 43:
                    return {'n': 43, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
