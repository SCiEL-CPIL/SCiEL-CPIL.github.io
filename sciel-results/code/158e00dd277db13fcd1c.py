import time
import random
from itertools import combinations

def construct():
    n = 18
    # Initial graph: Paley(17) + isolated vertex
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    edges = []
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                edges.append([u, v])
    
    # Helper to build adjacency matrix from edge list
    def build_adj(edge_list):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_list:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    # Helper to check K4-free
    def is_k4_free(adj):
        # Check all 4-tuples
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if not adj[u][w] or not adj[v][w]: continue
                    # u, v, w form a triangle. Check if any x connects to all 3
                    for x in range(w+1, n):
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            return False
        return True

    # Helper to find max triangle-free induced subgraph size
    # Uses branch and bound with bitmasks
    def max_triangle_free_size(adj):
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
        
        if not triangles:
            return n

        # Sort triangles?
        
        best_size = 0
        # Use a recursive search
        # State: current_mask, available_mask, depth?
        # Simpler: Iterate subsets? No.
        
        # Branch and bound:
        # Candidate set C.
        # If C is empty, return 0.
        # Upper bound: |C|.
        # If |C| <= best_size, return 0.
        # Pick a vertex v in C.
        # Option 1: Include v.
        #   New C = C \ {v} \ {neighbors of v that form triangles with v?}
        #   Wait, the constraint is "no triangles".
        #   If we include v, we cannot include any pair (u,w) such that (u,v,w) is a triangle.
        #   This is not a simple vertex removal.
        
        # Alternative: Max Independent Set in the "Triangle Conflict Graph"?
        # No, that's not right.
        
        # Let's use a simple backtracking on vertices.
        # Order vertices by degree?
        
        # Given n=18, we can just check all subsets of size > best_size?
        # If best_size is 9, we check subsets of size 10, 11, ...
        # Number of subsets of size 10 in 18 is 43758.
        # For each, check if it contains a triangle.
        # This is feasible.
        
        # Optimization:
        # If we find a subset of size k that is triangle-free, we can update best_size.
        # We want the MAX size.
        
        # Let's try to find if there is a triangle-free set of size k.
        # We can iterate k from n down to best_size + 1.
        # First k for which a triangle-free set exists is the answer.
        
        # To check if a specific set S is triangle-free:
        # Check if any triangle from `triangles` is a subset of S.
        
        def is_triangle_free_mask(mask):
            for u, v, w in triangles:
                if (mask & (1 << u)) and (mask & (1 << v)) and (mask & (1 << w)):
                    return False
            return True

        # Iterate k from n down to 1
        # We start from the current best_size (from previous call or 0)
        # But for simplicity in local search, we might just compute from scratch or use a lower bound.
        
        # Efficient check:
        # If we have a set of vertices S, we can prune.
        
        # Let's just do a branch and bound that maximizes size.
        
        def dfs(mask, current_size):
            nonlocal best_size
            if current_size + bin(mask).count('0') <= best_size: # Upper bound check? No, mask is remaining candidates
                # Actually, mask represents available vertices.
                # Max possible size = current_size + popcount(mask)
                if current_size + bin(mask).count('0') <= best_size:
                    return
            
            if mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return

            # Pick a vertex v from mask
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just lowest index.
            v = (mask & -mask).bit_length() - 1
            bit_v = 1 << v
            
            # Option 1: Exclude v
            dfs(mask & ~bit_v, current_size)
            
            # Option 2: Include v
            # If we include v, we must exclude any (u, w) such that (u, v, w) is a triangle.
            # This is tricky because excluding a pair is not the same as excluding vertices.
            # However, if we include v, we cannot include both u and w for any triangle (u,v,w).
            # This doesn't force us to exclude u or w individually.
            
            # So simple vertex exclusion branch and bound doesn't work directly for "no triangles".
            # It works for "independent set" (no edges).
            
            # For triangle-free, we can use the fact that a triangle-free graph is a subgraph of a bipartite graph? No.
            
            # Let's stick to the subset enumeration for n=18 if the number of triangles is small.
            # Or use ILP.
            
            # Given the constraints, let's use a heuristic local search that doesn't require exact evaluation at every step, 
            # but evaluates exactly at the end or periodically.
            # However, the task requires minimizing the EXACT value.
            # So we must know the exact value to compare moves.
            
            # Let's use `scipy.optimize.milp` for the exact value.
            pass

        # Since implementing a fast exact solver for max triangle-free induced subgraph is complex,
        # and n=18 is small, let's use the subset enumeration approach with pruning.
        
        # We want to find the max k such that there exists a subset of size k with no triangles.
        # We can iterate k from n down to 1.
        # For a fixed k, we check if any subset of size k is triangle-free.
        # To speed this up, we can use a backtracking search for "does there exist a triangle-free set of size k?".
        
        def exists_triangle_free_set_of_size_k(k, triangles):
            # Backtracking
            # Vertices: 0..17
            # We pick vertices one by one.
            # If adding a vertex creates a triangle with two previously picked vertices, we fail.
            
            # Sort vertices?
            # To prune, we can order vertices.
            
            # State: picked list, next vertex index
            # If len(picked) + (n - next_idx) < k, return False
            
            # This might be slow if k is large and triangles are dense.
            
            # Let's try a different approach:
            # The complement of a triangle-free graph is...
            
            # Let's just use the ILP solver from scipy.
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            # Variables x_v in {0, 1}
            # Maximize sum(x_v)
            # Subject to: x_u + x_v + x_w <= 2 for all triangles (u, v, w)
            
            c = np.ones(n) # Maximize sum x -> Minimize -sum x
            A = np.zeros((len(triangles), n))
            for i, (u, v, w) in enumerate(triangles):
                A[i, u] = 1
                A[i, v] = 1
                A[i, w] = 1
            
            if len(triangles) == 0:
                return n
                
            constraints = [LinearConstraint(A, 0, 2)]
            integrality = np.ones(n)
            bounds = Bounds(0, 1)
            
            res = milp(c=-c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                return int(round(-res.fun))
            else:
                return 0 # Fallback

        # Local Search
        adj = build_adj(edges)
        current_best_size = max_triangle_free_size(adj)
        best_edges = edges[:]
        best_score = current_best_size
        
        start_time = time.time()
        deadline = start_time + 110 # 10s margin
        
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            # Pick a random edge to flip
            # We can also try to add/remove random non-edges
            if random.random() < 0.5:
                # Flip an existing edge
                if not edges: break
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                new_edges = edges[:]
                new_edges.pop(idx)
            else:
                # Add a random non-edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or adj[u][v]:
                    continue
                new_edges = edges + [[u, v]]
            
            new_adj = build_adj(new_edges)
            
            if not is_k4_free(new_adj):
                continue
                
            new_size = max_triangle_free_size(new_adj)
            
            if new_size < best_score:
                best_score = new_size
                best_edges = new_edges[:]
                edges = new_edges
                adj = new_adj
                # Reset iterations or continue
            elif new_size == best_score:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    edges = new_edges
                    adj = new_adj
                    
        return best_edges

    return edges
