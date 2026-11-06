import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Helper to check C4-free property
    # A graph is C4-free if and only if no two vertices have more than 1 common neighbor.
    # This is equivalent to (A @ A)[i, j] <= 1 for all i != j.
    def is_c4_free(adj):
        A = adj.astype(int)
        A2 = A @ A
        # Diagonal elements of A2 are degrees, which can be > 1.
        # We only care about off-diagonal elements.
        # If any off-diagonal element is >= 2, there is a C4.
        # Note: A[i,j] can be 1 (edge) or 0 (non-edge).
        # If A[i,j]=1 and (A^2)[i,j]>=1, it's a triangle.
        # If A[i,j]=0 and (A^2)[i,j]>=2, it's a C4.
        # Actually, if (A^2)[i,j] >= 2, there are at least 2 common neighbors.
        # This forms a C4 u-v-w-x-u where v,x are common neighbors of u,w.
        # Even if u-w is an edge, the subgraph contains a C4 (u-v-w-x-u is a cycle).
        # So the condition "no two vertices have > 1 common neighbor" is sufficient and necessary for C4-free.
        
        # Check off-diagonal
        # Create a mask of off-diagonal
        mask = ~np.eye(adj.shape[0], dtype=bool)
        if np.any(A2[mask] >= 2):
            return False
        return True

    def get_min_degree(adj):
        return int(np.min(adj.sum(axis=1))) if adj.size > 0 else 0

    def try_local_search(adj, n, target_min_deg, max_time=10):
        """
        Try to improve the graph by swapping edges to increase min degree or n.
        Here we focus on maintaining C4-free while trying to add edges to boost min degree.
        """
        start_ls = time.time()
        adj = adj.copy()
        
        # Identify vertices with degree < target_min_deg
        def get_deficient_vertices():
            degrees = adj.sum(axis=1)
            return [i for i in range(n) if degrees[i] < target_min_deg]
        
        deficient = get_deficient_vertices()
        
        if not deficient:
            return adj

        # Try to add edges to deficient vertices
        # We need to find an edge (u, v) such that u is deficient, v is not adjacent to u,
        # and adding (u, v) doesn't create a C4.
        # Condition for no C4 when adding (u, v):
        # 1. u and v must have at most 1 common neighbor. (Actually, if they have 0, it's safe. If 1, it's safe? 
        #    If they have 1 common neighbor w, adding u-v creates a triangle u-v-w-u. This is not a C4.
        #    So adding an edge is safe if common_neighbors(u, v) <= 1.
        #    Wait, if they have 2 common neighbors, it creates a C4.
        #    So the condition is: common_neighbors(u, v) <= 1.
        
        # We can iterate through deficient vertices and try to connect them to non-neighbors.
        # To be efficient, we can precompute common neighbor counts.
        # But n is small (<= 47).
        
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False
            deficient = get_deficient_vertices()
            if not deficient:
                break
            
            random.shuffle(deficient)
            for u in deficient:
                if (time.time() - start_ls) > max_time:
                    break
                # Find candidates v
                # v must not be adjacent to u
                # v must have common_neighbors(u, v) <= 1
                # We want to pick v that is "central" or has high degree to help others? 
                # Or just pick any valid v.
                
                # Get non-neighbors of u
                non_neighbors = np.where(adj[u] == 0)[0]
                non_neighbors = [v for v in non_neighbors if v != u]
                
                random.shuffle(non_neighbors)
                
                for v in non_neighbors:
                    # Check common neighbors
                    common = np.sum(adj[u] & adj[v])
                    if common <= 1:
                        # Safe to add
                        adj[u, v] = True
                        adj[v, u] = True
                        improved = True
                        # Re-evaluate deficient list immediately? 
                        # Or just break and let outer loop handle it.
                        # Breaking inner loop allows other deficient vertices to be processed.
                        break
            
            if not improved:
                # Stuck. Try edge swaps?
                # Remove an edge (a, b) and add (u, v) where u is deficient.
                # This is complex. For now, let's see if greedy addition works.
                # If greedy fails, we might need to start with a denser seed.
                pass

        return adj

    # Strategy:
    # 1. Try to construct a valid graph for n=47.
    # 2. If fails, try n=46.
    # 3. Use Polarity Graph of Projective Plane of order q as a seed.
    #    For q=7, n=57. Too big.
    #    For q=5, n=31. Min degree q+1=6.
    #    We need min degree n-39.
    #    For n=46, min degree 7.
    #    For n=47, min degree 8.
    
    # Let's try a different seed:
    # Random regular graph with high degree, then remove C4s.
    # Or, use the known extremal construction for C4-free graphs.
    # Reiman graph (q=2, n=7, deg=3)
    # Polarity graph q=3 (n=13, deg=4)
    # Polarity graph q=4 (n=21, deg=5)
    # Polarity graph q=5 (n=31, deg=6)
    # Polarity graph q=6 (n=43, deg=7) -> This is a C4-free graph on 43 vertices with min degree 7.
    # If we take this graph (n=43, min_deg=7), it satisfies min_deg >= 43-39 = 4.
    # But we want to maximize n.
    # Can we extend the q=6 Polarity graph to n=46 or 47?
    
    # Let's generate the Polarity graph for q=6.
    # Vertices are points of PG(2, q). n = q^2 + q + 1.
    # For q=6, n = 36 + 6 + 1 = 43.
    # Two points are adjacent if one lies on the polar of the other.
    # This graph is C4-free.
    # Min degree is q+1 = 7.
    
    # We have a valid graph for n=43. Score = 43/47? No, score is value/seed.
    # Seed value is likely 39 (trivial). Or maybe the seed program returns 39?
    # The prompt says "Maximize n". "value is known to be 46 or 47".
    # If I return 43, it's better than 39.
    # But can I get 46?
    
    # Let's try to start with the q=6 Polarity graph (n=43) and add 3 vertices.
    # Or start with a random graph on 46 vertices and prune.
    
    def generate_polarity_graph(q):
        """
        Generate the Polarity graph of the projective plane of order q.
        Vertices are points (x, y, z) with x^2 + y^2 + z^2 = 0? No, that's for specific fields.
        Standard construction:
        Vertices: (x, y) for x,y in GF(q), plus (1, 0), (0, 1), (1, 1)? 
        Actually, easier to use the standard adjacency definition.
        
        Points in PG(2, q) can be represented as:
        (x, y, 1) for x, y in GF(q)
        (x, 1, 0) for x in GF(q)
        (0, 1, 0)
        Total: q^2 + q + 1.
        
        Adjacency: P is adjacent to Q if P lies on the polar of Q.
        For the standard polarity (x, y, z) -> (x, y, -z) ? No.
        Let's use the graph defined by:
        Vertices are pairs (x, y) in GF(q)^2 plus two points at infinity?
        
        Alternative: Use the known explicit construction for the Reiman graph (q=2) and generalize.
        For general q, the Polarity graph has vertices (i, j) for 0<=i,j<q, plus (inf, k) for 0<=k<q, plus (inf, inf).
        Wait, that's q^2 + q + 1.
        
        Let's use the simpler "Erdos-Renyi" style construction if possible, or just stick to Randomized Greedy.
        The previous attempt at Randomized Greedy was flawed because it didn't ensure high enough degree.
        
        Let's try a "Dense Random Graph + Pruning" approach.
        1. Start with a random graph G(n, p) where p is high enough to ensure min degree >= target.
           For n=46, target_deg=7. p=0.2 gives expected degree 9.2.
        2. Iteratively find C4s and remove edges.
        3. If min degree drops below target, stop and try to repair.
        
        This is likely to fail because pruning reduces degree.
        
        Better: "Greedy Add with Degree Constraint".
        1. Start with empty graph.
        2. Sort edges by a heuristic: prefer edges that connect low-degree vertices.
        3. Add edge if it doesn't create a C4.
        4. This tends to build a regular-like graph.
        
        Let's implement this improved greedy.
        """
        pass

    # Let's implement the "Greedy Add with Degree Constraint" properly.
    
    def greedy_construct(n, target_deg, time_limit=10):
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # List of all possible edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # We will process edges in multiple passes, shuffling each time.
        # Or, dynamically select the next edge.
        
        # Dynamic selection:
        # At each step, find the pair (u, v) with the lowest sum of degrees (or u has low degree)
        # such that adding (u, v) is safe (no C4).
        
        # To make it faster, we can precompute "safe" edges? 
        # Safe condition: common_neighbors(u, v) <= 1.
        
        # Let's just iterate.
        # n=46, edges ~ 1000.
        # If we do 1000 steps, and each step checks O(n^2) for safety, it's 10^6 ops. Very fast.
        
        remaining_edges = set(edges)
        
        # We want to fill degrees up to target_deg.
        # We'll loop until no more edges can be added or time runs out.
        
        while remaining_edges and (time.time() - start_time) < deadline - 5:
            # Find the best edge to add
            best_edge = None
            best_score = -1
            best_u, best_v = -1, -1
            
            # To avoid O(E * E) check, we can sample or iterate.
            # Since E is small, iterate all remaining edges.
            # But checking safety is O(n^2) or O(n) with bitsets.
            
            # Let's use bitsets for neighbors.
            # neighbors[i] is a bitset of neighbors of i.
            
            # Actually, let's just iterate through a random subset of remaining edges to find a candidate.
            # Or, simply pick the edge (u, v) where u is the vertex with minimum degree, 
            # and v is a non-neighbor with min common neighbors.
            
            # Find vertex u with min degree
            u = int(np.argmin(degrees))
            if degrees[u] >= target_deg:
                # If min degree is met, we can stop or try to add more?
                # No, we just need min degree >= target.
                # If all vertices have degree >= target, we are done.
                break
            
            # Find candidates for v
            non_neighbors = [v for v in range(n) if v != u and not adj[u, v]]
            if not non_neighbors:
                # u is connected to everyone. Can't add more edges to u.
                # Mark u as "full" for this pass? 
                # Just break out and let other vertices be handled.
                # If all deficient vertices are full, we are stuck.
                break
            
            # Find v in non_neighbors that minimizes common neighbors
            min_common = n
            best_v = -1
            for v in non_neighbors:
                # Count common neighbors
                common = np.sum(adj[u] & adj[v])
                if common < min_common:
                    min_common = common
                    best_v = v
                    if min_common == 0:
                        break
            
            if best_v == -1:
                break
            
            # Check if adding (u, best_v) is safe
            # Safe if min_common <= 1.
            if min_common <= 1:
                adj[u, best_v] = True
                adj[best_v, u] = True
                degrees[u] += 1
                degrees[best_v] += 1
                remaining_edges.discard((u, best_v) if u < best_v else (best_v, u))
            else:
                # Cannot add edge to u with any non-neighbor safely.
                # This implies u is "stuck". 
                # We should try to add edges to other deficient vertices.
                # To do this, we need to skip u for a while.
                # Let's just break and restart the loop, but maybe shuffle the vertex order?
                # The argmin will pick the same u again.
                # We need a way to skip stuck vertices.
                
                # Simple hack: Temporarily increase degree of u? No.
                # Let's just pick a random deficient vertex instead of argmin.
                deficient = [i for i in range(n) if degrees[i] < target_deg]
                if not deficient:
                    break
                u = random.choice(deficient)
                # Try again for this u
                non_neighbors = [v for v in range(n) if v != u and not adj[u, v]]
                if not non_neighbors:
                    continue
                min_common = n
                best_v = -1
                for v in non_neighbors:
                    common = np.sum(adj[u] & adj[v])
                    if common < min_common:
                        min_common = common
                        best_v = v
                        if min_common == 0:
                            break
                if best_v != -1 and min_common <= 1:
                    adj[u, best_v] = True
                    adj[best_v, u] = True
                    degrees[u] += 1
                    degrees[best_v] += 1
                    remaining_edges.discard((u, best_v) if u < best_v else (best_v, u))
                else:
                    # Stuck on this u too. 
                    # If we are stuck on all deficient vertices, we fail.
                    # Let's check if any deficient vertex can be improved.
                    # If not, break.
                    # For efficiency, just break and verify.
                    break

        return adj

    # Try n=47
    n_target = 47
    target_deg = n_target - 39
    adj = greedy_construct(n_target, target_deg)
    
    if get_min_degree(adj) >= target_deg and is_c4_free(adj):
        edges = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n_target, 'edges': edges}
    
    # Try n=46
    n_target = 46
    target_deg = n_target - 39
    adj = greedy_construct(n_target, target_deg)
    
    if get_min_degree(adj) >= target_deg and is_c4_free(adj):
        edges = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n_target, 'edges': edges}
    
    # Fallback: Return the Polarity graph for q=6 (n=43) which is guaranteed C4-free with min degree 7.
    # We need to construct it explicitly.
    # Vertices: (x, y) for x,y in 0..5, plus (inf, k) for k in 0..5, plus (inf, inf).
    # Indices:
    # 0-35: (x, y)
    # 36-41: (inf, k)
    # 42: (inf, inf)
    
    n = 43
    adj = np.zeros((n, n), dtype=bool)
    
    def get_idx(x, y):
        if x == 'inf' and y == 'inf':
            return 42
        if x == 'inf':
            return 36 + y
        if y == 'inf':
            return 36 + x # Wait, symmetry?
            # Let's define points as:
            # Type 1: (x, y) with x,y in GF(q). Index = x*q + y.
            # Type 2: (x, inf) with x in GF(q). Index = q*q + x.
            # Type 3: (inf, inf). Index = q*q + q.
            
    # Let's re-define indices clearly.
    # 0 to q^2-1: (x, y)
    # q^2 to q^2+q-1: (x, inf)
    # q^2+q: (inf, inf)
    
    q = 6
    n = q*q + q + 1
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Polarity: (x, y) is adjacent to (x', y') if x*x' + y*y' = 1? 
    # For the standard polarity graph (Reiman type), the adjacency is:
    # (x, y) ~ (x', y') iff x*x' + y*y' = 1
    # (x, y) ~ (x', inf) iff x*x' = 1
    # (x, y) ~ (inf, inf) iff x=0, y=0? No.
    # (x, inf) ~ (x', inf) iff x = x' ? No, they are not adjacent.
    # (x, inf) ~ (inf, inf) iff x = 0 ?
    
    # Let's use the property that it's a C4-free graph with min degree q+1.
    # We can just use the known structure or generate it via random greedy if explicit is hard.
    # But random greedy failed for 46/47.
    # Let's try to improve the random greedy by using a better heuristic.
    
    # Actually, the "Greedy Add with Degree Constraint" might work for 46 if we run it multiple times with different seeds.
    
    best_result = None
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(attempt)
        n_try = 46
        target = 7
        adj = greedy_construct(n_try, target)
        if get_min_degree(adj) >= target and is_c4_free(adj):
            edges = []
            for i in range(n_try):
                for j in range(i+1, n_try):
                    if adj[i, j]:
                        edges.append([i, j])
            best_result = {'n': n_try, 'edges': edges}
            break
    
    if best_result:
        return best_result
    
    # If all else fails, return a valid small graph.
    # n=39, empty graph. Min degree 0 >= 0.
    return {'n': 39, 'edges': []}
