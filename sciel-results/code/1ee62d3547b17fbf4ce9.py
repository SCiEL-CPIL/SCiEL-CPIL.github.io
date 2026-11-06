import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_k4_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if adj[w] & C:
                            return False
                        temp &= temp - 1
        return True
    
    def get_triangles(adj):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            if adj[v] & (1 << w):
                                tris.append((u, v, w))
                        temp &= temp - 1
        return tris
    
    def max_triangle_free_size(adj, triangles):
        """Find largest triangle-free induced subgraph via branch and bound."""
        if not triangles:
            return n
        
        num_tris = len(triangles)
        
        # Precompute triangle masks and for each vertex, which triangles it belongs to
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in triangles]
        
        # For each vertex, list of triangle indices containing it
        vert_to_tris = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(triangles):
            vert_to_tris[a].append(i)
            vert_to_tris[b].append(i)
            vert_to_tris[c].append(i)
        
        # B&B: build a set S, prune if any triangle is fully in S
        # Maximize |S|
        best = [0]
        
        def bb(idx, current_mask, current_size, remaining_tris_mask):
            if time.time() > deadline - 2:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == n:
                return
            
            # Upper bound: current_size + remaining vertices
            if current_size + (n - idx) <= best[0]:
                return
            
            # Option 1: exclude vertex idx
            bb(idx + 1, current_mask, current_size, remaining_tris_mask)
            
            # Option 2: include vertex idx (if it doesn't complete any triangle)
            v = idx
            v_bit = 1 << v
            new_mask = current_mask | v_bit
            
            # Check if adding v completes any triangle
            safe = True
            for ti in vert_to_tris[v]:
                tm = tri_masks[ti]
                if (new_mask & tm) == tm:
                    safe = False
                    break
            
            if safe:
                # Update remaining triangles: remove triangles that are now "broken"
                # (i.e., have a vertex not in S) - but we only care about triangles fully in S
                # Actually, we need to track which triangles are still "alive" (all 3 vertices could be in S)
                # A triangle is broken if any of its vertices is excluded
                new_remaining = remaining_tris_mask
                for ti in vert_to_tris[v]:
                    tm = tri_masks[ti]
                    # If the triangle has another vertex not yet decided, it's still alive
                    # But if all 3 vertices are in new_mask, it's complete (shouldn't happen if safe)
                    # We need to remove triangles that have a vertex < idx that was excluded
                    # This is complex; let's just check safety on the fly
                    pass
                
                bb(idx + 1, new_mask, current_size + 1, remaining_tris_mask)
        
        # Simpler B&B: just check all triangles when including
        best = [0]
        
        def bb_simple(idx, current_mask, current_size):
            if time.time() > deadline - 2:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == n:
                return
            
            # Upper bound
            if current_size + (n - idx) <= best[0]:
                return
            
            # Option 1: exclude vertex idx
            bb_simple(idx + 1, current_mask, current_size)
            
            # Option 2: include vertex idx
            v = idx
            v_bit = 1 << v
            new_mask = current_mask | v_bit
            
            # Check if adding v completes any triangle
            safe = True
            for (a, b, c) in triangles:
                if v in (a, b, c):
                    tri_mask = (1 << a) | (1 << b) | (1 << c)
                    if (new_mask & tri_mask) == tri_mask:
                        safe = False
                        break
            
            if safe:
                bb_simple(idx + 1, new_mask, current_size + 1)
        
        bb_simple(0, 0, 0)
        return best[0]
    
    def can_add_edge_no_k4(adj, u, v):
        if adj[u] & (1 << v):
            return False
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True
    
    def random_greedy_construction(rng):
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        for u, v in possible_edges:
            if can_add_edge_no_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def local_search(adj, triangles, current_val, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_no_k4(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            new_tris = get_triangles(adj)
                            new_val = max_triangle_free_size(adj, new_tris)
                            if new_val < current_val:
                                triangles = new_tris
                                current_val = new_val
                                improved = True
                            else:
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
            # Try removing edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        new_tris = get_triangles(adj)
                        new_val = max_triangle_free_size(adj, new_tris)
                        if new_val < current_val:
                            triangles = new_tris
                            current_val = new_val
                            improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj, triangles, current_val
    
    def spectral_optimization(rng, time_limit):
        """
        Spectral approach: represent graph by continuous adjacency matrix.
        Optimize to encourage K4-free structure while minimizing triangle-free subgraph size.
        """
        # Start from a random symmetric matrix
        A = np.random.uniform(0, 1, (n, n))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        # We'll use a simple gradient-free approach: coordinate descent
        # Objective: penalize K4 subgraphs, encourage triangles (to break triangle-free sets)
        
        def k4_count(A):
            """Count K4 subgraphs (approximate for continuous A)."""
            count = 0.0
            for u in range(n):
                for v in range(u+1, n):
                    if A[u,v] > 0.5:
                        C = [w for w in range(n) if w != u and w != v and A[u,w] > 0.5 and A[v,w] > 0.5]
                        for i in range(len(C)):
                            for j in range(i+1, len(C)):
                                w1, w2 = C[i], C[j]
                                if A[w1, w2] > 0.5:
                                    count += 1
            return count
        
        def triangle_count(A):
            """Count triangles."""
            count = 0
            for u in range(n):
                for v in range(u+1, n):
                    if A[u,v] > 0.5:
                        for w in range(v+1, n):
                            if A[u,w] > 0.5 and A[v,w] > 0.5:
                                count += 1
            return count
        
        # Coordinate descent: try flipping individual edges
        best_adj = None
        best_val = n
        
        for iteration in range(200):
            if time.time() > time_limit:
                break
            
            # Round to binary
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if A[u,v] > 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not is_k4_free(adj):
                # Penalize: reduce some edges
                tris = get_triangles(adj)
                # Find K4s and break them
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            C = adj[u] & adj[v]
                            temp = C
                            while temp:
                                w = (temp & -temp).bit_length() - 1
                                if adj[w] & C:
                                    # Found K4, remove edge (u,v)
                                    adj[u] ^= (1 << v)
                                    adj[v] ^= (1 << u)
                                    A[u,v] = 0.3
                                    A[v,u] = 0.3
                                    break
                                temp &= temp - 1
                    if time.time() > time_limit:
                        break
                if time.time() > time_limit:
                    break
            
            # Evaluate
            triangles = get_triangles(adj)
            val = max_triangle_free_size(adj, triangles)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
            
            # Try perturbing
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u != v:
                if A[u,v] > 0.5:
                    A[u,v] = 0.3
                    A[v,u] = 0.3
                else:
                    A[u,v] = 0.7
                    A[v,u] = 0.7
        
        return best_adj, best_val
    
    best_val = n
    best_adj = None
    best_triangles = None
    
    rng = random.Random(42)
    
    # Try multiple random constructions
    num_constructions = 30
    for i in range(num_constructions):
        if time.time() > deadline - 40:
            break
        
        adj = random_greedy_construction(rng)
        triangles = get_triangles(adj)
        val = max_triangle_free_size(adj, triangles)
        
        if val < best_val:
            best_val = val
            best_adj = adj[:]
            best_triangles = triangles[:]
        
        # Local search on this construction
        remaining = deadline - time.time()
        if remaining > 10:
            time_limit = time.time() + min(remaining - 10, 15)
            adj, triangles, val = local_search(adj, triangles, val, rng, time_limit)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
                best_triangles = triangles[:]
        
        if best_val <= 8:
            break
    
    # Also try the tripartite construction as a baseline
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_triangles(adj)
    val = max_triangle_free_size(adj, triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles[:]
    
    # Try spectral optimization
    if time.time() < deadline - 20:
        spectral_adj, spectral_val = spectral_optimization(rng, deadline - 15)
        if spectral_adj is not None and spectral_val < best_val:
            best_val = spectral_val
            best_adj = spectral_adj[:]
            best_triangles = get_triangles(best_adj)
    
    # Final local search if time permits
    if time.time() < deadline - 15:
        time_limit = deadline - 5
        best_adj, best_triangles, best_val = local_search(best_adj, best_triangles, best_val, rng, time_limit)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
