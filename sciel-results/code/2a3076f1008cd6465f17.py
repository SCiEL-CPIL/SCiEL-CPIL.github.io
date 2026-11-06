import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # 145s safety margin for 150s limit
    best_n = 0
    best_edges = []
    
    # Helper: Check if graph is C5-free using efficient method
    def is_c5_free(n, adj):
        """Check if graph with adjacency matrix adj has no C5 subgraph.
        A C5 exists iff there's an edge (u,v) and a path of length 3 from u to v
        in G - edge(u,v)."""
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # Check if there's a path u-a-b-c-v with all distinct vertices
                # a ~ u, c ~ v, a ~ b, b ~ c, all distinct from u,v
                found = False
                # Get neighbors of u (excluding v) and neighbors of v (excluding u)
                nbrs_u = [x for x in range(n) if adj[u][x] and x != v]
                nbrs_v = [y for y in range(n) if adj[v][y] and y != u]
                
                for a in nbrs_u:
                    for c in nbrs_v:
                        if a == c:
                            continue
                        # Check if a and c share a common neighbor b
                        # b must be distinct from u, v, a, c
                        for b in range(n):
                            if b == u or b == v or b == a or b == c:
                                continue
                            if adj[a][b] and adj[b][c]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    return False
        return True

    # Helper: Check if graph has independent set of size k using branch and bound
    def has_independent_set_of_size(n, adj, k=9):
        """Check if graph has an independent set of size k."""
        # Use recursive backtracking with pruning
        # State: current set of vertices that can still be added
        # We want to find if we can pick k non-adjacent vertices
        
        # Precompute: for each vertex, its non-neighbors (potential companions)
        non_nbrs = []
        for i in range(n):
            non_nbrs.append(set(range(n)) - set(adj[i]) - {i})
        
        def can_find_independent(candidates, need):
            if need == 0:
                return True
            if len(candidates) < need:
                return False
            if not candidates:
                return False
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with fewest connections within candidates
            best_v = None
            min_conflicts = float('inf')
            for v in candidates:
                conflicts = sum(1 for u in candidates if adj[v][u])
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_v = v
            
            # Option 1: include best_v
            new_candidates = {u for u in candidates if u != best_v and not adj[best_v][u]}
            if can_find_independent(new_candidates, need - 1):
                return True
            
            # Option 2: exclude best_v
            new_candidates2 = candidates - {best_v}
            if can_find_independent(new_candidates2, need):
                return True
            
            return False
        
        return can_find_independent(set(range(n)), k)

    def edges_to_adj(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return None
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def validate(n, edges):
        # Check simple graph
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
        
        adj = edges_to_adj(n, edges)
        if adj is None:
            return False
        
        if not is_c5_free(n, adj):
            return False
            
        if has_independent_set_of_size(n, adj, 9):
            return False
        return True

    # 1. Guaranteed Candidate: n=32 (8 disjoint K4)
    n_32 = 32
    edges_32 = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i+1, 4):
                edges_32.append([base+i, base+j])
    
    if validate(n_32, edges_32):
        best_n = n_32
        best_edges = edges_32

    # 2. Search for n=33, 34, 35 using structured constructions
    # We prioritize deterministic clique unions because they are known to be valid
    # and avoid the overhead of random search if they succeed.
    
    def construct_clique_union(n, sizes):
        edges = []
        current = 0
        for s in sizes:
            for i in range(s):
                for j in range(i+1, s):
                    edges.append([current+i, current+j])
            current += s
        return edges

    for target_n in [35, 34, 33]: # Try largest first
        if time.time() > deadline - 15:
            break
        if target_n <= best_n:
            continue
            
        found = False
        
        # Strategy B (now first): Use different block structure
        # Union of 8 cliques. Sum of sizes = target_n.
        # Alpha = 8 < 9. C5-free.
        
        # Find a partition of target_n into 8 positive integers
        # Simple greedy: 7 K4s and 1 K_{target_n - 28}
        # target_n - 28 must be >= 1.
        # 35 - 28 = 7. 34 - 28 = 6. 33 - 28 = 5.
        
        if target_n >= 29 and target_n <= 35:
            # 7 K4s + 1 K_{target_n - 28}
            sizes = [4, 4, 4, 4, 4, 4, 4, target_n - 28]
            edges_candidate = construct_clique_union(target_n, sizes)
            
            if validate(target_n, edges_candidate):
                best_n = target_n
                best_edges = edges_candidate
                found = True
        
        if found:
            continue
            
        # Strategy A: Start from 8xK4 and add vertices (Random Search)
        # Only run if deterministic construction failed (unlikely) or if we have time
        if time.time() > deadline - 10:
            continue
            
        base_edges = list(edges_32)
        
        for new_verts in range(target_n - 32 + 1):
            if new_verts < 1:
                continue
            if time.time() > deadline - 5:
                break
                
            new_vertices = list(range(32, 32 + new_verts))
            edges = list(base_edges)
            
            attempts = 0
            while time.time() < deadline - 5 and attempts < 50:
                attempts += 1
                edges = list(base_edges)
                
                for nv in new_vertices:
                    max_degree = random.randint(2, 6)
                    candidates = list(range(32))
                    random.shuffle(candidates)
                    chosen = candidates[:max_degree]
                    
                    for u in chosen:
                        edges.append([nv, u])
                
                for i in range(len(new_vertices)):
                    for j in range(i+1, len(new_vertices)):
                        if random.random() < 0.3:
                            edges.append([new_vertices[i], new_vertices[j]])
                
                if validate(target_n, edges):
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break

    # Fallback to seed if nothing better
    if best_n == 0:
        best_n = 28
        best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
