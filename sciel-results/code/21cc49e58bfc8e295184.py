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
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 15:
            break
        if target_n <= best_n:
            continue
            
        found = False
        
        # Strategy A: Start from 8xK4 and add vertices
        # Try adding 1-3 vertices to the 32-vertex construction
        base_edges = list(edges_32)
        
        # Try different ways to add new vertices
        for new_verts in range(target_n - 32 + 1):
            if new_verts < 1:
                continue
            if time.time() > deadline - 10:
                break
                
            # Create new vertex set
            new_vertices = list(range(32, 32 + new_verts))
            edges = list(base_edges)
            
            # For each new vertex, decide which existing vertices to connect to
            # Must maintain C5-free and no independent set of size 9
            
            # Try random connections for new vertices
            attempts = 0
            while time.time() < deadline - 10 and attempts < 200:
                attempts += 1
                edges = list(base_edges)
                
                valid = True
                for nv in new_vertices:
                    # Choose random subset of existing vertices to connect to
                    # Keep degree low to avoid creating C5s
                    max_degree = random.randint(2, 6)
                    candidates = list(range(32))
                    random.shuffle(candidates)
                    chosen = candidates[:max_degree]
                    
                    for u in chosen:
                        edges.append([nv, u])
                
                # Also add edges between new vertices if needed
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
        
        if found:
            continue
        
        # Strategy B: Use different block structure
        # Try combinations of cliques that sum to target_n
        # C5-free graphs can be built from cliques
        # We need no independent set of size 9
        
        # Try: some K4s, some K3s, some K2s, some K1s
        # Independence number = number of blocks (one vertex from each block)
        # So we need at most 8 blocks
        
        # For target_n = 33: need 8 blocks summing to 33
        # Possible: 4+4+4+4+4+4+4+5 (but K5 might create issues)
        # Or: 4+4+4+4+4+4+3+6, etc.
        
        # Actually, let's try: 7 blocks of K4 (28 vertices) + 1 block of K5 (5 vertices) = 33
        # But K5 is fine for C5-free (it's a clique)
        # Independence number = 8 (one from each block) < 9, good!
        
        if target_n == 33:
            # 7 K4s + 1 K5
            edges_33 = []
            for block in range(7):
                base = block * 4
                for i in range(4):
                    for j in range(i+1, 4):
                        edges_33.append([base+i, base+j])
            base = 28
            for i in range(5):
                for j in range(i+1, 5):
                    edges_33.append([base+i, base+j])
            
            if validate(33, edges_33):
                best_n = 33
                best_edges = edges_33
                found = True
        
        # For target_n = 34: 7 K4s + 1 K6 = 34, or 8 blocks summing to 34
        if target_n == 34 and not found:
            # 7 K4s + 1 K6
            edges_34 = []
            for block in range(7):
                base = block * 4
                for i in range(4):
                    for j in range(i+1, 4):
                        edges_34.append([base+i, base+j])
            base = 28
            for i in range(6):
                for j in range(i+1, 6):
                    edges_34.append([base+i, base+j])
            
            if validate(34, edges_34):
                best_n = 34
                best_edges = edges_34
                found = True
        
        # For target_n = 35: 7 K4s + 1 K7 = 35, or other combinations
        if target_n == 35 and not found:
            # 7 K4s + 1 K7
            edges_35 = []
            for block in range(7):
                base = block * 4
                for i in range(4):
                    for j in range(i+1, 4):
                        edges_35.append([base+i, base+j])
            base = 28
            for i in range(7):
                for j in range(i+1, 7):
                    edges_35.append([base+i, base+j])
            
            if validate(35, edges_35):
                best_n = 35
                best_edges = edges_35
                found = True

    # Fallback to seed if nothing better
    if best_n == 0:
        best_n = 28
        best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
