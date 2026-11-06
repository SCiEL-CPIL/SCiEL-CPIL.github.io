import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # 145s safety margin for 150s limit
    best_n = 0
    best_edges = []
    
    # Helper: Check if graph is C5-free
    def is_c5_free(n, edges):
        if n > 35:
            return False
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            adj[u][v] = True
            adj[v][u] = True
        
        # Check for any 5-cycle
        # A 5-cycle u-v-w-x-y-u implies path u-v-w-x-y of length 4 and edge y-u.
        # We can check for each edge (u,v), if there is a path of length 3 from u to v in G-(u,v).
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # Temporarily consider u-v edge removed
                # We need a path u -> a -> b -> c -> v
                # a is neighbor of u (a != v)
                # c is neighbor of v (c != u)
                # b is neighbor of a and c, b != u, v, a, c
                found = False
                neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
                neighbors_v = [y for y in range(n) if adj[v][y] and y != u]
                
                for a in neighbors_u:
                    for c in neighbors_v:
                        if a == c:
                            continue
                        # Check if a and c have a common neighbor b
                        # b != u, v, a, c
                        for b in range(n):
                            if b in (u, v, a, c):
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

    # Helper: Check if graph has an independent set of size k
    def has_independent_set_of_size(n, edges, k=9):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Greedy approach to find a large independent set
        def greedy_is(vertices):
            rem = list(vertices)
            is_set = []
            while rem:
                # Pick vertex with min degree in induced subgraph of rem
                best_v = None
                min_deg = float('inf')
                random.shuffle(rem)
                for v in rem:
                    deg = len([u for u in rem if u in adj[v]])
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                is_set.append(best_v)
                rem.remove(best_v)
                # Remove neighbors
                new_rem = []
                for u in rem:
                    if u not in adj[best_v]:
                        new_rem.append(u)
                rem = new_rem
            return len(is_set)

        # Try multiple greedy passes
        all_vertices = list(range(n))
        for _ in range(50):
            size = greedy_is(all_vertices)
            if size >= k:
                return True
        return False

    def validate(n, edges):
        # Check simple graph
        if len(edges) != len(set((min(u,v), max(u,v)) for u,v in edges)):
            return False
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
        
        if not is_c5_free(n, edges):
            return False
            
        if has_independent_set_of_size(n, edges, 9):
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

    # 2. Search for n=33, 34, 35
    # We use a local search approach.
    # Start with a random graph and try to remove C5s and reduce independence number.
    
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 10:
            break
        if target_n <= best_n:
            continue
            
        found = False
        attempts = 0
        while time.time() < deadline - 10 and not found and attempts < 100:
            attempts += 1
            # Generate random graph
            p = 0.5
            edges = []
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < p:
                        edges.append([u, v])
            
            # Local search: try to remove edges that are part of C5s
            # and add edges to reduce independence number?
            # Actually, adding edges reduces independence number but might create C5s.
            # Removing edges increases independence number but removes C5s.
            # We need a balance.
            
            # Let's try to repair a random graph by removing edges that form C5s.
            # This might make it too sparse and increase alpha.
            
            # Alternative: Start with a dense graph and remove C5s.
            # Then check alpha. If alpha >= 9, add edges to reduce alpha (carefully to avoid C5s).
            
            # For now, let's just check random graphs.
            if is_c5_free(target_n, edges):
                if not has_independent_set_of_size(target_n, edges, 9):
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            # If not found, try a modified version:
            # Start with 8 K4s + 1 vertex, and add random edges between K4s.
            if target_n == 33:
                edges_33 = []
                for block in range(8):
                    base = block * 4
                    for i in range(4):
                        for j in range(i+1, 4):
                            edges_33.append([base+i, base+j])
                # Vertex 32
                # Connect 32 to one vertex in each K4
                for block in range(8):
                    base = block * 4
                    edges_33.append([32, base + random.randint(0, 3)])
                
                # Add some random edges between K4s to reduce alpha
                for _ in range(10):
                    u = random.randint(0, 31)
                    v = random.randint(0, 31)
                    if u != v and u // 4 != v // 4:
                        edges_33.append([u, v])
                
                if is_c5_free(33, edges_33):
                    if not has_independent_set_of_size(33, edges_33, 9):
                        best_n = 33
                        best_edges = edges_33
                        found = True
                        break

    # Fallback to seed if nothing better
    if best_n == 0:
        best_n = 28
        best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
