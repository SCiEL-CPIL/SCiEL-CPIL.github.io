import random
import time
import numpy as np
from itertools import combinations

DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    n = 40
    random.seed(42)
    
    # Helper: check if adding edge (u,v) creates a C6
    # A C6 through edge (u,v) exists if there's a path of length 4 from u to v
    # (i.e., u-a-b-c-d-v where all edges exist and all distinct)
    
    def has_path_len_4(adj, u, v):
        """Check if there's a path of exactly 4 edges from u to v (5 vertices)."""
        # BFS/DFS for path of length 4 from u to v
        # Path: u - x - y - z - v (4 edges, 5 vertices, all distinct)
        neighbors_u = [w for w in range(n) if adj[u][w] and w != v]
        for x in neighbors_u:
            neighbors_x = [w for w in range(n) if adj[x][w] and w != u and w != v]
            for y in neighbors_x:
                neighbors_y = [w for w in range(n) if adj[y][w] and w != u and w != x and w != v]
                for z in neighbors_y:
                    if adj[z][v]:
                        return True
        return False
    
    def count_c6s(adj):
        """Count number of C6s in the graph."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Check if there's a path of length 4 from i to j
                    # That would form a C6 with edge (i,j)
                    # But we need to count each C6 once
                    pass
        # More efficient: count paths of length 5 between non-adjacent pairs
        # Actually, let's just count by checking all 6-tuples - too slow
        # Use: a C6 is determined by a path of length 5 whose endpoints are adjacent
        # Count = sum over edges (u,v) of (number of paths of length 4 from u to v)
        # But each C6 has 6 edges, so we'd count each 6 times
        return count
    
    def find_and_remove_c6s(adj):
        """Find C6s and remove edges to break them. Returns number of edges removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            # Find all C6s by looking for paths of length 5 between adjacent vertices
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v]:
                        continue
                    # Check for path of length 4 from u to v
                    # u-a-b-c-d-v
                    neighbors_u = [w for w in range(n) if adj[u][w] and w != v]
                    for a in neighbors_u:
                        neighbors_a = [w for w in range(n) if adj[a][w] and w != u and w != v]
                        for b in neighbors_a:
                            neighbors_b = [w for w in range(n) if adj[b][w] and w != u and w != a and w != v]
                            for c in neighbors_b:
                                neighbors_c = [w for w in range(n) if adj[c][w] and w != u and w != a and w != b and w != v]
                                for d in neighbors_c:
                                    if adj[d][v]:
                                        # Found C6: u-a-b-c-d-v
                                        # Remove edge (c,d) or some edge to break it
                                        # Remove a random edge from the cycle
                                        cycle_edges = [(u,a), (a,b), (b,c), (c,d), (d,v), (v,u)]
                                        # Pick the middle edge to remove
                                        e = cycle_edges[3]  # (c,d)
                                        if adj[e[0]][e[1]]:
                                            adj[e[0]][e[1]] = 0
                                            adj[e[1]][e[0]] = 0
                                            removed += 1
                                            changed = True
                                        break
                                if changed:
                                    break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return removed
    
    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Phase 1: Initialize with a random graph at density ~0.4
    # 40*39/2 = 780 possible edges. Density 0.4 gives ~312 edges, too many.
    # We need to end up with ~105 edges. Let's start lower.
    # Actually, let's start with a structured graph.
    
    # Try a different approach: start with a Turan-like or random graph
    # and iteratively remove C6s.
    
    # Start with random graph at density that gives ~120 edges
    target_initial = 120
    density = target_initial / (n * (n-1) / 2)
    
    best_adj = None
    best_edges = -1
    
    # Multiple random restarts
    num_restarts = 5
    edges_per_restart = 100  # approximate
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Initialize random graph
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Phase 2: Remove C6s
        find_and_remove_c6s(adj)
        
        # Phase 3: Try to add edges
        for attempt in range(50):
            if time.time() - start_time > DEADLINE:
                break
            # Try adding a random non-edge
            added = False
            for _ in range(200):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or adj[u][v]:
                    continue
                # Check if adding (u,v) creates a C6
                if not has_path_len_4(adj, u, v):
                    adj[u][v] = 1
                    adj[v][u] = 1
                    added = True
                    break
            if not added:
                break
        
        # Phase 4: Try edge swaps to improve
        for attempt in range(30):
            if time.time() - start_time > DEADLINE:
                break
            swapped = False
            for _ in range(100):
                # Pick a random edge to remove
                u1 = random.randint(0, n-1)
                v1 = random.randint(0, n-1)
                if u1 == v1 or not adj[u1][v1]:
                    continue
                # Pick a random non-edge to add
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2 or adj[u2][v2]:
                    continue
                # Check if removing (u1,v1) and adding (u2,v2) keeps graph C6-free
                # Temporarily remove (u1,v1)
                adj[u1][v1] = 0
                adj[v1][u1] = 0
                # Check if adding (u2,v2) creates a C6
                if not has_path_len_4(adj, u2, v2):
                    adj[u2][v2] = 1
                    adj[v2][u2] = 1
                    swapped = True
                    break
                else:
                    # Restore
                    adj[u1][v1] = 1
                    adj[v1][u1] = 1
            if not swapped:
                break
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj.copy()
    
    # If we didn't find anything good, use the K5 construction as fallback
    if best_adj is None or best_edges < 80:
        adj = np.zeros((n, n), dtype=np.int8)
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    adj[5*k+a][5*k+b] = 1
                    adj[5*k+b][5*k+a] = 1
        best_adj = adj
        best_edges = 80
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
