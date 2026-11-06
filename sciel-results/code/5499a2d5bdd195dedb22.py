"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _has_c6(adj, n):
    """Check if graph with adjacency matrix has a C6 using BFS from each edge."""
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            # Try to find a path of length 4 from u to v not using edge (u,v) directly
            # i.e., u-a-b-c-d-v where none of a,b,c,d is u or v, and edges exist
            # Use BFS from neighbors of u (excluding v) to see if we can reach neighbors of v
            # in exactly 4 steps
            # Level 1: neighbors of u excluding v
            level1 = [w for w in range(n) if adj[u][w] and w != v]
            if not level1:
                continue
            # Level 2: neighbors of level1 excluding u and v
            level2 = set()
            for w in level1:
                for x in range(n):
                    if adj[w][x] and x != u and x != v:
                        level2.add(x)
            if not level2:
                continue
            # Level 3: neighbors of level2 excluding u, v, and level1
            level3 = set()
            for x in level2:
                for y in range(n):
                    if adj[x][y] and y != u and y != v and y not in level1:
                        level3.add(y)
            if not level3:
                continue
            # Level 4: check if any neighbor of level3 is a neighbor of v
            for y in level3:
                for z in range(n):
                    if adj[y][z] and adj[v][z] and z != u and z != v and z not in level1 and z not in level2:
                        return True
    return False

def _would_create_c6(adj, n, u, v):
    """Check if adding edge (u,v) creates a C6. A new C6 must use the new edge."""
    # New C6: u - a - b - c - d - v - u (using new edge u-v)
    # So we need a path of length 4 from u to v in the graph WITHOUT edge (u,v)
    # (which doesn't exist yet, so just in current graph)
    # But we need the path to not use u or v as internal vertices
    # BFS from u's neighbors (excluding v) to v's neighbors (excluding u) in 3 more steps
    
    # Neighbors of u excluding v
    nu = [w for w in range(n) if adj[u][w] and w != v]
    if not nu:
        return False
    
    # Neighbors of v excluding u
    nv = set(w for w in range(n) if adj[v][w] and w != u)
    if not nv:
        return False
    
    # BFS: from nu, can we reach nv in exactly 3 steps?
    # Step 1: from each node in nu, go to its neighbors (excluding u, v)
    level1 = set()
    for w in nu:
        for x in range(n):
            if adj[w][x] and x != u and x != v:
                level1.add(x)
    if not level1:
        return False
    
    # Step 2: from level1, go to neighbors (excluding u, v, nu)
    level2 = set()
    for x in level1:
        for y in range(n):
            if adj[x][y] and y != u and y != v and y not in nu:
                level2.add(y)
    if not level2:
        return False
    
    # Step 3: from level2, check if we can reach any node in nv
    for y in level2:
        for z in range(n):
            if adj[y][z] and z in nv:
                return True
    return False

def _greedy_build(seed, n=40, max_time=110):
    """Greedy edge addition with random restarts."""
    random.seed(seed)
    start = time.time()
    
    # Start with a random graph at moderate density
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    # Random initial edges
    num_initial = random.randint(50, 80)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    for u, v in all_pairs[:num_initial]:
        if not _would_create_c6(adj, n, u, v):
            adj[u][v] = adj[v][u] = True
            edges.append([u, v])
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    best_edge_list = edges[:]
    
    # Try to add more edges greedily
    remaining = [p for p in all_pairs[num_initial:] if not adj[p[0]][p[1]]]
    random.shuffle(remaining)
    
    for u, v in remaining:
        if time.time() - start > max_time:
            break
        if not adj[u][v] and not _would_create_c6(adj, n, u, v):
            adj[u][v] = adj[v][u] = True
            edges.append([u, v])
            best_edges = len(edges)
            best_adj = [row[:] for row in adj]
            best_edge_list = edges[:]
    
    return best_edge_list, best_edges

def construct():
    n = 40
    deadline = 115  # safety margin before 120s
    start = time.time()
    
    best_edges = []
    best_count = 0
    
    # Multiple random restarts
    for seed in range(200):
        if time.time() - start > deadline:
            break
        edge_list, count = _greedy_build(seed, n, max_time=50)
        if count > best_count:
            best_count = count
            best_edges = edge_list
    
    # Fallback: if we didn't beat the baseline, use the 8xK5 construction
    if best_count < 100:
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    return best_edges

# EVOLVE-BLOCK-END
