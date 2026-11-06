import random
import time
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before the 150s kill

def _score_from_adj(adj):
    """O(n^3) score: max deg(u)+deg(v)+deg(w) over all triangles."""
    deg = [0]*N
    for i in range(N):
        deg[i] = sum(adj[i])
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _score_from_adj_np(adj_np, deg_np):
    """Vectorized score computation using numpy. adj_np is 0/1 int array, deg_np is degree array."""
    # Triangles: for each pair (i,j) that is an edge, count common neighbors
    # A triangle (i,j,k) exists if adj[i][j] & adj[i][k] & adj[j][k]
    # We need max over triangles of deg[i]+deg[j]+deg[k]
    
    # Use matrix multiplication to find triangles
    # T[i][j] = number of common neighbors of i and j
    T = adj_np @ adj_np
    # Triangles are where T[i][j] > 0 and adj[i][j] == 1
    # For each triangle (i,j,k), the degree sum is deg[i]+deg[j]+deg[k]
    
    # Vectorized approach: iterate over all pairs
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if adj_np[i, j]:
                # Find all k > j such that adj[i,k] and adj[j,k]
                common = (adj_np[i, :] & adj_np[j, :]).astype(np.int32)
                # Only consider k > j to avoid double counting
                common[:j+1] = 0
                for k in np.nonzero(common)[0]:
                    s = deg_np[i] + deg_np[j] + deg_np[k]
                    if s > best:
                        best = s
    return best

def _count_edges(adj):
    return sum(sum(row) for row in adj) // 2

def _get_edges_and_nonedges(adj):
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append((i,j))
            else:
                nonedges.append((i,j))
    return edges, nonedges

def _apply_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1

def _get_all_triangles(adj):
    """Return list of all triangles (i,j,k) with i<j<k."""
    triangles = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        triangles.append((i, j, k))
    return triangles

def _compute_score_from_triangles(triangles, deg):
    """Compute max degree sum over given triangles."""
    if not triangles:
        return 0
    best = 0
    for (i, j, k) in triangles:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best

def _try_swap_score(adj, deg, triangles, cur_score, e_remove, e_add):
    """
    Compute new score if we swap e_remove for e_add.
    Returns (new_score, new_triangles_list) or None if invalid.
    Does NOT modify adj, deg, or triangles.
    """
    u, v = e_remove
    x, y = e_add
    
    # Check validity
    if not adj[u][v]:
        return None
    if adj[x][y]:
        return None
    
    # Compute new degrees
    new_deg = deg[:]
    new_deg[u] -= 1
    new_deg[v] -= 1
    new_deg[x] += 1
    new_deg[y] += 1
    
    # Triangles destroyed: contain both u and v
    destroyed_set = set()
    for tri in triangles:
        if u in tri and v in tri:
            destroyed_set.add(tri)
    
    # Triangles created: contain both x and y, and all three edges exist after swap
    created_list = []
    tri_set = set(triangles)
    for z in range(N):
        if z != x and z != y:
            # After swap, adj[x][y] = True
            # Need adj[x][z] and adj[y][z] to be True (unchanged by swap unless z is u or v)
            adj_xz = adj[x][z]
            adj_yz = adj[y][z]
            
            # If z == u, then adj[x][z] = adj[x][u] which might be affected if (x,u) == (u,v) or (x,u) == (v,u)
            if x == u and z == v:
                adj_xz = False
            elif x == v and z == u:
                adj_xz = False
            if y == u and z == v:
                adj_yz = False
            elif y == v and z == u:
                adj_yz = False
            
            if adj_xz and adj_yz:
                a, b, c = sorted([x, y, z])
                new_tri = (a, b, c)
                if new_tri not in tri_set:
                    created_list.append(new_tri)
    
    # Build new triangle list
    new_triangles = [tri for tri in triangles if tri not in destroyed_set]
    new_triangles.extend(created_list)
    
    # Compute new score
    new_score = _compute_score_from_triangles(new_triangles, new_deg)
    
    return new_score, new_triangles

def _init_adj(a, b, rng, needed):
    """Create initial adjacency list with K_{a,b} plus internal edges."""
    adj = [[False]*N for _ in range(N)]
    for i in range(a):
        for j in range(a, a+b):
            adj[i][j] = adj[j][i] = True
    
    if needed > 0:
        internal_pairs = []
        for i in range(a):
            for j in range(i+1, a):
                internal_pairs.append((i, j))
        for i in range(a, a+b):
            for j in range(i+1, a+b):
                internal_pairs.append((i, j))
        
        if len(internal_pairs) >= needed:
            selected = rng.sample(internal_pairs, needed)
            for (u, v) in selected:
                adj[u][v] = adj[v][u] = True
    
    return adj

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    # Multiple restarts with different initial structures
    num_restarts = 25
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Initialize: K_{a,b} + internal edges
        splits = [(8, 8), (7, 9), (9, 7), (6, 10), (10, 6), (5, 11), (11, 5), (4, 12), (12, 4)]
        a, b = splits[r % len(splits)]
        
        needed = 65 - a * b
        
        if needed < 0 or needed > a*(a-1)//2 + b*(b-1)//2:
            a, b = 8, 8
            needed = 1
        
        adj = _init_adj(a, b, rng, needed)
        
        # Verify we have exactly 65 edges
        ec = _count_edges(adj)
        if ec != 65:
            adj = [[False]*N for _ in range(N)]
            for i in range(8):
                for j in range(8, 16):
                    adj[i][j] = adj[j][i] = True
            adj[0][1] = adj[1][0] = True
        
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
        
        # Get initial triangles
        triangles = _get_all_triangles(adj)
        
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Simulated Annealing with incremental scoring
        start_temp = 5.0
        end_temp = 0.05
        max_iters = 20000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 1:
                break
            
            # Pick random edge to remove
            e_remove = edges[rng.randrange(len(edges))]
            # Pick random non-edge to add
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            if (u == x and v == y) or (u == y and v == x):
                temp *= cooling_rate
                continue
            
            result = _try_swap_score(adj, deg, triangles, cur_score, e_remove, e_add)
            
            if result is None:
                temp *= cooling_rate
                continue
            
            new_score, new_triangles = result
            
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                _apply_swap(adj, deg, e_remove, e_add)
                cur_score = new_score
                triangles = new_triangles
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            
            temp *= cooling_rate
        
        if time.time() - t0 > DEADLINE - 5:
            break
    
    # Final hill-climbing polish on the best solution
    if best_adj is not None and time.time() - t0 < DEADLINE - 10:
        adj = [row[:] for row in best_adj]
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj)
        triangles = _get_all_triangles(adj)
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Aggressive hill climbing: try all single swaps, accept if better
        improved = True
        while improved and time.time() - t0 < DEADLINE - 5:
            improved = False
            for ei in range(len(edges)):
                if time.time() - t0 > DEADLINE - 5:
                    break
                e_remove = edges[ei]
                for ni in range(len(nonedges)):
                    e_add = nonedges[ni]
                    u, v = e_remove
                    x, y = e_add
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    
                    result = _try_swap_score(adj, deg, triangles, cur_score, e_remove, e_add)
                    if result is None:
                        continue
                    
                    new_score, new_triangles = result
                    
                    if new_score < cur_score:
                        _apply_swap(adj, deg, e_remove, e_add)
                        cur_score = new_score
                        triangles = new_triangles
                        
                        if cur_score < best_score:
                            best_score = cur_score
                            best_adj = [row[:] for row in adj]
                            best_deg = deg[:]
                        
                        edges.remove(e_remove)
                        nonedges.append(e_remove)
                        nonedges.remove(e_add)
                        edges.append(e_add)
                        improved = True
                        break
                if improved:
                    break
    
    if best_adj is None:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    ec = _count_edges(best_adj)
    if ec != TARGET_EDGES:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_adj[i][j]]}
