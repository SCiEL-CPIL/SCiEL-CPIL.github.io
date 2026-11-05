import random
import time
import math
import sys
from itertools import combinations

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    def build_adj(edges):
        """Build adjacency bitsets and degrees from edge set."""
        adj = [0] * N
        deg = [0] * N
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def find_triangles(adj, deg):
        """Find all triangles and compute max degree sum."""
        max_sum = 0
        tri_list = []
        for i in range(N):
            for j in range(i + 1, N):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j
                common = adj[i] & adj[j]
                # Only consider k > j
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = (common & -common).bit_length() - 1
                    common &= common - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
                    tri_list.append((i, j, k))
        return max_sum, tri_list
    
    def make_k88_plus_edge(extra_edge):
        """Create K_{8,8} plus one extra edge."""
        edges = set()
        for i in range(8):
            for j in range(8, 16):
                edges.add((i, j))
        edges.add(extra_edge)
        return list(edges)
    
    def make_random_graph():
        """Create a random graph with 65 edges."""
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        return all_edges[:TARGET_EDGES]
    
    def make_near_regular():
        """Create a near-regular graph with 65 edges.
        65 edges, 16 vertices: average degree = 130/16 = 8.125
        So 2 vertices have degree 9, 14 have degree 8.
        """
        # Start with 8-regular (64 edges) and add 1 edge
        # 8-regular on 16 vertices: circulant with S = {1,2,3,4,5,6,7,8}
        # But that's 16*8/2 = 64 edges
        edges = set()
        for i in range(N):
            for d in range(1, 9):
                j = (i + d) % N
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
        # Now add one more edge
        all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
        missing = [e for e in all_possible if e not in edges]
        if missing:
            edges.add(missing[0])
        return list(edges)
    
    def local_search(adj, deg, max_sum, tri_list, max_time, seed=0):
        """Fast local search using bitset operations."""
        t0_ls = time.time()
        rng_ls = random.Random(seed)
        
        best_score = max_sum
        best_adj = adj[:]
        best_deg = deg[:]
        
        # Build edge list
        edge_set = set()
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edge_set.add((i, j))
        
        # Non-edge list
        nonedge_set = set()
        for i in range(N):
            for j in range(i + 1, N):
                if not (adj[i] >> j) & 1:
                    nonedge_set.add((i, j))
        
        edges = list(edge_set)
        nonedges = list(nonedge_set)
        
        iters = 0
        stall = 0
        stall_limit = 5000
        
        while time_left() > 0.5:
            if iters % 1000 == 0 and stall > stall_limit:
                break
            
            # Random swap
            if not edges or not nonedges:
                break
            
            idx_rm = rng_ls.randrange(len(edges))
            idx_add = rng_ls.randrange(len(nonedges))
            e_rm = edges[idx_rm]
            e_add = nonedges[idx_add]
            
            u, v = e_rm
            x, y = e_add
            
            # Check if swap is valid (no duplicate)
            if (u == x and v == y) or (u == y and v == x):
                iters += 1
                continue
            
            # Apply swap
            old_adj = adj[:]
            old_deg = deg[:]
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            # Recompute score
            new_max, _ = find_triangles(adj, deg)
            
            if new_max < best_score:
                best_score = new_max
                best_adj = adj[:]
                best_deg = deg[:]
                stall = 0
            else:
                stall += 1
            
            # Accept if better or equal with some probability
            if new_max <= max_sum or (new_max < max_sum + 2 and rng_ls.random() < 0.3):
                # Keep the swap
                max_sum = new_max
                # Update edge lists
                edge_set.remove(e_rm)
                edge_set.add(e_add)
                nonedge_set.remove(e_add)
                nonedge_set.add(e_rm)
                edges = list(edge_set)
                nonedges = list(nonedge_set)
            else:
                # Undo
                adj = old_adj
                deg = old_deg
            
            iters += 1
        
        return best_adj, best_deg, best_score
    
    best_overall_score = float('inf')
    best_overall_adj = None
    
    # Phase 1: K_{8,8} + edge (0,1) - internal edge
    edges = make_k88_plus_edge((0, 1))
    adj, deg = build_adj(edges)
    max_sum, tri_list = find_triangles(adj, deg)
    best_adj, best_deg, best_score = local_search(adj, deg, max_sum, tri_list, time_left(), seed=100)
    if best_score < best_overall_score:
        best_overall_score = best_score
        best_overall_adj = best_adj
    
    if time_left() < 1.0:
        result_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (best_overall_adj[i] >> j) & 1]
        return {'n': N, 'edges': result_edges}
    
    # Phase 2: K_{8,8} + edge (0, 8) - this is already in K8,8, so use (0, 1) variant
    # Try different internal edges
    for extra in [(0, 1), (0, 2), (7, 0), (3, 4)]:
        if time_left() < 1.0:
            break
        edges = make_k88_plus_edge(extra)
        adj, deg = build_adj(edges)
        max_sum, tri_list = find_triangles(adj, deg)
        best_adj, best_deg, best_score = local_search(adj, deg, max_sum, tri_list, time_left(), seed=200 + extra[0] * 10 + extra[1])
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_adj = best_adj
    
    if time_left() < 1.0:
        result_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (best_overall_adj[i] >> j) & 1]
        return {'n': N, 'edges': result_edges}
    
    # Phase 3: Near-regular start
    edges = make_near_regular()
    adj, deg = build_adj(edges)
    max_sum, tri_list = find_triangles(adj, deg)
    best_adj, best_deg, best_score = local_search(adj, deg, max_sum, tri_list, time_left(), seed=500)
    if best_score < best_overall_score:
        best_overall_score = best_score
        best_overall_adj = best_adj
    
    if time_left() < 1.0:
        result_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (best_overall_adj[i] >> j) & 1]
        return {'n': N, 'edges': result_edges}
    
    # Phase 4: Random restarts
    for seed in [1000, 2000, 3000, 4000]:
        if time_left() < 1.0:
            break
        edges = make_random_graph()
        adj, deg = build_adj(edges)
        max_sum, tri_list = find_triangles(adj, deg)
        best_adj, best_deg, best_score = local_search(adj, deg, max_sum, tri_list, time_left(), seed=seed)
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_adj = best_adj
    
    # Final result
    result_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (best_overall_adj[i] >> j) & 1]
    return {'n': N, 'edges': result_edges}
