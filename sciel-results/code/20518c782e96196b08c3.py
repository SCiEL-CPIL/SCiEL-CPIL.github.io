"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 31
M = 91

def count_c4(adj):
    """Count C4s from numpy adjacency matrix."""
    common = adj @ adj
    c4 = 0
    for u in range(N):
        for v in range(u+1, N):
            c = common[u,v]
            if c >= 2:
                c4 += c*(c-1)//2
    return c4

def delta_add(adj, u, v):
    """C4s created by adding edge (u,v)."""
    nv = adj[v]
    nu = adj[u]
    count = 0
    for a in range(N):
        if nv[a] and a != v:
            for b in range(N):
                if nu[b] and b != u and a != b and adj[a,b]:
                    count += 1
    return count

def make_adj(edges):
    adj = np.zeros((N, N), dtype=np.int32)
    for u, v in edges:
        adj[u,v] = 1
        adj[v,u] = 1
    return adj

def local_search(edges, time_budget):
    """Hill-climbing with 2-opt moves and fast delta computation."""
    adj = make_adj(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = list(edges)
    
    start_time = time.time()
    deadline = start_time + time_budget
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                return best_edges
            u1, v1 = edges[i]
            e1 = (min(u1,v1), max(u1,v1))
            
            remove_delta = delta_add(adj, u1, v1)
            
            adj[u1,v1] = 0
            adj[v1,u1] = 0
            edge_set.discard(e1)
            
            best_move = None
            best_delta = 0
            
            for u2, v2 in all_pairs:
                e2 = (min(u2,v2), max(u2,v2))
                if e2 in edge_set:
                    continue
                add_delta = delta_add(adj, u2, v2)
                delta = add_delta - remove_delta
                if delta < best_delta:
                    best_delta = delta
                    best_move = (u2, v2)
            
            if best_move is not None:
                u2, v2 = best_move
                e2 = (min(u2,v2), max(u2,v2))
                adj[u2,v2] = 1
                adj[v2,u2] = 1
                edge_set.add(e2)
                edges[i] = [u2, v2]
                current_c4 += best_delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = list(edges)
                improved = True
            else:
                adj[u1,v1] = 1
                adj[v1,u1] = 1
                edge_set.add(e1)
        
        if not improved:
            break
    
    return best_edges

def random_regular_init(seed):
    """Random near-6-regular graph with exactly M edges."""
    random.seed(seed)
    target = [6]*27 + [5]*4
    random.shuffle(target)
    
    edges = []
    edge_set = set()
    deg = [0]*N
    
    candidates = [(i,j) for i in range(N) for j in range(i+1,N)]
    random.shuffle(candidates)
    
    for u,v in candidates:
        if len(edges) >= M:
            break
        if deg[u] < target[u] and deg[v] < target[v]:
            edges.append([u,v])
            edge_set.add((u,v))
            deg[u] += 1
            deg[v] += 1
    
    if len(edges) < M:
        for u,v in candidates:
            if len(edges) >= M:
                break
            if (u,v) not in edge_set:
                edges.append([u,v])
                edge_set.add((u,v))
                deg[u] += 1
                deg[v] += 1
    
    return edges

def construct():
    """Main entry: multiple restarts from random regular inits + fast local search."""
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_c4 = float('inf')
    
    for seed in range(200):
        if time.time() > deadline - 12:
            break
        edges = random_regular_init(seed)
        if len(edges) != M:
            continue
        adj = make_adj(edges)
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = list(edges)
        
        if time.time() < deadline - 8:
            ls_edges = local_search(list(edges), 3)
            adj_ls = make_adj(ls_edges)
            c4_ls = count_c4(adj_ls)
            if c4_ls < best_c4:
                best_c4 = c4_ls
                best_edges = list(ls_edges)
    
    if best_edges is None or len(best_edges) != M:
        pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        best_edges = pairs[:M]
    
    return best_edges
# EVOLVE-BLOCK-END
