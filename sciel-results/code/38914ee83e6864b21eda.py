import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_c4_fast(adj):
    """
    Fast C4 count using only the upper triangle of A^2.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    vals = A2[np.triu_indices(n, k=1)].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_c4s_containing_edge(u, v, adj):
    """
    Count the number of 4-cycles that use the edge (u, v).
    Works for both edges and non-edges (counting C4s that would be formed).
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    
    # Exclude v from nu and u from nv
    nu_mask = nu != v
    nv_mask = nv != u
    nu_f = nu[nu_mask]
    nv_f = nv[nv_mask]
    
    count = 0
    if len(nu_f) < len(nv_f):
        for w in nu_f:
            count += np.sum(adj[w, nv_f])
    else:
        for x in nv_f:
            count += np.sum(adj[x, nu_f])
            
    return count

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 135  # 15s margin from 150s limit

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    edges_to_add = m - current_edges
    
    # Calculate initial C4 count
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)

    if edges_to_add > 0:
        # Greedy addition of edges
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for _ in range(edges_to_add):
            if time.time() > deadline:
                break
                
            min_c4s = float('inf')
            best_edge = None
            
            for u, v in all_possible_edges:
                if (u, v) in edge_set:
                    continue
                
                cost = count_c4s_containing_edge(u, v, adj)
                if cost < min_c4s:
                    min_c4s = cost
                    best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add(best_edge)
                current_c4 += min_c4s
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
            else:
                break

    # Phase 1: Exhaustive 1-edge-swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            
            u1, v1 = e_rm
            lost = count_c4s_containing_edge(u1, v1, adj)
            
            for e_add in non_edges:
                u2, v2 = e_add
                gained = count_c4s_containing_edge(u2, v2, adj)
                
                delta = gained - lost
                
                if delta < 0:
                    # Apply swap
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_c4 = count_c4_fast(adj)
                    
                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = new_c4
                        improved = True
                        break
                    else:
                        # Reject
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

            if improved:
                break

    # Phase 2: Simulated Annealing
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_edges_list = list(edge_set)
    
    while T > T_min and time.time() < deadline:
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e_add = (u2, v2)
            if e_add not in edge_set:
                break
        
        lost = count_c4s_containing_edge(u1, v1, adj)
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        gained = count_c4s_containing_edge(u2, v2, adj)
        
        delta = gained - lost
        
        # Exact check if delta is promising or random decision
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Verify exact count
            new_c4 = count_c4_fast(adj)
            
            if delta < 0:
                # If delta was negative, it should be an improvement, but verify
                pass 
            else:
                # If delta was positive, we accepted it due to SA.
                # We should only keep it if it's actually an improvement over best?
                # No, SA explores. We track best.
                pass
            
            # Update current state
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            # Reject
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate

    # Phase 3: Final local search pass
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            
            u1, v1 = e_rm
            lost = count_c4s_containing_edge(u1, v1, adj)
            
            for e_add in non_edges:
                u2, v2 = e_add
                gained = count_c4s_containing_edge(u2, v2, adj)
                
                delta = gained - lost
                
                if delta < 0:
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_c4 = count_c4_fast(adj)
                    
                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = new_c4
                        improved = True
                        break
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

            if improved:
                break

    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
