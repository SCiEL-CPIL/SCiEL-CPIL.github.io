import numpy as np
import time
import random
import math

def count_c4(adj):
    """Count 4-cycles in graph represented by adjacency matrix."""
    n = adj.shape[0]
    A2 = (adj.astype(np.int32) @ adj.astype(np.int32))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 145

    # === Phase 0: Construct base graph using PG(2,5) polarity ===
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
    
    # Adjust to exactly 91 edges
    if current_edges > m:
        while current_edges > m and time.time() < deadline:
            max_c4s = -1
            worst_edge = None
            for e in list(edge_set):
                u, v = e
                common = np.sum(adj[u] & adj[v])
                if common > max_c4s:
                    max_c4s = common
                    worst_edge = e
            if worst_edge:
                u, v = worst_edge
                adj[u, v] = 0
                adj[v, u] = 0
                edge_set.remove(worst_edge)
                current_edges -= 1
            else:
                break
    elif current_edges < m:
        # Add edges that create fewest C4s
        while current_edges < m and time.time() < deadline:
            min_c4s = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        # Count C4s created: paths i-a-b-j
                        count = 0
                        nu = np.where(adj[i] == 1)[0]
                        nv = np.where(adj[j] == 1)[0]
                        for a in nu:
                            count += np.sum(adj[a, nv])
                        if count < min_c4s:
                            min_c4s = count
                            best_edge = (i, j)
            if best_edge:
                u, v = best_edge
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add(best_edge)
                current_edges += 1
            else:
                break

    # === Phase 1: Exact delta-based local search with 1-swap moves ===
    # Precompute adjacency as bitsets for fast intersection
    adj_bits = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                adj_bits[i, j] = 1

    def c4s_containing_edge(u, v):
        """Count C4s that use edge (u,v) in current graph."""
        # C4s through (u,v): for each common neighbor pair (a,b) with a in N(u)\{v}, b in N(v)\{u}, edge(a,b)
        # Actually: C4 = u-a-b-v. a in N(u), b in N(v), a!=v, b!=u, edge(a,b)
        nu = adj_bits[u]
        nv = adj_bits[v]
        nu[v] = 0
        nv[u] = 0
        count = 0
        for a in range(n):
            if nu[a]:
                count += np.sum(adj_bits[a] & nv)
        return int(count)

    def new_c4s_for_edge(u, v):
        """Count C4s created by adding edge (u,v)."""
        # Same as c4s_containing_edge but for a non-edge
        nu = adj_bits[u]
        nv = adj_bits[v]
        count = 0
        for a in range(n):
            if nu[a]:
                count += np.sum(adj_bits[a] & nv)
        return int(count)

    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    best_adj_bits = adj_bits.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        # Compute C4 contribution for each edge and non-edge
        edge_c4s = {}
        for e in edges_list:
            edge_c4s[e] = c4s_containing_edge(e[0], e[1])
        
        non_edge_c4s = {}
        for e in non_edges:
            non_edge_c4s[e] = new_c4s_for_edge(e[0], e[1])
        
        # Find best swap: minimize (c4s_added - c4s_removed)
        best_delta = 0
        best_swap = None
        for e_rm in edges_list:
            c4s_rm = edge_c4s[e_rm]
            for e_add in non_edges:
                c4s_add = non_edge_c4s[e_add]
                delta = c4s_add - c4s_rm
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e_rm, e_add)
        
        if best_swap and best_delta < 0:
            e_rm, e_add = best_swap
            u1, v1 = e_rm
            u2, v2 = e_add
            
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            adj_bits[u1, v1] = 0
            adj_bits[v1, u1] = 0
            adj_bits[u2, v2] = 1
            adj_bits[v2, u2] = 1
            
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            
            new_c4 = count_c4(adj)
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                best_adj_bits = adj_bits.copy()
                current_c4 = new_c4
                improved = True
            else:
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                adj_bits[u1, v1] = 1
                adj_bits[v1, u1] = 1
                adj_bits[u2, v2] = 0
                adj_bits[v2, u2] = 0
                edge_set.remove(e_add)
                edge_set.add(e_rm)

    # === Phase 2: Simulated Annealing with fast delta estimation ===
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    adj_bits = best_adj_bits.copy()
    current_c4 = best_c4
    
    T = 8.0
    T_min = 0.05
    cooling_rate = 0.997
    
    while T > T_min and time.time() < deadline:
        # Random edge to remove
        e_rm = random.choice(list(edge_set))
        u1, v1 = e_rm
        
        # Random non-edge to add
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 > v2:
                u2, v2 = v2, u2
            if u2 == v2:
                continue
            if (u2, v2) not in edge_set:
                break
        
        # Compute delta
        c4s_rm = c4s_containing_edge(u1, v1)
        c4s_add = new_c4s_for_edge(u2, v2)
        delta = c4s_add - c4s_rm
        
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        adj_bits[u1, v1] = 0
        adj_bits[v1, u1] = 0
        adj_bits[u2, v2] = 1
        adj_bits[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        actual_delta = new_c4 - current_c4
        
        if actual_delta < 0 or random.random() < math.exp(-actual_delta / T):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add((u2, v2))
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                best_adj_bits = adj_bits.copy()
        else:
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            adj_bits[u1, v1] = 1
            adj_bits[v1, u1] = 1
            adj_bits[u2, v2] = 0
            adj_bits[v2, u2] = 0
            
        T *= cooling_rate

    # === Phase 3: Final local search polish ===
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    adj_bits = best_adj_bits.copy()
    current_c4 = best_c4
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        edge_c4s = {}
        for e in edges_list:
            edge_c4s[e] = c4s_containing_edge(e[0], e[1])
        
        non_edge_c4s = {}
        for e in non_edges:
            non_edge_c4s[e] = new_c4s_for_edge(e[0], e[1])
        
        best_delta = 0
        best_swap = None
        for e_rm in edges_list:
            c4s_rm = edge_c4s[e_rm]
            for e_add in non_edges:
                c4s_add = non_edge_c4s[e_add]
                delta = c4s_add - c4s_rm
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e_rm, e_add)
        
        if best_swap and best_delta < 0:
            e_rm, e_add = best_swap
            u1, v1 = e_rm
            u2, v2 = e_add
            
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            adj_bits[u1, v1] = 0
            adj_bits[v1, u1] = 0
            adj_bits[u2, v2] = 1
            adj_bits[v2, u2] = 1
            
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            
            new_c4 = count_c4(adj)
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                best_adj_bits = adj_bits.copy()
                current_c4 = new_c4
                improved = True
            else:
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                adj_bits[u1, v1] = 1
                adj_bits[v1, u1] = 1
                adj_bits[u2, v2] = 0
                adj_bits[v2, u2] = 0
                edge_set.remove(e_add)
                edge_set.add(e_rm)

    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
