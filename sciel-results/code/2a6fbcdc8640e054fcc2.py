import numpy as np
import time
import random

def count_c4(adj):
    """
    Count the number of 4-cycles in a graph represented by an adjacency matrix.
    A 4-cycle u-w-x-v-u corresponds to 4 distinct vertices u,w,x,v.
    The number of such cycles is (1/8) * sum over all pairs (u,v) of (common_neighbors(u,v))^2?
    No, that counts closed walks of length 4.
    Number of C4s = 1/8 * sum_{u,v} (A^2[u,v])^2 - ...
    Actually, simpler:
    For each pair of vertices (u, v), let k be the number of common neighbors.
    Each C4 u-w-x-v-u is counted 4 times (once for each edge)?
    No, a C4 has 4 pairs of opposite vertices.
    For a specific C4 u-w-x-v, the pairs of opposite vertices are (u,x) and (w,v).
    u and x have 2 common neighbors (w and v).
    w and v have 2 common neighbors (u and x).
    So each C4 contributes 2 to the sum of C(k,2) over all pairs?
    Sum_{u<v} C(common(u,v), 2) = 2 * (Number of C4s).
    So Number of C4s = 1/2 * Sum_{u<v} C(common(u,v), 2).
    """
    n = adj.shape[0]
    c4_count = 0
    # Compute A^2
    # A^2[u,v] is the number of paths of length 2 from u to v, i.e., common neighbors.
    A2 = adj @ adj
    for u in range(n):
        for v in range(u + 1, n):
            k = A2[u, v]
            c4_count += k * (k - 1) // 2
    return c4_count // 2

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 110
    
    # 1. Initialize with a greedy C4-free graph + 1 edge (from seed logic)
    # We'll run a quick greedy construction to get a good starting point.
    
    def greedy_construct(seed):
        random.seed(seed)
        all_pairs = list([(i, j) for i in range(n) for j in range(i + 1, n)])
        random.shuffle(all_pairs)
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        for u, v in all_pairs:
            if len(edges) == 90:
                break
            Nu = np.where(adj[u] == 1)[0]
            Nv = np.where(adj[v] == 1)[0]
            if len(Nu) == 0 or len(Nv) == 0:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
            else:
                sub = adj[np.ix_(Nu, Nv)]
                if np.sum(sub) == 0:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edges.append((u, v))
        
        if len(edges) < 90:
            return None, None
            
        # Add best edge
        A = adj.copy()
        A3 = A @ A @ A
        min_cost = float('inf')
        best_edge = None
        for u, v in all_pairs:
            if adj[u, v] == 0:
                cost = A3[u, v]
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (u, v)
                    
        if best_edge is None:
            return None, None
            
        adj[best_edge[0], best_edge[1]] = 1
        adj[best_edge[1], best_edge[0]] = 1
        edges.append(best_edge)
        return adj, edges

    # Try to get a good start
    start_adj = None
    start_edges = None
    for s in range(10):
        start_adj, start_edges = greedy_construct(s)
        if start_adj is not None:
            break
            
    if start_adj is None:
        # Fallback to random
        random.seed(0)
        all_pairs = list([(i, j) for i in range(n) for j in range(i + 1, n)])
        random.shuffle(all_pairs)
        start_edges = all_pairs[:91]
        start_adj = np.zeros((n, n), dtype=np.int8)
        for u, v in start_edges:
            start_adj[u, v] = 1
            start_adj[v, u] = 1

    current_adj = start_adj.copy()
    current_edges = set(start_edges)
    current_c4 = count_c4(current_adj)
    
    best_adj = current_adj.copy()
    best_c4 = current_c4
    best_edges = list(current_edges)
    
    # 2. Simulated Annealing
    # Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    all_pairs_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline and T > T_min:
        # Pick random edge to remove
        out_edge = random.choice(list(current_edges))
        u_out, v_out = out_edge
        
        # Pick random non-edge to add
        # Efficiently pick from non-edges
        # We can just pick a random pair and check if it's in the set
        # Or maintain a list of non-edges, but that's slow to update.
        # Given n=31, total pairs 465. Current edges 91. Non-edges 374.
        # Picking a random pair and checking membership is O(1) average if we use a set.
        
        while True:
            in_edge = random.choice(all_pairs_list)
            if in_edge not in current_edges:
                break
                
        u_in, v_in = in_edge
        
        # Calculate Delta C4
        # Remove out_edge
        current_adj[u_out, v_out] = 0
        current_adj[v_out, u_out] = 0
        current_edges.discard(out_edge)
        
        # Add in_edge
        current_adj[u_in, v_in] = 1
        current_adj[v_in, u_in] = 1
        current_edges.add(in_edge)
        
        new_c4 = count_c4(current_adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = current_adj.copy()
                best_edges = list(current_edges)
        else:
            # Reject
            current_adj[u_out, v_out] = 1
            current_adj[v_out, u_out] = 1
            current_edges.add(out_edge)
            
            current_adj[u_in, v_in] = 0
            current_adj[v_in, u_in] = 0
            current_edges.discard(in_edge)
            
        T *= alpha
        
    # Final check
    if best_c4 > 1:
        # Try a few more greedy restarts if we didn't get 1
        pass # Already did in init
        
    return [[u, v] for u, v in best_edges]
