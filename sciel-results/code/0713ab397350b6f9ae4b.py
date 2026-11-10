import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for return
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # --- Phase 1: Fast Greedy Baseline ---
    def greedy_search():
        adj = np.zeros((n, n), dtype=bool)
        current_edges = []
        # Simple random order
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        # Re-use a simplified Q3 check for greedy?
        # The original creates_q3 was buggy. 
        # We can use the find_q3 logic but it's heavy.
        # For greedy, we can just check if adding edge creates Q3 using the same find_q3 logic on the *current* graph + edge?
        # No, find_q3 checks the whole graph.
        # We can use a faster check: 
        # When adding (u,v), check if (u,v) is part of a Q3.
        # This is the original creates_q3 logic.
        
        def creates_q3_fast(u, v, adj):
            # Check if adding (u,v) creates a Q3.
            # This is equivalent to finding Q3 \ {(u,v)} in adj.
            # Map (u,v) to (0,1).
            
            n_u = np.where(adj[u])[0]
            n_u = n_u[n_u != v]
            n_v = np.where(adj[v])[0]
            n_v = n_v[n_v != u]
            
            if len(n_u) < 2 or len(n_v) < 2:
                return False

            for m2 in n_u:
                for m4 in n_u:
                    if m4 == m2:
                        continue
                    
                    common_24 = np.intersect1d(np.where(adj[m2])[0], np.where(adj[m4])[0])
                    mask_24 = np.ones(n, dtype=bool)
                    mask_24[u] = False
                    mask_24[v] = False
                    mask_24[m2] = False
                    mask_24[m4] = False
                    cand_m6 = common_24[mask_24]
                    
                    if len(cand_m6) == 0:
                        continue
                        
                    for m6 in cand_m6:
                        common_v2 = np.intersect1d(np.where(adj[v])[0], np.where(adj[m2])[0])
                        mask_v2 = np.ones(n, dtype=bool)
                        mask_v2[u] = False
                        mask_v2[v] = False
                        mask_v2[m2] = False
                        mask_v2[m4] = False
                        mask_v2[m6] = False
                        cand_m3 = common_v2[mask_v2]
                        
                        if len(cand_m3) == 0:
                            continue
                            
                        for m3 in cand_m3:
                            common_v4 = np.intersect1d(np.where(adj[v])[0], np.where(adj[m4])[0])
                            mask_v4 = np.ones(n, dtype=bool)
                            mask_v4[u] = False
                            mask_v4[v] = False
                            mask_v4[m2] = False
                            mask_v4[m3] = False
                            mask_v4[m4] = False
                            mask_v4[m6] = False
                            cand_m5 = common_v4[mask_v4]
                            
                            if len(cand_m5) == 0:
                                continue
                                
                            for m5 in cand_m5:
                                common_356 = np.intersect1d(np.intersect1d(np.where(adj[m3])[0], np.where(adj[m5])[0]), np.where(adj[m6])[0])
                                mask_356 = np.ones(n, dtype=bool)
                                mask_356[u] = False
                                mask_356[v] = False
                                mask_356[m2] = False
                                mask_356[m3] = False
                                mask_356[m4] = False
                                mask_356[m5] = False
                                mask_356[m6] = False
                                cand_m7 = common_356[mask_356]
                                
                                if len(cand_m7) > 0:
                                    return True
            return False

        for u, v in edges_copy:
            if time.time() > deadline - 10.0: # Leave time for ILP
                break
            if not creates_q3_fast(u, v, adj):
                adj[u, v] = True
                adj[v, u] = True
                current_edges.append([u, v])
        return current_edges

    best_greedy = greedy_search()
    best_count = len(best_greedy)
    best_edges = best_greedy[:]
    
    # --- Phase 2: ILP Cutting Plane ---
    c = np.full(num_edges, -1.0)
    integrality = np.ones(num_edges, dtype=int)
    bounds = Bounds(lb=np.zeros(num_edges), ub=np.ones(num_edges))
    
    A_rows = []
    b_vals = []
    
    def find_q3_full(adj):
        # Same as before, but optimized slightly
        neighbors = [np.where(adj[i])[0] for i in range(n)]
        for u, v in all_edges:
            if not adj[u, v]:
                continue
            n_u = neighbors[u]
            n_u = n_u[n_u != v]
            n_v = neighbors[v]
            n_v = n_v[n_v != u]
            if len(n_u) < 2 or len(n_v) < 2:
                continue
            for m2 in n_u:
                for m4 in n_u:
                    if m4 == m2:
                        continue
                    common_24 = np.intersect1d(neighbors[m2], neighbors[m4])
                    mask_24 = np.ones(n, dtype=bool)
                    mask_24[u] = False
                    mask_24[v] = False
                    mask_24[m2] = False
                    mask_24[m4] = False
                    cand_m6 = common_24[mask_24]
                    if len(cand_m6) == 0:
                        continue
                    for m6 in cand_m6:
                        common_v2 = np.intersect1d(neighbors[v], neighbors[m2])
                        mask_v2 = np.ones(n, dtype=bool)
                        mask_v2[u] = False
                        mask_v2[v] = False
                        mask_v2[m2] = False
                        mask_v2[m4] = False
                        mask_v2[m6] = False
                        cand_m3 = common_v2[mask_v2]
                        if len(cand_m3) == 0:
                            continue
                        for m3 in cand_m3:
                            common_v4 = np.intersect1d(neighbors[v], neighbors[m4])
                            mask_v4 = np.ones(n, dtype=bool)
                            mask_v4[u] = False
                            mask_v4[v] = False
                            mask_v4[m2] = False
                            mask_v4[m3] = False
                            mask_v4[m4] = False
                            mask_v4[m6] = False
                            cand_m5 = common_v4[mask_v4]
                            if len(cand_m5) == 0:
                                continue
                            for m5 in cand_m5:
                                common_356 = np.intersect1d(np.intersect1d(neighbors[m3], neighbors[m5]), neighbors[m6])
                                mask_356 = np.ones(n, dtype=bool)
                                mask_356[u] = False
                                mask_356[v] = False
                                mask_356[m2] = False
                                mask_356[m3] = False
                                mask_356[m4] = False
                                mask_356[m5] = False
                                mask_356[m6] = False
                                cand_m7 = common_356[mask_356]
                                if len(cand_m7) > 0:
                                    m7 = cand_m7[0]
                                    return [
                                        (u, v), (u, m2), (u, m4), (v, m3), (v, m5),
                                        (m2, m3), (m2, m6), (m4, m5), (m4, m6),
                                        (m3, m7), (m5, m7), (m6, m7)
                                    ]
        return None

    while time.time() < deadline - 5.0:
        if A_rows:
            A_ub = np.array(A_rows)
            b_ub = np.array(b_vals)
            constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        else:
            constraints = LinearConstraint(np.zeros((0, num_edges)), -np.inf, np.zeros(0))
            
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        
        if not res.success:
            break
            
        x = res.x
        adj = np.zeros((n, n), dtype=bool)
        current_edges = []
        for idx, val in enumerate(x):
            if val > 0.5:
                u, v = all_edges[idx]
                current_edges.append([u, v])
                adj[u, v] = True
                adj[v, u] = True
                
        q3 = find_q3_full(adj)
        
        if q3 is None:
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
            break
        else:
            row = np.zeros(num_edges)
            for e in q3:
                row[edge_index[e]] = 1.0
            A_rows.append(row)
            b_vals.append(11.0)

    return best_edges
