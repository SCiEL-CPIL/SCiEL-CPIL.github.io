import numpy as np
import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}

    # Precompute C4s
    c4s = []
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v3, v4), max(v3, v4))
                e4 = (min(v1, v3), max(v1, v3))
                
                idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
                c4s.append(idxs)
                
    # Try ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        c = -np.ones(num_edges)
        A = np.zeros((len(c4s), num_edges))
        for k, idxs in enumerate(c4s):
            for idx in idxs:
                A[k, idx] = 1.0
        b = np.full(len(c4s), 3.0)
        constraints = LinearConstraint(A, -np.inf, b)
        integrality = np.ones(num_edges)
        bounds = Bounds(lb=0, ub=1)
        
        start_time = time.time()
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 100.0})
        
        if res.status == 0:
            x = res.x
            selected_edges = []
            for i in range(num_edges):
                if x[i] > 0.5:
                    selected_edges.append(list(all_edges[i]))
            return selected_edges
    except Exception:
        pass

    # Fallback: Greedy
    # Precompute edge to C4s mapping for fast validity check
    edge_c4s = [[] for _ in range(num_edges)]
    for k, idxs in enumerate(c4s):
        for idx in idxs:
            edge_c4s[idx].append(k)

    def is_valid_add(ei, S):
        for c4_idx in edge_c4s[ei]:
            # Check if the other 3 edges in this C4 are in S
            cnt = 0
            for e_idx in c4s[c4_idx]:
                if e_idx in S:
                    cnt += 1
            if cnt == 3:
                return False
        return True

    S = set()
    order = list(range(num_edges))
    random.shuffle(order)
    for ei in order:
        if is_valid_add(ei, S):
            S.add(ei)
            
    return [[all_edges[i][0], all_edges[i][1]] for i in S]
