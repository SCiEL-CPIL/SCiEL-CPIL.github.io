import numpy as np
import time
import random
from collections import defaultdict

def construct():
    n = 20
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        A = np.zeros((n, n), dtype=int)
        K = np.zeros((n, n), dtype=int)
        edges = 0
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            valid = True
            for x in range(n):
                if A[u, x] == 1:
                    if K[x, v] > 1:
                        valid = False
                        break
            if valid:
                for y in range(n):
                    if A[v, y] == 1:
                        if K[u, y] > 1:
                            valid = False
                            break
            
            if valid:
                A[u, v] = 1
                A[v, u] = 1
                edges += 1
                for x in range(n):
                    if A[u, x] == 1 and x != v:
                        K[x, v] += 1
                        K[v, x] += 1
                for y in range(n):
                    if A[v, y] == 1 and y != u:
                        K[u, y] += 1
                        K[y, u] += 1
        
        res_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    res_edges.append([u, v])
        return res_edges

    best_fallback_edges = []
    best_fallback_count = 0
    for s in range(5):
        e = run_greedy(s)
        if len(e) > best_fallback_count:
            best_fallback_count = len(e)
            best_fallback_edges = e
            
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        x_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        x_idx = {p: i for i, p in enumerate(x_pairs)}
        num_x = len(x_pairs)
        
        y_triples = []
        for u, v in x_pairs:
            for w in range(n):
                if w != u and w != v:
                    y_triples.append((u, v, w))
        y_idx = {t: i for i, t in enumerate(y_triples)}
        num_y = len(y_triples)
        
        num_vars = num_x + num_y
        
        c = np.zeros(num_vars)
        for i in range(num_x):
            c[i] = -1.0
            
        A_rows = []
        b_rows = []
        
        for (u, v, w), y_i in y_idx.items():
            x_uv_i = x_idx[(u, v)]
            if v < w:
                x_vw_i = x_idx[(v, w)]
            else:
                x_vw_i = x_idx[(w, v)]
                
            row = np.zeros(num_vars)
            row[y_i] = 1.0
            row[x_uv_i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            row = np.zeros(num_vars)
            row[y_i] = 1.0
            row[x_vw_i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            row = np.zeros(num_vars)
            row[x_uv_i] = 1.0
            row[x_vw_i] = 1.0
            row[y_i] = -1.0
            A_rows.append(row)
            b_rows.append(1.0)
            
        y_groups = defaultdict(list)
        for (u, v, w), y_i in y_idx.items():
            y_groups[(u, v)].append(y_i)
            
        for (u, v), indices in y_groups.items():
            row = np.zeros(num_vars)
            for idx in indices:
                row[idx] = 1.0
            A_rows.append(row)
            b_rows.append(2.0)
            
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        integrality = np.ones(num_vars)
        
        lb = np.zeros(num_vars)
        ub = np.ones(num_vars)
        
        res = milp(c,
                   constraints=LinearConstraint(A_ub, -np.inf, b_ub),
                   integrality=integrality,
                   bounds=Bounds(lb, ub),
                   options={"time_limit": 100})
        
        if res.success and res.x is not None:
            x_sol = res.x[:num_x]
            edges = []
            for i, (u, v) in enumerate(x_pairs):
                if x_sol[i] > 0.5:
                    edges.append([u, v])
            
            A = np.zeros((n, n), dtype=int)
            for u, v in edges:
                A[u, v] = 1
                A[v, u] = 1
            K = A @ A
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    if K[u, v] > 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(edges) > best_fallback_count:
                return edges
            elif valid:
                return edges
            else:
                return best_fallback_edges
        else:
            return best_fallback_edges

    except Exception:
        return best_fallback_edges
