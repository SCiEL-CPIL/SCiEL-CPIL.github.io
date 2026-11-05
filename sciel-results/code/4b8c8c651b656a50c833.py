import itertools
import random
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16
N2 = N * N

points = list(itertools.product(range(N), repeat=2))

dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

iso_with = [[0] * N2 for _ in range(N2)]
iso_triples = set()
for i in range(N2):
    for j in range(i + 1, N2):
        mask = 0
        for k in range(N2):
            if k == i or k == j:
                continue
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                mask |= (1 << k)
                triple = tuple(sorted((i, j, k)))
                iso_triples.add(triple)
        iso_with[i][j] = mask
        iso_with[j][i] = mask

def is_valid_maximal(S):
    if not S:
        return False
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            mask = iso_with[a][b]
            for k in S:
                if k == a or k == b:
                    continue
                if (mask >> k) & 1:
                    return False
    S_bits = 0
    for p in S:
        S_bits |= (1 << p)
    all_points_mask = (1 << N2) - 1
    not_S_mask = all_points_mask ^ S_bits
    B_mask = 0
    for i in range(len(S)):
        a = S[i]
        for j in range(len(S)):
            if i == j: continue
            b = S[j]
            B_mask |= iso_with[a][b]
    return (B_mask & not_S_mask) == not_S_mask

def solve_ilp(timeout=100):
    start_time = time.time()
    triples_by_point = [[] for _ in range(N2)]
    for t in iso_triples:
        i, j, k = t
        triples_by_point[i].append((j, k))
        triples_by_point[j].append((i, k))
        triples_by_point[k].append((i, j))
        
    z_indices = {}
    var_count = N2
    for i in range(N2):
        for j, k in triples_by_point[i]:
            z_idx = var_count
            var_count += 1
            z_indices[(i, j, k)] = z_idx
            
    total_vars = var_count
    c = np.zeros(total_vars)
    for i in range(N2):
        c[i] = 1.0
        
    rows = []
    cols = []
    vals = []
    b_ub = []
    
    for i, j, k in iso_triples:
        rows.append(len(b_ub)); cols.append(i); vals.append(1.0)
        rows.append(len(b_ub)); cols.append(j); vals.append(1.0)
        rows.append(len(b_ub)); cols.append(k); vals.append(1.0)
        b_ub.append(2.0)
        
    for i in range(N2):
        for j, k in triples_by_point[i]:
            z_idx = z_indices[(i, j, k)]
            rows.append(len(b_ub)); cols.append(z_idx); vals.append(1.0)
            rows.append(len(b_ub)); cols.append(j); vals.append(-1.0)
            b_ub.append(0.0)
            
            rows.append(len(b_ub)); cols.append(z_idx); vals.append(1.0)
            rows.append(len(b_ub)); cols.append(k); vals.append(-1.0)
            b_ub.append(0.0)
            
            rows.append(len(b_ub)); cols.append(z_idx); vals.append(-1.0)
            rows.append(len(b_ub)); cols.append(j); vals.append(1.0)
            rows.append(len(b_ub)); cols.append(k); vals.append(1.0)
            b_ub.append(1.0)
            
    for i in range(N2):
        rows.append(len(b_ub)); cols.append(i); vals.append(1.0)
        for j, k in triples_by_point[i]:
            z_idx = z_indices[(i, j, k)]
            rows.append(len(b_ub)); cols.append(z_idx); vals.append(-1.0)
        b_ub.append(0.0)
        
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_ub), total_vars))
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    integrality = np.ones(total_vars)
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={'time_limit': timeout, 'mip_rel_gap': 0.0}
        )
        if res.status in [0, 3] and res.x is not None:
            x_sol = res.x[:N2]
            S = [i for i in range(N2) if x_sol[i] > 0.5]
            if is_valid_maximal(S):
                return S
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = float('inf')
    best_set = []
    
    # Try ILP
    if time.time() < deadline - 50:
        S_ilp = solve_ilp(timeout=100)
        if S_ilp:
            best_size = len(S_ilp)
            best_set = S_ilp
            
    # Fallback / Local Search
    degrees = [0] * N2
    for i in range(N2):
        for j in range(N2):
            if i != j and iso_with[i][j]:
                degrees[i] += 1
                
    order_high = sorted(range(N2), key=lambda x: -degrees[x])
    order_low = sorted(range(N2), key=lambda x: degrees[x])
    
    while time.time() < deadline - 5:
        if random.random() < 0.5:
            base_order = order_high
        else:
            base_order = order_low
            
        order = list(base_order)
        random.shuffle(order)
        
        S = greedy_construct(order)
        
        if len(S) < best_size:
            S_shrunk = shrink(S)
            if len(S_shrunk) < len(S):
                S = S_shrunk
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
                
    if not best_set:
        S = greedy_construct(list(range(N2)))
        best_set = S if S else [0]

    return [points[i] for i in best_set]
