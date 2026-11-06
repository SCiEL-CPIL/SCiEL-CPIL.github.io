import time
import random
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

N = 16

def _groetzsch_edges():
    """Grötzsch graph (Mycielski of C5): 11 vertices, 20 edges."""
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    orig_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in orig_edges:
        edges.append([v, 5 + u])
        edges.append([u, 5 + v])
    for i in range(5):
        edges.append([10, 5 + i])
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free(adj):
    for u in range(N):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj):
    colors = [-1] * N
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:
                forbidden |= (1 << colors[u])
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def _is_valid(adj):
    return _is_triangle_free(adj) and not _is_3_colorable(adj)

def _try_add_edge(adj, u, v):
    if (adj[u] >> v) & 1:
        return False
    if adj[u] & adj[v]:
        return False
    return True

def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _greedy_add_edges(adj, deadline, rng, prioritize_new=False):
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        
        if prioritize_new:
            def sort_key(e):
                u, v = e
                if u >= 11 or v >= 11:
                    return (0, rng.random())
                else:
                    return (1, rng.random())
            missing.sort(key=sort_key)
        else:
            rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                cnt += 1
                improved = True
    return adj, cnt

def _local_search(adj, deadline, rng, max_swaps=30):
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)

    cur_adj, cur_cnt = _greedy_add_edges(cur_adj, deadline, rng)
    if cur_cnt > best_cnt:
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if (cur_adj[u] >> v) & 1]
        if not existing:
            break

        for _ in range(max_swaps):
            if time.time() > deadline - 2.0:
                break
            rem_u, rem_v = rng.choice(existing)
            adj_copy = [a for a in cur_adj]
            _remove_edge(adj_copy, rem_u, rem_v)
            adj_copy, cnt_copy = _greedy_add_edges(adj_copy, deadline, rng)
            if cnt_copy > cur_cnt:
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cnt_copy
                    best_adj = [a for a in cur_adj]

        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not ((cur_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 2.0:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]
                break

    return best_adj, best_cnt

def _solve_milp(base_adj, deadline):
    if not HAS_SCIPY_MILP:
        return None, -1
        
    edge_indices = {}
    idx = 0
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edge_indices[(u, v)] = idx
            all_edges.append((u, v))
            idx += 1
            
    n_vars = len(all_edges)
    c = -np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    core_edges = _groetzsch_edges()
    for u, v in core_edges:
        i = edge_indices[(min(u,v), max(u,v))]
        lb[i] = 1
        ub[i] = 1
        
    A_rows = []
    b_ub = []
    b_lb = []
    
    for u in range(N):
        for v in range(u + 1, N):
            for w in range(v + 1, N):
                i_uv = edge_indices[(u, v)]
                i_vw = edge_indices[(v, w)]
                i_wu = edge_indices[(u, w)]
                
                row = np.zeros(n_vars)
                row[i_uv] = 1
                row[i_vw] = 1
                row[i_wu] = 1
                A_rows.append(row)
                b_ub.append(2)
                b_lb.append(0)
                
    A = np.array(A_rows)
    b_ub_arr = np.array(b_ub)
    b_lb_arr = np.array(b_lb)
    
    constraints = LinearConstraint(A, b_lb_arr, b_ub_arr)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, bounds=bounds, integrality=np.ones(n_vars))
        if res.status == 0:
            x = res.x
            best_adj = [0] * N
            cnt = 0
            for i, (u, v) in enumerate(all_edges):
                if x[i] > 0.5:
                    _add_edge(best_adj, u, v)
                    cnt += 1
            return best_adj, cnt
        else:
            return None, -1
    except Exception:
        return None, -1

def construct():
    start = time.time()
    deadline = start + 140.0
    
    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    base_cnt = _edge_count(base_adj)
    
    best_adj = [a for a in base_adj]
    best_cnt = base_cnt
    
    if HAS_SCIPY_MILP and time.time() < deadline - 30.0:
        milp_adj, milp_cnt = _solve_milp(base_adj, deadline)
        if milp_adj is not None and milp_cnt > best_cnt:
            if _is_valid(milp_adj):
                best_adj = milp_adj
                best_cnt = milp_cnt
    
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        result_adj, result_cnt = _local_search(best_adj, deadline, rng)
        if result_cnt > best_cnt and _is_valid(result_adj):
            best_adj = result_adj
            best_cnt = result_cnt
            
        for seed in range(10):
            if time.time() > deadline - 5.0:
                break
            rng_r = random.Random(seed * 137 + 42)
            adj_copy = [a for a in base_adj]
            res_adj, res_cnt = _local_search(adj_copy, deadline, rng_r)
            if res_cnt > best_cnt and _is_valid(res_adj):
                best_adj = res_adj
                best_cnt = res_cnt

    if not _is_valid(best_adj):
        best_adj = [a for a in base_adj]
        best_cnt = base_cnt

    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])

    return out
