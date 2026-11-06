import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

N = 16

def _groetzsch_edges():
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

def _greedy_add_edges(adj, deadline, rng):
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                cnt += 1
                improved = True
    return adj, cnt

def _local_search_fast(adj, deadline, rng, core_edges_set):
    """
    Local search that never removes core edges.
    Does not check 3-colorability (guaranteed by core).
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)

    # Greedy add
    cur_adj, cur_cnt = _greedy_add_edges(cur_adj, deadline, rng)
    if cur_cnt > best_cnt:
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]

    # Swaps
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Get non-core edges
        existing = []
        for u in range(N):
            for v in range(u + 1, N):
                if (cur_adj[u] >> v) & 1:
                    if (u, v) not in core_edges_set and (v, u) not in core_edges_set:
                        existing.append((u, v))
        
        if not existing:
            break

        # Try removing one non-core edge and adding more
        for _ in range(10):
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
                    
        # Try pure add
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

def _solve_milp(core_edges):
    """
    Solve the maximum triangle-free graph problem containing core_edges using MILP.
    """
    # Total edges in K16
    total_edges = N * (N - 1) // 2
    # Map edge to index
    edge_to_idx = {}
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_to_idx[(u, v)] = idx
            idx += 1
    
    # Objective: maximize sum x_e
    c = np.ones(total_edges)
    
    # Constraints:
    # 1. Core edges must be 1
    # 2. Triangle constraints: x_uv + x_vw + x_wu <= 2
    
    # We can use LinearConstraint
    # But scipy.optimize.milp expects A_ub x <= b_ub
    # Let's build the matrix A for triangle constraints.
    
    # Number of triangles in K16 is C(16,3) = 560
    num_triangles = 560
    A = np.zeros((num_triangles, total_edges), dtype=np.int8)
    b = np.full(num_triangles, 2, dtype=np.int8)
    
    tri_idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            for w in range(v + 1, N):
                # Triangle u-v-w
                i_uv = edge_to_idx[(u, v)]
                i_vw = edge_to_idx[(v, w)]
                i_uw = edge_to_idx[(u, w)]
                A[tri_idx, i_uv] = 1
                A[tri_idx, i_vw] = 1
                A[tri_idx, i_uw] = 1
                tri_idx += 1
                
    # Core constraints: x_e = 1 for e in core_edges
    # This can be handled by fixing the variables or adding constraints.
    # scipy.optimize.milp supports bounds.
    # If we set lower bound = 1 and upper bound = 1 for core edges, they are fixed.
    
    lb = np.zeros(total_edges)
    ub = np.ones(total_edges)
    
    for u, v in core_edges:
        if u > v: u, v = v, u
        i = edge_to_idx[(u, v)]
        lb[i] = 1
        ub[i] = 1
        
    bounds = Bounds(lb, ub)
    
    # We don't need explicit equality constraints for core if bounds are set.
    # We only have inequality constraints for triangles.
    # A_ub = A, b_ub = b
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=np.ones(total_edges))
        if res.status == 0:
            x = res.x
            # Reconstruct graph
            adj = [0] * N
            count = 0
            for u in range(N):
                for v in range(u + 1, N):
                    i = edge_to_idx[(u, v)]
                    if x[i] > 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
            return adj, count
    except Exception:
        pass
    return None, 0

def construct():
    start = time.time()
    deadline = start + 140.0

    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    core_edges_set = set()
    for u, v in base_edges:
        if u > v: u, v = v, u
        core_edges_set.add((u, v))
        
    # Try MILP first
    milp_adj, milp_cnt = _solve_milp(base_edges)
    if milp_adj is not None and _is_valid(milp_adj):
        # Verify
        if _is_triangle_free(milp_adj) and not _is_3_colorable(milp_adj):
            # Convert to list
            out = []
            for u in range(N):
                m = milp_adj[u]
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    m ^= lb
                    if v > u:
                        out.append([u, v])
            return out

    # Fallback to fast local search
    best_adj = [a for a in base_adj]
    best_cnt = _edge_count(base_adj)
    rng = random.Random(42)
    
    # Run local search multiple times
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        adj = [a for a in base_adj]
        result_adj, result_cnt = _local_search_fast(adj, deadline, rng, core_edges_set)
        if result_cnt > best_cnt and _is_triangle_free(result_adj):
            # Check 3-colorability just once for the best found
            if not _is_3_colorable(result_adj):
                best_cnt = result_cnt
                best_adj = [a for a in result_adj]

    # Final check
    if not _is_valid(best_adj):
        best_adj = [a for a in base_adj]
        best_cnt = _edge_count(base_adj)

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
