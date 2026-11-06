import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 31
    target_edges = 91
    q = 5

    # 1. Construct the Polarity Graph of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    assert len(points) == 31

    adj_base = np.zeros((n, n), dtype=np.uint8)
    base_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                base_edges.append((i, j))
                adj_base[i, j] = 1
                adj_base[j, i] = 1

    if len(base_edges) != 90:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return [list(e) for e in all_pairs[:91]]

    # Precompute A2 and A3 for base graph
    A_base = adj_base.astype(np.int32)
    A2_base = np.dot(A_base, A_base)
    np.fill_diagonal(A2_base, 0)
    A3_base = np.dot(A2_base, A_base)

    # Find all non-edges sorted by number of C4s they'd create (A3 value)
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_base[i, j] == 0:
                c4s = A3_base[i, j]
                non_edges.append((c4s, i, j))
    non_edges.sort()

    def count_c4(adj_mat):
        A2 = np.dot(adj_mat.astype(np.int32), adj_mat.astype(np.int32))
        np.fill_diagonal(A2, 0)
        prod = A2 * (A2 - 1)
        sum_upper = np.sum(np.triu(prod, 1))
        return sum_upper // 4

    # Try multiple candidate edges for the 91st edge
    best_overall = None
    best_overall_score = float('inf')
    
    num_candidates = min(15, len(non_edges))
    
    for cand_idx in range(num_candidates):
        if time.time() > deadline - 25:
            break
        
        c4s_initial, u_extra, v_extra = non_edges[cand_idx]
        
        # Build initial graph
        edges = list(base_edges) + [(u_extra, v_extra)]
        adj = adj_base.copy()
        adj[u_extra, v_extra] = 1
        adj[v_extra, u_extra] = 1
        
        current_score = count_c4(adj)
        
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = list(edges)
        
        # If we already have 1 C4, this is likely optimal
        if current_score <= 1:
            continue
        
        # Simulated Annealing to improve
        temp = 1.0
        cooling_rate = 0.9995
        sa_edges = list(edges)
        sa_adj = adj.copy()
        sa_current = current_score
        sa_best = current_score
        sa_best_edges = list(sa_edges)
        
        while time.time() < deadline - 5:
            # Pick random edge to remove
            e_idx = random.randint(0, len(sa_edges) - 1)
            u, v = sa_edges[e_idx]
            
            # Pick random non-edge to add
            found = False
            for _ in range(30):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and sa_adj[a, b] == 0:
                    if (a == u and b == v) or (a == v and b == u):
                        continue
                    found = True
                    break
            
            if not found:
                continue
            
            # Apply move
            sa_adj[u, v] = 0
            sa_adj[v, u] = 0
            sa_adj[a, b] = 1
            sa_adj[b, a] = 1
            sa_edges[e_idx] = (a, b)
            
            new_score = count_c4(sa_adj)
            
            if new_score < sa_current:
                sa_current = new_score
                if new_score < sa_best:
                    sa_best = new_score
                    sa_best_edges = list(sa_edges)
                    if sa_best < best_overall_score:
                        best_overall_score = sa_best
                        best_overall = list(sa_edges)
            else:
                delta = new_score - sa_current
                if random.random() < np.exp(-delta / temp):
                    sa_current = new_score
                else:
                    # Revert
                    sa_adj[u, v] = 1
                    sa_adj[v, u] = 1
                    sa_adj[a, b] = 0
                    sa_adj[b, a] = 0
                    sa_edges[e_idx] = (u, v)
            
            temp *= cooling_rate
            if temp < 0.01:
                temp = 0.01
    
    if best_overall is None:
        u_extra, v_extra = non_edges[0][1], non_edges[0][2]
        best_overall = list(base_edges) + [(u_extra, v_extra)]
    
    return [list(e) for e in best_overall]

if __name__ == "__main__":
    edges = construct()
    print(f"Generated {len(edges)} edges")
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A2 = np.dot(adj, adj)
    np.fill_diagonal(A2, 0)
    prod = A2 * (A2 - 1)
    sum_upper = np.sum(np.triu(prod, 1))
    c4s = sum_upper // 4
    print(f"C4 count: {c4s}")
