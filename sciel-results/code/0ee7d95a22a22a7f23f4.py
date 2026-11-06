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

def count_c4s_containing_edge(adj, u, v):
    """
    Count the number of C4s that contain the edge (u, v).
    A C4 containing (u,v) has form u-a-b-v where a in N(u), b in N(v), (a,b) in E, a != b.
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    count = 0
    if len(nu) < len(nv):
        for a in nu:
            if a == v:
                continue
            count += np.sum(adj[a, nv])
    else:
        for b in nv:
            if b == u:
                continue
            count += np.sum(adj[b, nu])
    return int(count)

def count_new_c4s_for_edge(adj, u, v):
    """
    Count the number of new C4s created by adding edge (u, v) to the graph.
    This equals the number of length-3 paths from u to v.
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    count = 0
    if len(nu) < len(nv):
        for a in nu:
            if a == v:
                continue
            count += np.sum(adj[a, nv])
    else:
        for b in nv:
            if b == u:
                continue
            count += np.sum(adj[b, nu])
    return int(count)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 130  # 20s margin from 150s limit

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

    if edges_to_add > 0:
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

        min_c4s = float('inf')
        best_edge = None

        for u, v in all_possible_edges:
            c = count_new_c4s_for_edge(adj, u, v)
            if c < min_c4s:
                min_c4s = c
                best_edge = (u, v)

        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)

            added = 1
            while added < edges_to_add and time.time() < deadline:
                all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
                min_c4s_iter = float('inf')
                best_edge_iter = None
                for u, v in all_possible_edges:
                    c = count_new_c4s_for_edge(adj, u, v)
                    if c < min_c4s_iter:
                        min_c4s_iter = c
                        best_edge_iter = (u, v)
                if best_edge_iter:
                    u, v = best_edge_iter
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edge_set.add(best_edge_iter)
                    added += 1
                else:
                    break

    # Phase 1: Exhaustive 1-edge-swap local search
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

        edge_c4s = {}
        for e in current_edges_list:
            edge_c4s[e] = count_c4s_containing_edge(adj, e[0], e[1])

        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            c4s_rm = edge_c4s[e_rm]
            for e_add in non_edges:
                u2, v2 = e_add
                c4s_add = count_new_c4s_for_edge(adj, u2, v2)
                delta = c4s_add - c4s_rm
                if delta < 0:
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1

                    new_c4 = count_c4(adj)

                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = new_c4
                        improved = True
                        adj = best_adj.copy()
                        break
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

    # Phase 2: Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999

    current_edges_list = list(edge_set)
    current_c4 = best_c4
    adj = best_adj.copy()
    edge_set = set(best_edge_set)

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

        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1

        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add

            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0

        T *= cooling_rate

    # Phase 3: Final local search
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

        edge_c4s = {}
        for e in current_edges_list:
            edge_c4s[e] = count_c4s_containing_edge(adj, e[0], e[1])

        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            c4s_rm = edge_c4s[e_rm]
            for e_add in non_edges:
                u2, v2 = e_add
                c4s_add = count_new_c4s_for_edge(adj, u2, v2)
                delta = c4s_add - c4s_rm
                if delta < 0:
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1

                    new_c4 = count_c4(adj)

                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = new_c4
                        improved = True
                        adj = best_adj.copy()
                        break
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
