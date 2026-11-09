"""Optimized Q3-free graph construction on 16 vertices.

Combines bitmask-based Q3 detection + greedy/local-search (current) with
simulated-annealing edge swaps (inspiration) to escape local optima.
"""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 16

    # ---- Precompute Q3 structure ----
    # Q3 vertices 0..7, edges connect vertices differing by exactly 1 bit.
    q3_adj = {i: set() for i in range(8)}
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_adj[i].add(j)
                q3_adj[j].add(i)
                q3_edges.append((i, j))

    # For each Q3 edge (a,b), record the witness structure:
    #   Na = other neighbors of a, Nb = other neighbors of b (each size 2)
    #   p = unique common neighbor of Na (excluding a,b)
    #   q = unique common neighbor of Nb (excluding a,b)
    #   and p-q is a Q3 edge.
    q3_patterns = {}
    for a, b in q3_edges:
        Na = q3_adj[a] - {b}
        Nb = q3_adj[b] - {a}
        if len(Na) != 2 or len(Nb) != 2:
            continue
        common_a = set.intersection(*[q3_adj[x] for x in Na]) - {a, b}
        common_b = set.intersection(*[q3_adj[x] for x in Nb]) - {a, b}
        if len(common_a) == 1 and len(common_b) == 1:
            p = next(iter(common_a))
            q = next(iter(common_b))
            if p in q3_adj[q]:
                q3_patterns[(a, b)] = (tuple(sorted(Na)), tuple(sorted(Nb)), p, q)
                q3_patterns[(b, a)] = (tuple(sorted(Nb)), tuple(sorted(Na)), q, p)

    def check_q3_with_bitmask(adj_bm, u, v):
        """Return True iff the graph (with edge u-v present) is Q3-free,
        assuming every other Q3 embedding must use the new edge (u,v)."""
        nu = adj_bm[u] & ~(1 << v)
        nv = adj_bm[v] & ~(1 << u)
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        forbidden_base = (1 << u) | (1 << v)

        for (Na, Nb, p_idx, q_idx) in q3_patterns.values():
            for i in range(len(nu_list)):
                X1 = nu_list[i]
                for j in range(i + 1, len(nu_list)):
                    X2 = nu_list[j]
                    common_x = (adj_bm[X1] & adj_bm[X2]) & \
                        ~(forbidden_base | (1 << X1) | (1 << X2))
                    if common_x == 0:
                        continue
                    for k in range(len(nv_list)):
                        Z1 = nv_list[k]
                        if Z1 == X1 or Z1 == X2:
                            continue
                        for l in range(k + 1, len(nv_list)):
                            Z2 = nv_list[l]
                            if Z2 == X1 or Z2 == X2 or Z2 == Z1:
                                continue
                            common_z = (adj_bm[Z1] & adj_bm[Z2]) & \
                                ~(forbidden_base | (1 << Z1) | (1 << Z2))
                            if common_z == 0:
                                continue
                            forbidden_pq = forbidden_base | (1 << X1) | \
                                (1 << X2) | (1 << Z1) | (1 << Z2)
                            P_cands = [i for i in range(n)
                                       if (common_x >> i) & 1 and not ((forbidden_pq >> i) & 1)]
                            Q_cands = [i for i in range(n)
                                       if (common_z >> i) & 1 and not ((forbidden_pq >> i) & 1)]
                            for P in P_cands:
                                for Q in Q_cands:
                                    if P != Q and ((adj_bm[P] >> Q) & 1):
                                        return False
        return True

    def build_adj_from_edges(edges):
        adj_bm = [0] * n
        for u, v in edges:
            adj_bm[u] |= (1 << v)
            adj_bm[v] |= (1 << u)
        return adj_bm

    def try_add(adj_bm, u, v):
        """Add edge (u,v); return True and keep it if Q3-free, else revert."""
        adj_bm[u] |= (1 << v)
        adj_bm[v] |= (1 << u)
        if check_q3_with_bitmask(adj_bm, u, v):
            return True
        adj_bm[u] &= ~(1 << v)
        adj_bm[v] &= ~(1 << u)
        return False

    def greedy_fill(adj_bm, edges, candidate_edges, deadline_guard):
        for u, v in candidate_edges:
            if time.time() > deadline_guard:
                break
            if (adj_bm[u] >> v) & 1:
                continue
            if try_add(adj_bm, u, v):
                edges.append((u, v))

    def polish(adj_bm, edges, deadline_guard):
        """Repeatedly sweep all missing edges, adding any that are feasible."""
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        improved = True
        while improved and time.time() < deadline_guard:
            improved = False
            for u, v in combinations(range(n), 2):
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if try_add(adj_bm, u, v):
                    edges.append((u, v))
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    improved = True

    def simulated_annealing(adj_bm, edges, deadline_guard):
        """SA over edge add/remove moves to escape local optima.

        Objective = edge count. Additions always accepted; removals accepted
        with probability exp(-1/T) (cooling) so the search can leave optima.
        """
        if time.time() > deadline_guard:
            return
        all_pairs = list(combinations(range(n), 2))
        T = 1.0
        cool = 0.985
        best_local = len(edges)
        while time.time() < deadline_guard:
            T *= cool
            if T < 1e-4:
                break
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if time.time() > deadline_guard:
                    return
                present = (adj_bm[u] >> v) & 1
                if present:
                    # candidate removal
                    if len(edges) <= 1:
                        continue
                    if random.random() > min(1.0, T):
                        continue
                    adj_bm[u] &= ~(1 << v)
                    adj_bm[v] &= ~(1 << u)
                    edges.remove((u, v) if (u, v) in edges else (v, u))
                    if len(edges) > best_local:
                        best_local = len(edges)
                else:
                    # candidate addition (always accepted if feasible)
                    if try_add(adj_bm, u, v):
                        edges.append((u, v))
                        if len(edges) > best_local:
                            best_local = len(edges)

    def start_graph(part_a_vertices):
        part_a = set(part_a_vertices)
        part_b = [v for v in range(n) if v not in part_a]
        adj_bm = [0] * n
        edges = []
        for a in part_a:
            for b in part_b:
                adj_bm[a] |= (1 << b)
                adj_bm[b] |= (1 << a)
                edges.append((a, b))
        for i in range(len(part_a_vertices)):
            for j in range(i + 1, len(part_a_vertices)):
                a1 = part_a_vertices[i]
                a2 = part_a_vertices[j]
                adj_bm[a1] |= (1 << a2)
                adj_bm[a2] |= (1 << a1)
                edges.append((a1, a2))
        candidates = []
        for i in range(len(part_b)):
            for j in range(i + 1, len(part_b)):
                candidates.append((part_b[i], part_b[j]))
        random.shuffle(candidates)
        return adj_bm, edges, candidates

    best_edges = []
    best_count = 0

    def update_best(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)

    # ---- Multi-start: greedy + polish + SA over diverse random part A ----
    sizes = [2, 3, 3, 3, 4, 4, 5, 5, 6]
    for sz in sizes:
        if time.time() > deadline - 25:
            break
        part_a = random.sample(range(n), sz)
        adj_bm, edges, candidates = start_graph(part_a)
        greedy_fill(adj_bm, edges, candidates, deadline - 20)
        polish(adj_bm, edges, deadline - 15)
        simulated_annealing(adj_bm, edges, deadline - 8)
        polish(adj_bm, edges, deadline - 4)
        update_best(edges)

    # ---- Final high-precision polish on the global best ----
    if best_edges and time.time() < deadline - 2:
        adj_bm = build_adj_from_edges(best_edges)
        polish(adj_bm, best_edges, deadline - 1)
        update_best(best_edges)

    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
