import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))

    m = len(edges)

    def count_c4(u, v):
        # number of 4-cycles u-a-b-v where a~u, b~v, a~b, a!=b
        nu = adj[u]
        nv = adj[v]
        c = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    c += 1
        return c

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    def total_c4():
        e_list = list(edges)
        L = len(e_list)
        total = 0
        for i in range(L):
            a, b = e_list[i]
            for j in range(i+1, L):
                c, d = e_list[j]
                if len({a,b,c,d}) < 4:
                    continue
                if (c in adj[a] and d in adj[b]) or (d in adj[a] and c in adj[b]):
                    total += 1
        return total

    def edge_key(e):
        return (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])

    # ---- Adjust to exactly 91 edges using local-search-quality moves ----
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = None
            for (u, v) in list(edges):
                c = count_c4(u, v)
                if best_c is None or c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            remove_edge(best[0], best[1])
    elif m < 91:
        to_add = 91 - m
        edge_set = set(edges)
        for _ in range(to_add):
            best = None
            best_c = None
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    c = count_c4(u, v)
                    if best_c is None or c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(best)
            add_edge(best[0], best[1])

    # ---- Phase 1: Greedy steepest descent with exact delta ----
    edge_set = set(edge_key(e) for e in edges)
    
    def exact_swap_delta(ru, rv, au, av):
        """Compute exact change in total C4 count for removing (ru,rv) and adding (au,av)."""
        # Count C4s destroyed by removing (ru,rv)
        # A C4 uses edge (ru,rv) if it has the form ru-ru2-rv2-rv-ru or ru-rv2-ru2-rv-ru etc.
        # Actually, a C4 contains edge (ru,rv) when ru and rv are adjacent in the cycle.
        # C4s containing edge (ru,rv): paths of length 3 from ru to rv that don't use edge (ru,rv) directly
        # i.e., ru - x - y - rv where x~ru, y~rv, x~y, x!=rv, y!=ru, and the path doesn't use (ru,rv)
        # But we need to be careful: the C4 is ru-x-y-rv-ru, so we need x~ru, x~y, y~rv, and rv~ru (which is true)
        # The 4 vertices must be distinct: ru, x, y, rv all different
        
        c4s_using_ru_rv = 0
        for x in adj[ru]:
            if x == rv:
                continue
            for y in adj[rv]:
                if y == ru or y == x:
                    continue
                if y in adj[x]:
                    c4s_using_ru_rv += 1
        
        # Count C4s created by adding (au,av)
        c4s_created_au_av = 0
        for x in adj[au]:
            if x == av:
                continue
            for y in adj[av]:
                if y == au or y == x:
                    continue
                if y in adj[x]:
                    c4s_created_au_av += 1
        
        # But wait: some C4s might be both destroyed and created if they share structure.
        # Actually, removing (ru,rv) only affects C4s that USE edge (ru,rv).
        # Adding (au,av) only creates C4s that USE edge (au,av).
        # These are disjoint sets of C4s (since one uses (ru,rv) and the other uses (au,av)),
        # UNLESS (ru,rv) == (au,av) which can't happen in a valid swap.
        # So delta = c4s_created - c4s_destroyed
        return c4s_created_au_av - c4s_using_ru_rv

    # Greedy phase
    for it in range(50000):
        if time.time() > deadline - 60:
            break
        best_d = 0
        best_move = None
        
        # Precompute C4 counts for all edges and non-edges
        edge_c4 = {}
        for e in edges:
            ek = edge_key(e)
            edge_c4[ek] = count_c4(e[0], e[1])
        
        nonedge_c4 = {}
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k not in edge_set:
                    nonedge_c4[k] = count_c4(u, v)
        
        # Sort candidates
        rem_sorted = sorted(edge_c4.items(), key=lambda x: x[1])
        add_sorted = sorted(nonedge_c4.items(), key=lambda x: x[1])
        
        improved = False
        # Try top candidates
        for rk, rc in rem_sorted[:20]:
            if improved:
                break
            ru, rv = rk
            for ak, ac in add_sorted[:20]:
                au, av = ak
                # Skip if same edge
                if rk == ak:
                    continue
                d = exact_swap_delta(ru, rv, au, av)
                if d < best_d:
                    best_d = d
                    best_move = (rk, ak)
        
        if best_move is not None and best_d < 0:
            rk, ak = best_move
            ru, rv = rk
            au, av = ak
            remove_edge(ru, rv)
            edge_set.discard(rk)
            add_edge(au, av)
            edge_set.add(ak)
            edges.remove((ru, rv))
            edges.append((au, av))
            improved = True
        
        if not improved and it % 100 == 0:
            # Random perturbation
            if edges:
                re = edges[random.randint(0, len(edges)-1)]
                re_key = edge_key(re)
                # Pick random non-edge
                non_edges = [k for k in nonedge_c4.keys()]
                if non_edges:
                    ae = random.choice(non_edges)
                    remove_edge(re[0], re[1])
                    edge_set.discard(re_key)
                    add_edge(ae[0], ae[1])
                    edge_set.add(ae)
                    edges.remove(re)
                    edges.append(ae)

    # ---- Phase 2: Simulated Annealing with spectral-guided moves ----
    def spectral_score():
        """Compute a spectral proxy for C4 count using eigenvalues."""
        A = np.zeros((n, n), dtype=float)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        # trace(A^4) / 8 - corrections gives C4 count
        # But computing full eigenvalues is expensive. Use a cheaper proxy:
        # sum of squared eigenvalues of A relates to trace(A^2) = 2*|E|
        # For C4, we need trace(A^4). Let's compute it directly.
        A2 = A @ A
        A4 = A2 @ A2
        trace_A4 = np.trace(A4)
        # trace(A^4) = sum_{i,j} (A^2)_{ij}^2
        # This includes walks, not just cycles. 
        # Number of C4s = (trace(A^4) - 4*sum_v deg(v)^2 + 2*|E|) / 8
        # Actually: trace(A^4) counts all closed walks of length 4.
        # Closed walks of length 4:
        # - v->u->v->u->v: for each edge (u,v), contributes 2 (v->u->v->u->v and u->v->u->v->u)
        #   Actually for each edge, there are 2 such walks (starting from either endpoint)
        # - v->u->w->v->v? No, must be closed.
        # Standard formula: C4 = (trace(A^4) - 4*sum_v d_v^2 + 2*|E|) / 8
        deg = np.sum(A, axis=1)
        sum_d2 = np.sum(deg**2)
        c4_count = (trace_A4 - 4*sum_d2 + 2*len(edges)) / 8
        return int(round(c4_count))

    # SA phase
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    current_c4 = total_c4()
    best_c4 = current_c4
    best_edges = list(edges)
    best_adj = [set(s) for s in adj]
    
    for it in range(200000):
        if time.time() > deadline:
            break
        if it % 5000 == 0:
            T = max(T_min, T * cooling)
        
        # Random swap: remove random edge, add random non-edge
        e_idx = random.randint(0, len(edges)-1)
        ru, rv = edges[e_idx]
        re_key = edge_key((ru, rv))
        
        # Pick random non-edge
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            ae = (u, v) if u < v else (v, u)
            if ae not in edge_set:
                break
        
        # Compute delta
        d = exact_swap_delta(ru, rv, ae[0], ae[1])
        
        # Acceptance criterion
        if d < 0 or random.random() < np.exp(-d / T):
            # Apply swap
            remove_edge(ru, rv)
            edge_set.discard(re_key)
            add_edge(ae[0], ae[1])
            edge_set.add(ae)
            edges[e_idx] = ae
            
            current_c4 += d  # approximate; will recompute periodically
            
            if it % 1000 == 0:
                current_c4 = total_c4()
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
                best_adj = [set(s) for s in adj]
        
        # Periodic recompute to correct drift
        if it % 2000 == 0:
            current_c4 = total_c4()
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
                best_adj = [set(s) for s in adj]

    # Restore best solution
    adj = [set(s) for s in best_adj]
    edges = list(best_edges)
    edge_set = set(edge_key(e) for e in edges)

    # ---- Phase 3: Final polish with exact local search ----
    for it in range(50000):
        if time.time() > deadline:
            break
        
        edge_c4 = {}
        for e in edges:
            ek = edge_key(e)
            edge_c4[ek] = count_c4(e[0], e[1])
        
        nonedge_c4 = {}
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k not in edge_set:
                    nonedge_c4[k] = count_c4(u, v)
        
        rem_sorted = sorted(edge_c4.items(), key=lambda x: x[1])
        add_sorted = sorted(nonedge_c4.items(), key=lambda x: x[1])
        
        improved = False
        for rk, rc in rem_sorted[:10]:
            if improved:
                break
            ru, rv = rk
            for ak, ac in add_sorted[:10]:
                au, av = ak
                if rk == ak:
                    continue
                d = exact_swap_delta(ru, rv, au, av)
                if d < 0:
                    remove_edge(ru, rv)
                    edge_set.discard(rk)
                    add_edge(au, av)
                    edge_set.add(ak)
                    edges.remove((ru, rv))
                    edges.append((au, av))
                    improved = True
                    break
        
        if not improved:
            break

    return [[e[0], e[1]] for e in edges]
