# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 145.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_station = np.zeros(n_nodes, dtype=bool)
    for c in customers:
        is_customer[c] = True
    for s in stations:
        is_station[s] = True

    # Precompute customer demand array indexed by node
    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def kmeans_cluster(points, k, max_iter=30, seed=42):
        """Simple k-means clustering. points: list of node indices. Returns list of lists of node indices."""
        rng = random.Random(seed)
        if k >= len(points):
            return [[p] for p in points]
        # Initialize centroids as k random points
        idxs = list(range(len(points)))
        rng.shuffle(idxs)
        centroids = [np.array(points[idxs[i]], dtype=float) for i in range(k)]
        
        for _ in range(max_iter):
            # Assign each point to nearest centroid
            assignments = [[] for _ in range(k)]
            for i, p in enumerate(points):
                pp = np.array(p, dtype=float)
                best_c = 0
                best_d = float('inf')
                for ci, c in enumerate(centroids):
                    d = np.sum((pp - c) ** 2)
                    if d < best_d:
                        best_d = d
                        best_c = ci
                assignments[best_c].append(p)
            
            # Update centroids
            new_centroids = []
            changed = False
            for ci in range(k):
                if assignments[ci]:
                    avg = np.mean([np.array(p, dtype=float) for p in assignments[ci]], axis=0)
                    if not np.allclose(avg, centroids[ci]):
                        changed = True
                    new_centroids.append(avg)
                else:
                    # Reinitialize empty cluster to a random point
                    new_centroids.append(np.array(points[rng.randint(0, len(points)-1)], dtype=float))
            centroids = new_centroids
            if not changed:
                break
        
        return assignments

    def simulate_fast(route):
        """Fast simulation with early exit. Returns (dist, time, battery) or None."""
        if not route:
            return (0.0, 0.0, battery_cap)
        
        t = 0.0
        bat = battery_cap
        cd = 0.0
        cur = 0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            cd += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        cd += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (cd, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate_fast(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def best_station_fix(route):
        """Insert cheapest single station if needed to make route feasible."""
        s = simulate_fast(route)
        if s is not None:
            return list(route), s[0]
        # Find where battery goes negative
        t = 0.0
        bat = battery_cap
        cur = 0
        neg_pos = -1
        for idx, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 and neg_pos == -1:
                neg_pos = idx
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        if neg_pos == -1:
            # Check return leg
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None  # Not a battery issue

        # Try inserting stations near where battery runs low
        best_route = None
        best_d = float('inf')
        search_start = max(0, neg_pos - 2)
        search_end = min(len(route) + 1, neg_pos + 3)
        for p in range(search_start, search_end + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        # Broader search if local didn't work
        for p in range(len(route) + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order, cluster_hint=None):
        routes = []
        # If cluster_hint is provided, pre-create routes for each cluster
        if cluster_hint is not None:
            for cluster in cluster_hint:
                if cluster:
                    routes.append([])
            # Assign each customer to its cluster's route
            for c in order:
                placed = False
                # Find which cluster this customer belongs to
                for ri, cluster in enumerate(cluster_hint):
                    if c in cluster:
                        r = routes[ri]
                        if route_demand(r) + demands[c] > capacity:
                            # Cluster route full, try other routes
                            break
                        # Find best position in this route
                        positions = []
                        for pos in range(len(r) + 1):
                            prev = r[pos - 1] if pos > 0 else 0
                            nxt = r[pos] if pos < len(r) else 0
                            delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            positions.append((delta, pos))
                        positions.sort()
                        tried = 0
                        for delta, pos in positions:
                            if tried >= 5:
                                break
                            tried += 1
                            cand = list(r[:pos]) + [c] + list(r[pos:])
                            fixed, cd = best_station_fix(cand)
                            if fixed is not None:
                                routes[ri] = fixed
                                placed = True
                                break
                        if placed:
                            break
                if not placed:
                    # Try all routes
                    best = None
                    best_cd = float('inf')
                    for ri, r in enumerate(routes):
                        if route_demand(r) + demands[c] > capacity:
                            continue
                        positions = []
                        for pos in range(len(r) + 1):
                            prev = r[pos - 1] if pos > 0 else 0
                            nxt = r[pos] if pos < len(r) else 0
                            delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            positions.append((delta, pos))
                        positions.sort()
                        tried = 0
                        for delta, pos in positions:
                            if tried >= 5:
                                break
                            tried += 1
                            cand = list(r[:pos]) + [c] + list(r[pos:])
                            fixed, cd = best_station_fix(cand)
                            if fixed is not None:
                                if cd < best_cd:
                                    best = (ri, fixed, cd)
                                    best_cd = cd
                                break
                    if best is not None:
                        routes[best[0]] = best[1]
                        placed = True
                if not placed:
                    fixed, cd = best_station_fix([c])
                    if fixed is None:
                        fixed = [c]
                    routes.append(fixed)
            return [r for r in routes if r]
        else:
            # Original greedy construction without cluster hint
            for c in order:
                placed = False
                best = None
                best_cd = float('inf')
                for ri, r in enumerate(routes):
                    if route_demand(r) + demands[c] > capacity:
                        continue
                    # Find all feasible positions sorted by distance delta
                    positions = []
                    for pos in range(len(r) + 1):
                        prev = r[pos - 1] if pos > 0 else 0
                        nxt = r[pos] if pos < len(r) else 0
                        delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                        positions.append((delta, pos))
                    positions.sort()
                    
                    # Try positions in order of increasing distance delta,
                    # but only simulate the top few to save time
                    tried = 0
                    for delta, pos in positions:
                        if tried >= 5:
                            break
                        tried += 1
                        cand = list(r[:pos]) + [c] + list(r[pos:])
                        fixed, cd = best_station_fix(cand)
                        if fixed is not None:
                            if cd < best_cd:
                                best = (ri, fixed, cd)
                                best_cd = cd
                            break  # First feasible position is good enough
                if best is not None:
                    routes[best[0]] = best[1]
                    placed = True
                if not placed:
                    fixed, cd = best_station_fix([c])
                    if fixed is None:
                        fixed = [c]
                    routes.append(fixed)
            return routes

    def two_opt(route, max_iter=200, tol=1e-7):
        if len(route) < 2:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Quick distance delta check
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - tol:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - tol:
                        route, cur_d = nr, sn[0]
                        improved = True
            # 3-opt: try reversing segment [i..j] and swapping with [j+1..k]
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        a = route[i - 1] if i > 0 else 0
                        b = route[i]
                        c = route[j]
                        d = route[j + 1]
                        e = route[k]
                        f = route[k + 1] if k + 1 < n else 0
                        old_len = dist[a, b] + dist[c, d] + dist[e, f]
                        new_len = dist[a, e] + dist[c, b] + dist[d, f]
                        if new_len >= old_len - tol:
                            continue
                        nr = (list(route[:i]) + list(route[j + 1:k + 1]) +
                              list(reversed(route[i:j + 1])) + list(route[k + 1:]))
                        if route_demand(nr) > capacity:
                            continue
                        sn = simulate_fast(nr)
                        if sn is None:
                            continue
                        if sn[0] < cur_d - tol:
                            route, cur_d = nr, sn[0]
                            improved = True
            # Or-opt: relocate blocks of length 1, 2, and 3 within the same route
            for blocklen in [1, 2, 3]:
                n = len(route)
                if n < blocklen + 1:
                    continue
                for _ in range(30):
                    improved = False
                    for i in range(n - blocklen + 1):
                        block = route[i:i + blocklen]
                        rest = route[:i] + route[i + blocklen:]
                        for j in range(len(rest) + 1):
                            if j <= i:
                                continue
                            cand = rest[:j] + block + rest[j:]
                            if cand == route:
                                continue
                            sn = simulate_fast(cand)
                            if sn is None:
                                continue
                            if sn[0] < cur_d - tol:
                                route, cur_d = cand, sn[0]
                                improved = True
                    if not improved:
                        break
        return route

    def relocate(routes, blocklen=1):
        nR = len(routes)
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            ri_full = routes[i]
            old_di = oldD[i]
            for pi in range(len(ri_full) - blocklen + 1):
                block = ri_full[pi:pi + blocklen]
                block_demand = 0.0
                for v in block:
                    if is_customer[v]:
                        block_demand += cust_demand[v]
                rest_i = ri_full[:pi] + ri_full[pi + blocklen:]
                sim_i = simulate_fast(rest_i) if rest_i else (0.0, 0.0, battery_cap)
                if sim_i is None:
                    continue
                di = sim_i[0] if rest_i else 0.0
                for j in range(nR):
                    if i == j:
                        continue
                    old_dj = oldD[j]
                    if route_demand(routes[j]) + block_demand > capacity:
                        continue
                    rj_base = routes[j]
                    # Find best insertion position by delta
                    best_pj = -1
                    best_delta = float('inf')
                    for pj in range(len(rj_base) + 1):
                        prev = rj_base[pj - 1] if pj > 0 else 0
                        nxt = rj_base[pj] if pj < len(rj_base) else 0
                        b0 = block[0]
                        b1 = block[-1]
                        delta = dist[prev, b0] + dist[b1, nxt] - dist[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pj = pj
                    if best_pj == -1:
                        continue
                    rj = rj_base[:best_pj] + list(block) + rj_base[best_pj:]
                    sim_j = simulate_fast(rj)
                    if sim_j is None:
                        continue
                    if di + sim_j[0] < old_di + old_dj - 1e-7:
                        routes[i] = rest_i
                        routes[j] = rj
                        routes = [r for r in routes if r]
                        return routes, True
        # Also try 2-opt swaps between routes
        for i in range(nR):
            for j in range(i + 1, nR):
                for pi in range(len(routes[i])):
                    ci = routes[i][pi]
                    if not is_customer[ci]:
                        continue
                    for pj in range(len(routes[j])):
                        cj = routes[j][pj]
                        if not is_customer[cj]:
                            continue
                        if route_demand(routes[i]) - demands[ci] + demands[cj] > capacity:
                            continue
                        if route_demand(routes[j]) - demands[cj] + demands[ci] > capacity:
                            continue
                        ni = list(routes[i])
                        nj = list(routes[j])
                        ni[pi] = cj
                        nj[pj] = ci
                        si = simulate_fast(ni)
                        sj = simulate_fast(nj)
                        if si is None or sj is None:
                            continue
                        if si[0] + sj[0] < oldD[i] + oldD[j] - 1e-7:
                            routes[i] = ni
                            routes[j] = nj
                            return routes, True
        return routes, False

    def swap(routes):
        nR = len(routes)
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            old_di = oldD[i]
            for j in range(i + 1, nR):
                old_dj = oldD[j]
                for pi in range(len(routes[i])):
                    ci = routes[i][pi]
                    if not is_customer[ci]:
                        continue
                    for pj in range(len(routes[j])):
                        cj = routes[j][pj]
                        if not is_customer[cj]:
                            continue
                        if route_demand(routes[i]) - demands[ci] + demands[cj] > capacity:
                            continue
                        if route_demand(routes[j]) - demands[cj] + demands[ci] > capacity:
                            continue
                        ni = list(routes[i])
                        nj = list(routes[j])
                        ni[pi] = cj
                        nj[pj] = ci
                        si = simulate_fast(ni)
                        sj = simulate_fast(nj)
                        if si is None or sj is None:
                            continue
                        if si[0] + sj[0] < old_di + old_dj - 1e-7:
                            routes[i] = ni
                            routes[j] = nj
                            return routes, True
        return routes, False

    def deep_search(routes, budget_end):
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        while time.time() < budget_end:
            improved = False
            # 2-opt + 3-opt + Or-opt on each route
            nr = [two_opt(r, max_iter=250) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
            # Relocate blocks of size 1
            nr, imp = relocate(routes, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
                    # Polish with 2-opt after relocation
                    nr2 = [two_opt(r, max_iter=200) for r in routes]
                    nd2 = total_dist(nr2)
                    if nd2 < cur - 1e-7:
                        routes, cur = nr2, nd2
            # Relocate blocks of size 2
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            # Relocate blocks of size 3
            nr, imp = relocate(routes, 3)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            # Cross-route swap
            nr, imp = swap(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    # Build diverse candidates
    candidates = []
    
    # Cluster-first approach: try different k values for k-means
    for k in [3, 4, 5, 6]:
        clusters = kmeans_cluster(customers, k, seed=k)
        # Sort customers within each cluster by distance from depot
        cluster_order = []
        for cluster in clusters:
            cluster_sorted = sorted(cluster, key=lambda v: dist[0, v])
            cluster_order.extend(cluster_sorted)
        candidates.append(greedy_construct(cluster_order, clusters))
    
    # Original diverse orders
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))
    order_dist = sorted(customers, key=lambda v: dist[0, v])
    candidates.append(greedy_construct(order_dist))
    # Nearest-neighbor construction
    nn_order = []
    remaining = set(customers)
    cur = 0
    while remaining:
        best_c = min(remaining, key=lambda v: dist[cur, v])
        nn_order.append(best_c)
        remaining.remove(best_c)
        cur = best_c
    candidates.append(greedy_construct(nn_order))
    rnd = random.Random(12345)
    for _ in range(5):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))
    # Add a few more random orders for diversity
    for _ in range(3):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

    scored = []
    for cand in candidates:
        if cand is None:
            continue
        c2 = [two_opt(r) for r in cand]
        d = total_dist(c2)
        if d < float("inf"):
            scored.append((d, [list(r) for r in c2]))
    scored.sort(key=lambda x: x[0])

    if scored:
        global_best, global_d = scored[0][1], scored[0][0]
    else:
        global_best, global_d = None, float("inf")

    # Deep search on top candidates
    for d0, cand in scored:
        if time.time() > deadline - 25.0:
            break
        r, dd = deep_search(cand, deadline - 22.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
    
    # Final high-precision polish pass with tighter tolerance
    if global_best is not None and time.time() < deadline - 12.0:
        polish_end = deadline - 10.0
        cur_polish = total_dist(global_best)
        improved_polish = True
        while improved_polish and time.time() < polish_end:
            improved_polish = False
            # Tighter 2-opt with more iterations
            nr = [two_opt(r, max_iter=300, tol=1e-9) for r in global_best]
            nd = total_dist(nr)
            if nd < cur_polish - 1e-9:
                global_best, cur_polish = nr, nd
                improved_polish = True
            # Additional relocation passes
            nr, imp = relocate(global_best, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur_polish - 1e-9:
                    global_best, cur_polish = nr, nd
                    improved_polish = True
                    # Polish again
                    nr2 = [two_opt(r, max_iter=200, tol=1e-9) for r in global_best]
                    nd2 = total_dist(nr2)
                    if nd2 < cur_polish - 1e-9:
                        global_best, cur_polish = nr2, nd2
                        improved_polish = True
            nr, imp = relocate(global_best, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur_polish - 1e-9:
                    global_best, cur_polish = nr, nd
                    improved_polish = True
            nr, imp = swap(global_best)
            if imp:
                nd = total_dist(nr)
                if nd < cur_polish - 1e-9:
                    global_best, cur_polish = nr, nd
                    improved_polish = True

    # Additional random restarts with deep search
    extra = random.Random(777)
    k = 0
    while time.time() < deadline - 25.0 and k < 60:
        # Mix random, perturbed, and cluster-based orders for better diversity
        if k % 4 == 0:
            o = customers[:]
            extra.shuffle(o)
        elif k % 4 == 1:
            # Perturb the best route order
            base = sorted(customers, key=lambda v: dist[0, v])
            o = base[:]
            for _ in range(extra.randint(2, 5)):
                i, j = extra.sample(range(len(o)), 2)
                o[i], o[j] = o[j], o[i]
        elif k % 4 == 2:
            # Cluster-based order with random k
            kk = extra.randint(3, 6)
            clusters = kmeans_cluster(customers, kk, seed=extra.randint(0, 1000))
            o = []
            for cluster in clusters:
                cluster_sorted = sorted(cluster, key=lambda v: dist[0, v])
                o.extend(cluster_sorted)
        else:
            # Cluster-based order: sort by angle in clusters
            o = customers[:]
            extra.shuffle(o)
            # Re-sort small chunks by distance
            for i in range(0, len(o) - 2, 3):
                chunk = o[i:i+3]
                chunk.sort(key=lambda v: dist[0, v])
                o[i:i+3] = chunk
        cand = greedy_construct(o)
        if cand is None:
            continue
        cand = [two_opt(r) for r in cand]
        d = total_dist(cand)
        if d == float("inf"):
            k += 1
            continue
        r, dd = deep_search(cand, deadline - 20.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

    # Extra final polish with very tight tolerance if time permits
    if global_best is not None and time.time() < deadline - 5.0:
        final_polish_end = deadline - 3.0
        cur_fp = total_dist(global_best)
        improved_fp = True
        while improved_fp and time.time() < final_polish_end:
            improved_fp = False
            # Very tight 2-opt + 3-opt + Or-opt
            nr = [two_opt(r, max_iter=500, tol=1e-10) for r in global_best]
            nd = total_dist(nr)
            if nd < cur_fp - 1e-10:
                global_best, cur_fp = nr, nd
                improved_fp = True
            # Final relocation
            nr, imp = relocate(global_best, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur_fp - 1e-10:
                    global_best, cur_fp = nr, nd
                    improved_fp = True
                    nr2 = [two_opt(r, max_iter=400, tol=1e-10) for r in global_best]
                    nd2 = total_dist(nr2)
                    if nd2 < cur_fp - 1e-10:
                        global_best, cur_fp = nr2, nd2
                        improved_fp = True
            # Relocate blocks of size 2
            nr, imp = relocate(global_best, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur_fp - 1e-10:
                    global_best, cur_fp = nr, nd
                    improved_fp = True
            # Cross-route swap
            nr, imp = swap(global_best)
            if imp:
                nd = total_dist(nr)
                if nd < cur_fp - 1e-10:
                    global_best, cur_fp = nr, nd
                    improved_fp = True

    routes = global_best

    def all_feasible(rs):
        served = []
        for r in rs:
            s = simulate_fast(r)
            if s is None:
                return False
            for v in r:
                if is_customer[v]:
                    served.append(v)
        return len(served) == len(customers) and len(set(served)) == len(customers)

    if routes is None or not all_feasible(routes):
        # Fallback: one customer per route
        routes = [[c] for c in customers]
        # Try to merge some routes if feasible
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if route_demand(routes[i]) + route_demand(routes[j]) > capacity:
                    continue
                cand = routes[i] + routes[j]
                fixed, cd = best_station_fix(cand)
                if fixed is not None:
                    routes[i] = fixed
                    routes.pop(j)
                    break

    return routes
# EVOLVE-BLOCK-END
