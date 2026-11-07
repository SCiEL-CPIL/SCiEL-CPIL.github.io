# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq
from itertools import permutations


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin vs 150s wall-clock kill

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

    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def simulate_fast(route):
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
        s = simulate_fast(route)
        if s is not None:
            return list(route), s[0]
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
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None

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

    def greedy_construct(order):
        routes = []
        for c in order:
            placed = False
            best = None
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                best_pos = -1
                best_delta = float('inf')
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                if best_pos == -1:
                    continue
                cand = list(r[:best_pos]) + [c] + list(r[best_pos:])
                fixed, cd = best_station_fix(cand)
                if fixed is None:
                    continue
                if best is None or cd < best[2]:
                    best = (ri, fixed, cd)
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        return routes

    def two_opt(route, max_iter=120):
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
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - 1e-7:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - 1e-7:
                        route, cur_d = nr, sn[0]
                        improved = True
        return route

    def or_opt(route, max_iter=80):
        """Or-opt: move blocks of size 1-3 to a different position in the same route."""
        if len(route) < 3:
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
            for block_size in [1, 2, 3]:
                for i in range(n - block_size + 1):
                    block = route[i:i + block_size]
                    block_demand = sum(cust_demand[v] for v in block if is_customer[v])
                    for j in range(n - block_size + 1):
                        if j == i or j + block_size == i or j == i - 1:
                            continue
                        if j > i:
                            j_adj = j - block_size
                        else:
                            j_adj = j
                        new_route = route[:i] + route[i + block_size:]
                        insert_pos = j_adj
                        new_route = new_route[:insert_pos] + block + new_route[insert_pos:]
                        if route_demand(new_route) > capacity:
                            continue
                        sn = simulate_fast(new_route)
                        if sn is None:
                            continue
                        if sn[0] < cur_d - 1e-7:
                            route, cur_d = new_route, sn[0]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
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

    def dp_reinsertion(removed, routes, budget_end):
        """LNS: reinsert removed customers using a priority queue over partial assignments."""
        if not removed:
            return routes, total_dist(routes)

        n_removed = len(removed)
        if n_removed == 0:
            return routes, total_dist(routes)

        if n_removed <= 4:
            best_total = float('inf')
            best_routes = None

            base_routes = []
            for r in routes:
                base_routes.append([v for v in r if v not in set(removed)])
            base_routes = [r for r in base_routes if r]

            base_d = total_dist(base_routes)
            if base_d == float('inf'):
                return routes, total_dist(routes)

            for perm in permutations(removed):
                if time.time() > budget_end:
                    break
                cur_routes = [list(r) for r in base_routes]
                valid = True
                for c in perm:
                    placed = False
                    best_ri = -1
                    best_pos = -1
                    best_delta = float('inf')
                    for ri, r in enumerate(cur_routes):
                        if route_demand(r) + demands[c] > capacity:
                            continue
                        for pos in range(len(r) + 1):
                            prev = r[pos - 1] if pos > 0 else 0
                            nxt = r[pos] if pos < len(r) else 0
                            delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            if delta < best_delta:
                                best_delta = delta
                                best_ri = ri
                                best_pos = pos
                    if best_ri == -1:
                        cur_routes.append([c])
                        placed = True
                    else:
                        cur_routes[best_ri] = cur_routes[best_ri][:best_pos] + [c] + cur_routes[best_ri][best_pos:]
                        placed = True
                    if not placed:
                        valid = False
                        break

                if not valid:
                    continue

                for ri in range(len(cur_routes)):
                    fixed, cd = best_station_fix(cur_routes[ri])
                    if fixed is not None:
                        cur_routes[ri] = fixed
                    cur_routes[ri] = two_opt(cur_routes[ri], max_iter=30)

                td = total_dist(cur_routes)
                if td < best_total:
                    best_total = td
                    best_routes = [list(r) for r in cur_routes]

            if best_routes is not None:
                return best_routes, best_total
            return routes, total_dist(routes)
        else:
            base_routes = []
            for r in routes:
                base_routes.append([v for v in r if v not in set(removed)])
            base_routes = [r for r in base_routes if r]

            base_d = total_dist(base_routes)
            if base_d == float('inf'):
                return routes, total_dist(routes)

            sorted_removed = sorted(removed, key=lambda v: due[v])
            cur_routes = [list(r) for r in base_routes]

            for c in sorted_removed:
                if time.time() > budget_end:
                    break
                best_ri = -1
                best_pos = -1
                best_delta = float('inf')
                for ri, r in enumerate(cur_routes):
                    if route_demand(r) + demands[c] > capacity:
                        continue
                    for pos in range(len(r) + 1):
                        prev = r[pos - 1] if pos > 0 else 0
                        nxt = r[pos] if pos < len(r) else 0
                        delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_ri = ri
                            best_pos = pos
                if best_ri == -1:
                    cur_routes.append([c])
                else:
                    cur_routes[best_ri] = cur_routes[best_ri][:best_pos] + [c] + cur_routes[best_ri][best_pos:]

            for ri in range(len(cur_routes)):
                fixed, cd = best_station_fix(cur_routes[ri])
                if fixed is not None:
                    cur_routes[ri] = fixed
                cur_routes[ri] = two_opt(cur_routes[ri], max_iter=40)

            td = total_dist(cur_routes)
            if td < float('inf'):
                return cur_routes, td
            return routes, total_dist(routes)

    def lns(routes, budget_end, rng):
        """Large Neighborhood Search: remove a random subset and reinsert."""
        cur_d = total_dist(routes)
        if cur_d == float('inf'):
            return routes, cur_d

        n_to_remove = min(5, max(2, len(customers) // 8))

        all_custs = [c for r in routes for c in r if is_customer[c]]
        if len(all_custs) < n_to_remove:
            n_to_remove = max(1, len(all_custs) - 1)

        removed = rng.sample(all_custs, min(n_to_remove, len(all_custs)))
        removed_set = set(removed)

        new_routes, new_d = dp_reinsertion(removed, routes, budget_end)
        if new_d < cur_d - 1e-9:
            return new_routes, new_d
        return routes, cur_d

    def deep_search(routes, budget_end):
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        rng = random.Random(int(time.time() * 1000) % 100000)
        while time.time() < budget_end:
            improved = False
            nr = [two_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
            nr = [or_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
            nr, imp = relocate(routes, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
                    nr = [two_opt(r) for r in routes]
                    nd2 = total_dist(nr)
                    if nd2 < cur - 1e-7:
                        routes, cur = nr, nd2
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            nr, imp = swap(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if time.time() < budget_end - 5.0:
                nr, nd = lns(routes, budget_end - 3.0, rng)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    # Build diverse candidates
    candidates = []
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))
    order_dist = sorted(customers, key=lambda v: dist[0, v])
    candidates.append(greedy_construct(order_dist))
    rnd = random.Random(12345)
    for _ in range(4):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

    scored = []
    for cand in candidates:
        if cand is None:
            continue
        c2 = [two_opt(r) for r in cand]
        c2 = [or_opt(r) for r in c2]
        d = total_dist(c2)
        if d < float("inf"):
            scored.append((d, [list(r) for r in c2]))
    scored.sort(key=lambda x: x[0])

    if scored:
        global_best, global_d = scored[0][1], scored[0][0]
    else:
        global_best, global_d = None, float("inf")

    for d0, cand in scored:
        if time.time() > deadline - 12.0:
            break
        r, dd = deep_search(cand, deadline - 10.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd

    extra = random.Random(777)
    k = 0
    while time.time() < deadline - 18.0 and k < 60:
        o = customers[:]
        extra.shuffle(o)
        cand = greedy_construct(o)
        if cand is None:
            continue
        cand = [two_opt(r) for r in cand]
        cand = [or_opt(r) for r in cand]
        d = total_dist(cand)
        if d == float("inf"):
            k += 1
            continue
        r, dd = deep_search(cand, deadline - 12.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

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
        routes = [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END
