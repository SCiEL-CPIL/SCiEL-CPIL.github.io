"""EV routing with time windows and battery; construction + local search with station handling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    cust = list(range(1, n_customers + 1))
    station_set = set(stations)

    def check_route_with_stations(route):
        """Check route feasibility, inserting stations as needed.
        Returns (feasible, total_distance, total_time, final_battery) or (False, 0, 0, 0)."""
        if not route:
            return True, 0.0, 0.0, float(battery)
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0, 0.0

        t = 0.0
        bat = float(battery)
        d_total = 0.0
        prev = 0

        for v in route:
            dd = dist[prev, v]
            # Check if we can reach v; if not, try inserting a station
            if bat < dd - 1e-9:
                # Try each station
                found_station = False
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_v = dist[st, v]
                    if bat < d_to_st - 1e-9:
                        continue
                    # Time to station
                    t_st = t + d_to_st / speed
                    # Recharge time: (battery - energy_used) * recharge_rate
                    # energy_used = d_to_st, so remaining = bat - d_to_st
                    recharge_time = (bat - d_to_st) * recharge_rate
                    t_after_recharge = t_st + recharge_time
                    # After recharge, battery is full
                    bat_after = float(battery)
                    # Now go to v
                    if bat_after < d_st_to_v - 1e-9:
                        continue
                    t_v = t_after_recharge + d_st_to_v / speed
                    bat_v = bat_after - d_st_to_v
                    # Check time window
                    t_arrive = max(t_v, ready[v])
                    if t_arrive > due[v] + 1e-9:
                        continue
                    # Feasible with this station
                    d_total += d_to_st + d_st_to_v
                    t = t_arrive + service[v]
                    bat = bat_v
                    prev = v
                    found_station = True
                    break
                if not found_station:
                    return False, 0.0, 0.0, 0.0
            else:
                bat -= dd
                t += dd / speed
                d_total += dd
                t = max(t, ready[v])
                if t > due[v] + 1e-9:
                    return False, 0.0, 0.0, 0.0
                t += service[v]
                prev = v

        # Return to depot
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            # Try stations for return
            found_station = False
            for st in stations:
                d_to_st = dist[prev, st]
                d_st_to_0 = dist[st, 0]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                recharge_time = (bat - d_to_st) * recharge_rate
                t_after_recharge = t_st + recharge_time
                bat_after = float(battery)
                if bat_after < d_st_to_0 - 1e-9:
                    continue
                t_final = t_after_recharge + d_st_to_0 / speed
                d_total += d_to_st + d_st_to_0
                if t_final > horizon + 1e-9:
                    continue
                found_station = True
                break
            if not found_station:
                return False, 0.0, 0.0, 0.0
        else:
            bat -= dd
            t += dd / speed
            d_total += dd
            if t > horizon + 1e-9:
                return False, 0.0, 0.0, 0.0

        return True, d_total, t, bat

    def fast_check_route(route):
        """Quick feasibility check without station insertion (for speed in local search)."""
        if not route:
            return True, 0.0
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0
        t = 0.0
        bat = float(battery)
        d_total = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0
            bat -= dd
            t += dd / speed
            d_total += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0
        bat -= dd
        t += dd / speed
        d_total += dd
        if t > horizon + 1e-9:
            return False, 0.0
        return True, d_total

    def savings_construct():
        """Clarke-Wright savings based construction."""
        routes = [[v] for v in cust]
        route_of = {v: i for i, v in enumerate(cust)}

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
                    savings.append((s, vi, vj))
        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Try merge options
            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_j[-1] == vj and route_i[0] == vi:
                options.append(route_j + route_i)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j[::-1])
            if route_j[0] == vj and route_i[-1] == vi:
                options.append(route_j + route_i[::-1])

            for candidate in options:
                if sum(demands[x] for x in candidate) > capacity:
                    continue
                f, _ = fast_check_route(candidate)
                if f:
                    routes[ri] = candidate
                    routes[rj] = []
                    for v in candidate:
                        route_of[v] = ri
                    # Clean up
                    non_empty = [r for r in routes if r]
                    for i, r in enumerate(non_empty):
                        for v in r:
                            route_of[v] = i
                    routes = non_empty
                    break

        return [r for r in routes if r]

    def greedy_construct():
        """Greedy nearest neighbor with time windows."""
        remaining = list(cust)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = float(battery)
            load = 0
            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1
                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue
                    t_after = t + d / speed
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue
                    time_slack = due[v] - t_after
                    score = d - 0.02 * time_slack
                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx
                if best_customer is None:
                    break
                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                route.append(v)
                remaining.pop(best_idx)
                prev = v
            if route:
                routes.append(route)
        return routes

    # Try multiple construction methods and pick best
    best_routes = None
    best_total = float('inf')

    for construct_fn in [savings_construct, greedy_construct]:
        if time.time() > deadline:
            break
        routes = construct_fn()
        # Validate with full check including stations
        total_d = 0.0
        valid = True
        for r in routes:
            f, d, _, _ = check_route_with_stations(r)
            if not f:
                valid = False
                break
            total_d += d
        if valid and total_d < best_total:
            best_total = total_d
            best_routes = [list(r) for r in routes]

    if best_routes is None:
        # Fallback: single customer per route
        best_routes = [[v] for v in cust]
        best_total = sum(dist[0, v] * 2 for v in cust)

    # Local search
    def total_dist(routes):
        s = 0.0
        for r in routes:
            f, d, _, _ = check_route_with_stations(r)
            if not f:
                return float('inf')
            s += d
        return s

    routes = [list(r) for r in best_routes]
    cur_total = total_dist(routes)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for j in range(len(routes[i])):
                v = routes[i][j]
                src = routes[i]
                src_without = src[:j] + src[j + 1:]
                best_gain = 1e-9
                best_target = None
                best_pos = None
                best_src_without = None
                best_new_tgt = None
                for k in range(len(routes)):
                    if k == i:
                        continue
                    tgt = routes[k]
                    for pos in range(len(tgt) + 1):
                        new_tgt = tgt[:pos] + [v] + tgt[pos:]
                        ok_t, d_t, _, _ = check_route_with_stations(new_tgt)
                        if not ok_t:
                            continue
                        ok_s, d_s, _, _ = check_route_with_stations(src_without)
                        if not ok_s:
                            continue
                        d0_t = check_route_with_stations(tgt)[1]
                        d0_s = check_route_with_stations(src)[1]
                        gain = (d_t + d_s) - (d0_t + d0_s)
                        if -gain > best_gain:
                            best_gain = -gain
                            best_target = k
                            best_pos = pos
                            best_src_without = src_without
                            best_new_tgt = new_tgt
                if best_target is not None:
                    routes[i] = best_src_without
                    routes[best_target] = best_new_tgt
                    if not routes[i]:
                        routes.pop(i)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Swap
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for k in range(i + 1, len(routes)):
                ri = routes[i]
                rk = routes[k]
                for a in range(len(ri)):
                    for b in range(len(rk)):
                        vi = ri[a]
                        vk = rk[b]
                        new_ri = list(ri)
                        new_rk = list(rk)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i, _, _ = check_route_with_stations(new_ri)
                        ok_k, d_k, _, _ = check_route_with_stations(new_rk)
                        if not (ok_i and ok_k):
                            continue
                        d0_i = check_route_with_stations(ri)[1]
                        d0_k = check_route_with_stations(rk)[1]
                        if d_i + d_k < d0_i + d0_k - 1e-9:
                            routes[i] = new_ri
                            routes[k] = new_rk
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt intra route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            n = len(ri)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    ok, d, _, _ = check_route_with_stations(new_ri)
                    if not ok:
                        continue
                    d0 = check_route_with_stations(ri)[1]
                    if d < d0 - 1e-9:
                        routes[i] = new_ri
                        ri = new_ri
                        n = len(ri)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if improved:
            continue

    final_total = total_dist(routes)
    if final_total < best_total:
        best_total = final_total
        best_routes = [list(r) for r in routes]

    return best_routes
# EVOLVE-BLOCK-END
