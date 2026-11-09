"""EV routing with time windows and battery; LNS-based structural approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    cust = list(range(1, n_customers + 1))
    all_nodes = list(range(0, N))

    def simulate_route(route):
        """Simulate a route; returns (ok, total_dist, total_time)."""
        if not route:
            return True, 0.0, 0.0
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0
        t = 0.0
        bat = float(battery_cap)
        d = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def simulate_route_with_stations(route, stations_to_insert):
        """Simulate route with optional station insertions.
        stations_to_insert: list of (position, station_id) pairs."""
        if not route:
            return True, 0.0, 0.0
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0

        # Build augmented route
        augmented = list(route)
        for pos, sid in sorted(stations_to_insert, key=lambda x: x[0]):
            augmented.insert(pos, sid)

        t = 0.0
        bat = float(battery_cap)
        d = 0.0
        prev = 0
        for v in augmented:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            if v in stations:
                # Recharge: time = (battery_cap - bat) * recharge_rate
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            else:
                t = max(t, ready[v])
                if t > due[v] + 1e-9:
                    return False, 0.0, 0.0
                t += service[v]
            prev = v
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def route_distance(route):
        """Compute distance of a valid route (no station handling)."""
        if not route:
            return 0.0
        d = 0.0
        prev = 0
        for v in route:
            d += dist[prev, v]
            prev = v
        d += dist[prev, 0]
        return d

    # ---------- Savings-based construction ----------
    def savings_construction():
        """Construct initial routes using Clarke-Wright savings heuristic."""
        # Start with each customer in its own route
        routes = [[c] for c in cust]
        # Savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings_list = []
        for i in range(len(cust)):
            for j in range(i+1, len(cust)):
                ci, cj = cust[i], cust[j]
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                savings_list.append((s, ci, cj))
        savings_list.sort(key=lambda x: -x[0])

        route_of = {}
        for c in cust:
            route_of[c] = [c]

        for s, ci, cj in savings_list:
            if s < 0:
                break
            ri = route_of.get(ci)
            rj = route_of.get(cj)
            if ri is None or rj is None:
                continue
            if ri is rj:
                continue
            # Check if ci and cj are at ends of their routes
            # Find position
            ci_at_end = (ri[-1] == ci)
            ci_at_start = (ri[0] == ci)
            cj_at_end = (rj[-1] == cj)
            cj_at_start = (rj[0] == cj)

            new_route = None
            if ci_at_end and cj_at_start:
                new_route = ri + rj
            elif ci_at_start and cj_at_end:
                new_route = rj + ri
            elif ci_at_end and cj_at_end:
                new_route = ri + rj[::-1]
            elif ci_at_start and cj_at_start:
                new_route = rj[::-1] + ri

            if new_route is None:
                continue

            # Check capacity
            if sum(demands[v] for v in new_route) > capacity:
                continue

            # Check feasibility
            ok, _, _ = simulate_route(new_route)
            if not ok:
                continue

            # Merge
            route_of[ci] = new_route
            route_of[cj] = new_route
            for v in new_route:
                route_of[v] = new_route

        # Build final list of routes
        seen = set()
        final_routes = []
        for c in cust:
            r = route_of[c]
            if id(r) not in seen:
                seen.add(id(r))
                final_routes.append(list(r))
        return final_routes

    # ---------- LNS (Large Neighborhood Search) ----------
    def lns_destroy(routes, n_destroy, rng):
        """Randomly remove n_destroy customers from routes, return (removed, remaining_routes)."""
        remaining = []
        removed = []
        for r in routes:
            new_r = list(r)
            for _ in range(min(n_destroy, len(new_r))):
                if not new_r:
                    break
                idx = rng.randint(0, len(new_r) - 1)
                removed.append(new_r.pop(idx))
            if new_r:
                remaining.append(new_r)
        return removed, remaining

    def greedy_insert(routes, customers, dist, demands, ready, due, service,
                      capacity, battery_cap, speed, recharge_rate, horizon,
                      simulate_fn, stations_list, rng):
        """Insert customers into routes greedily."""
        for v in customers:
            best_cost = float('inf')
            best_route_idx = None
            best_pos = None
            best_stations = []
            for ri, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    # Try without stations
                    new_r = r[:pos] + [v] + r[pos:]
                    ok, d, _ = simulate_fn(new_r)
                    if ok:
                        cost = d
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = ri
                            best_pos = pos
                            best_stations = []
                    # Try with one station insertion if needed
                    # Find positions to try inserting a station
                    # We'll try inserting station before and after v
                    for sid in stations_list:
                        for spos in [pos, pos + 1]:
                            if spos > len(r) + 1:
                                continue
                            new_r2 = r[:pos] + [v] + r[pos:]
                            # Insert station at spos in new_r2
                            if spos <= len(new_r2):
                                new_r3 = new_r2[:spos] + [sid] + new_r2[spos:]
                                ok3, d3, _ = simulate_fn(new_r3)
                                if ok3:
                                    # d3 already includes station distance
                                    if d3 < best_cost:
                                        best_cost = d3
                                        best_route_idx = ri
                                        best_pos = pos
                                        best_stations = [(spos, sid)]
            if best_route_idx is not None:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_pos] + [v] + r[best_pos:]
            else:
                # Start new route
                routes.append([v])
        return routes

    # ---------- Main solver ----------
    rng = random.Random(42)

    # Initial construction
    routes = savings_construction()
    if not routes:
        routes = [list(cust)]

    # Ensure all customers are in routes
    in_routes = set()
    for r in routes:
        for v in r:
            in_routes.add(v)
    missing = [c for c in cust if c not in in_routes]
    if missing:
        # Insert missing customers
        for v in missing:
            best_cost = float('inf')
            best_ri = None
            best_pos = None
            for ri, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [v] + r[pos:]
                    if sum(demands[x] for x in new_r) > capacity:
                        continue
                    ok, d, _ = simulate_route(new_r)
                    if ok and d < best_cost:
                        best_cost = d
                        best_ri = ri
                        best_pos = pos
            if best_ri is not None:
                r = routes[best_ri]
                routes[best_ri] = r[:best_pos] + [v] + r[best_pos:]
            else:
                routes.append([v])

    best_routes = [list(r) for r in routes]
    best_total = sum(route_distance(r) for r in routes)

    # LNS loop
    n_destroy = 3
    no_improve_count = 0

    while time.time() < deadline:
        # Destroy
        removed, remaining = lns_destroy(routes, n_destroy, rng)
        if not removed:
            break

        # Repair with greedy insertion
        new_routes = list(remaining)
        # Shuffle removed for diversity
        rng.shuffle(removed)
        new_routes = greedy_insert(new_routes, removed, dist, demands, ready, due,
                                   service, capacity, battery_cap, speed,
                                   recharge_rate, horizon, simulate_route,
                                   stations, rng)

        # Evaluate
        new_total = 0.0
        valid = True
        for r in new_routes:
            ok, d, _ = simulate_route(r)
            if not ok:
                valid = False
                break
            new_total += d
        if valid and new_total < best_total - 1e-9:
            best_total = new_total
            best_routes = [list(r) for r in new_routes]
            routes = [list(r) for r in new_routes]
            no_improve_count = 0
            # Occasionally adjust destroy size
            if rng.random() < 0.3:
                n_destroy = max(2, min(6, n_destroy + rng.choice([-1, 1])))
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                # Revert and change destroy size
                routes = [list(r) for r in best_routes]
                n_destroy = max(2, min(8, n_destroy + rng.choice([-1, 1, 2])))
                no_improve_count = 0

    # Final local search on best routes
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 2-opt
            for i in range(len(best_routes)):
                r = best_routes[i]
                n = len(r)
                if n < 2:
                    continue
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        ok, d, _ = simulate_route(new_r)
                        if ok:
                            old_d = route_distance(r)
                            if d < old_d - 1e-9:
                                best_routes[i] = new_r
                                improved = True
                                r = new_r
                                n = len(r)
                                break
                    if improved:
                        break
            # Relocate
            for i in range(len(best_routes)):
                ri = best_routes[i]
                for j in range(len(ri)):
                    v = ri[j]
                    src = ri[:j] + ri[j+1:]
                    ok_s, d_s, _ = simulate_route(src) if src else (True, 0.0, 0.0)
                    if not ok_s:
                        continue
                    for k in range(len(best_routes)):
                        if k == i:
                            continue
                        tgt = best_routes[k]
                        for pos in range(len(tgt) + 1):
                            new_tgt = tgt[:pos] + [v] + tgt[pos:]
                            ok_t, d_t, _ = simulate_route(new_tgt)
                            if not ok_t:
                                continue
                            old_total = d_s + route_distance(tgt)
                            if d_s + d_t < old_total - 1e-9:
                                best_routes[i] = src
                                best_routes[k] = new_tgt
                                if not best_routes[i]:
                                    best_routes.pop(i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # Swap
            for i in range(len(best_routes)):
                for k in range(i + 1, len(best_routes)):
                    ri = best_routes[i]
                    rk = best_routes[k]
                    for a in range(len(ri)):
                        for b in range(len(rk)):
                            new_ri = list(ri)
                            new_rk = list(rk)
                            new_ri[a], new_rk[b] = new_rk[b], new_ri[a]
                            ok_i, d_i, _ = simulate_route(new_ri)
                            ok_k, d_k, _ = simulate_route(new_rk)
                            if ok_i and ok_k:
                                if d_i + d_k < route_distance(ri) + route_distance(rk) - 1e-9:
                                    best_routes[i] = new_ri
                                    best_routes[k] = new_rk
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_routes
# EVOLVE-BLOCK-END
