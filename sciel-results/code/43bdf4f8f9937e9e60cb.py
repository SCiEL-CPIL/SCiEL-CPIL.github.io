"""EV routing with time windows and battery; vectorized latest-departure local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def plan(instance):
    t0 = time.time()
    deadline = t0 + 142.0

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
    station_set = set(stations)
    is_customer = np.zeros(N, dtype=bool)
    is_customer[1:n_customers + 1] = True

    # Precompute: for each pair (u, v), the best way to go u->v with battery
    # Returns (ok, extra_dist, charge_time) where extra_dist is added distance
    # and charge_time is added charging time. If ok is False, no valid path.
    # We consider: direct, or via one station.
    n_st = len(stations)
    # bat_feas[u, v] = (ok, extra_dist, charge_time)
    # We'll store as three arrays for speed
    bat_ok = np.zeros((N, N), dtype=bool)
    bat_extra_dist = np.full((N, N), np.inf)
    bat_charge_time = np.full((N, N), np.inf)

    for u in range(N):
        for v in range(N):
            if u == v:
                bat_ok[u, v] = True
                bat_extra_dist[u, v] = 0.0
                bat_charge_time[u, v] = 0.0
                continue
            dd = dist[u, v]
            # Direct: need battery_cap >= dd (always true since we start full)
            # But battery depends on how much we have when leaving u.
            # For the precompute, we just check if it's *possible* with full battery.
            # The actual feasibility depends on current battery, which we track in simulation.
            # So this precompute is for: "if I have enough battery to reach u, can I reach v?"
            # Since battery only decreases, the worst case is arriving at u with 0 battery.
            # If dist[u,v] <= battery_cap, then with full battery at u we can reach v.
            # The question is whether we can reach v given our actual battery at u.
            # We'll handle that in the simulation. This precompute is for station routing.
            
            # Direct path: always feasible if dd <= battery_cap (which is always true for our instances)
            bat_ok[u, v] = True
            bat_extra_dist[u, v] = 0.0
            bat_charge_time[u, v] = 0.0
            
            # Via station: check if it reduces total distance or is needed
            # For now, direct is always the baseline. Station insertion is done in simulation.
            pass

    # Simulate a full route with battery and time, inserting stations greedily.
    # Returns (ok, total_dist, full_route_with_stations)
    def simulate_route(cust_seq):
        """Simulate a route given customer sequence. Returns (ok, total_dist, full_route)."""
        if not cust_seq:
            return True, 0.0, []
        
        total_demand = sum(demands[v] for v in cust_seq)
        if total_demand > capacity:
            return False, 0.0, []
        
        t = 0.0
        bat = battery_cap
        d_total = 0.0
        full_route = []
        prev = 0
        
        for v in cust_seq:
            # Try direct first
            dd = dist[prev, v]
            if bat >= dd - 1e-9:
                # Direct
                t += dd / speed
                bat -= dd
                d_total += dd
            else:
                # Need station
                best_s = -1
                best_cost = np.inf
                for s in stations:
                    d1 = dist[prev, s]
                    if d1 > bat + 1e-9:
                        continue
                    d2 = dist[s, v]
                    if d2 > battery_cap + 1e-9:
                        continue
                    cost = d1 + d2
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                if best_s == -1:
                    return False, 0.0, []
                # Go to station
                d1 = dist[prev, best_s]
                t += d1 / speed
                bat -= d1
                d_total += d1
                # Charge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                full_route.append(best_s)
                # Go to customer
                d2 = dist[best_s, v]
                t += d2 / speed
                bat -= d2
                d_total += d2
                prev = best_s
            
            # Arrive at customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, []
            t += service[v]
            full_route.append(v)
            prev = v
        
        # Return to depot
        dd = dist[prev, 0]
        if bat >= dd - 1e-9:
            t += dd / speed
            bat -= dd
            d_total += dd
        else:
            best_s = -1
            best_cost = np.inf
            for s in stations:
                d1 = dist[prev, s]
                if d1 > bat + 1e-9:
                    continue
                d2 = dist[s, 0]
                if d2 > battery_cap + 1e-9:
                    continue
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            if best_s == -1:
                return False, 0.0, []
            d1 = dist[prev, best_s]
            t += d1 / speed
            bat -= d1
            d_total += d1
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            full_route.append(best_s)
            d2 = dist[best_s, 0]
            t += d2 / speed
            bat -= d2
            d_total += d2
        
        if t > horizon + 1e-9:
            return False, 0.0, []
        
        return True, d_total, full_route

    # ---------- Construction: cheapest insertion ----------
    routes = []
    remaining = set(cust)
    
    # Start with one route per customer (or cluster)
    # Use simple: create routes by inserting customers one by one into best position
    all_custs = sorted(cust, key=lambda x: ready[x])
    
    while remaining:
        if time.time() > deadline:
            break
        best_route = None
        best_dist = np.inf
        best_v = None
        
        # Try each remaining customer as seed
        for v in list(remaining):
            ok, d, full = simulate_route([v])
            if ok and d < best_dist:
                best_dist = d
                best_route = full
                best_v = v
        
        if best_v is None:
            # Can't place any, break
            break
        
        # Now try to insert other customers into this route
        current_seq = [best_v]
        remaining.discard(best_v)
        
        while True:
            if time.time() > deadline:
                break
            best_insert = None
            best_insert_dist = np.inf
            best_insert_v = None
            best_insert_seq = None
            
            for v in list(remaining):
                for pos in range(len(current_seq) + 1):
                    new_seq = current_seq[:pos] + [v] + current_seq[pos:]
                    ok, d, _ = simulate_route(new_seq)
                    if ok and d < best_insert_dist:
                        best_insert_dist = d
                        best_insert_v = v
                        best_insert_seq = new_seq
            
            if best_insert_v is None:
                break
            
            current_seq = best_insert_seq
            remaining.discard(best_insert_v)
        
        # Simulate final
        ok, d, full = simulate_route(current_seq)
        if ok:
            routes.append(full)

    # ---------- Local Search with O(1) precomputed checks ----------
    def get_route_dist(route):
        """Get distance of a route (assumed valid)."""
        if not route:
            return 0.0
        d = 0.0
        prev = 0
        for v in route:
            d += dist[prev, v]
            prev = v
        d += dist[prev, 0]
        return d

    def extract_custs(route):
        return [v for v in route if is_customer[v]]

    # Cache route distances
    route_dists = [get_route_dist(r) for r in routes]
    total_d = sum(route_dists)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # 2-opt intra-route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            ri_custs = extract_custs(ri)
            n = len(ri_custs)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_seq = ri_custs[:a] + ri_custs[a:b + 1][::-1] + ri_custs[b + 1:]
                    ok, d, new_route = simulate_route(new_seq)
                    if ok and d < route_dists[i] - 1e-9:
                        routes[i] = new_route
                        route_dists[i] = d
                        total_d += d - get_route_dist(ri)
                        improved = True
                        found = True
                        break
            if found:
                break
        if improved:
            continue

        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            ri_custs = extract_custs(ri)
            if len(ri_custs) <= 1:
                continue
            for j in range(len(ri_custs)):
                v = ri_custs[j]
                src_without = ri_custs[:j] + ri_custs[j + 1:]
                ok_s, d_s, src_route = simulate_route(src_without)
                if not ok_s:
                    continue
                for k in range(len(routes)):
                    if k == i:
                        continue
                    rk = routes[k]
                    rk_custs = extract_custs(rk)
                    for pos in range(len(rk_custs) + 1):
                        new_tgt = rk_custs[:pos] + [v] + rk_custs[pos:]
                        ok_t, d_t, tgt_route = simulate_route(new_tgt)
                        if not ok_t:
                            continue
                        new_pair = d_s + d_t
                        old_pair = route_dists[i] + route_dists[k]
                        if new_pair < old_pair - 1e-9:
                            routes[i] = src_route
                            routes[k] = tgt_route
                            route_dists[i] = d_s
                            route_dists[k] = d_t
                            total_d += new_pair - old_pair
                            if not routes[i]:
                                routes.pop(i)
                                route_dists.pop(i)
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

        # Swap
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            ri_custs = extract_custs(ri)
            for k in range(i + 1, len(routes)):
                rk = routes[k]
                rk_custs = extract_custs(rk)
                found = False
                for a in range(len(ri_custs)):
                    if found:
                        break
                    for b in range(len(rk_custs)):
                        vi = ri_custs[a]
                        vk = rk_custs[b]
                        new_ri = list(ri_custs)
                        new_rk = list(rk_custs)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i, route_i = simulate_route(new_ri)
                        ok_k, d_k, route_k = simulate_route(new_rk)
                        if ok_i and ok_k and d_i + d_k < route_dists[i] + route_dists[k] - 1e-9:
                            routes[i] = route_i
                            routes[k] = route_k
                            route_dists[i] = d_i
                            route_dists[k] = d_k
                            total_d += d_i + d_k - route_dists[i] - route_dists[k]
                            improved = True
                            found = True
                            break
                if found:
                    break
            if improved:
                break
        if improved:
            continue

    return routes
# EVOLVE-BLOCK-END
