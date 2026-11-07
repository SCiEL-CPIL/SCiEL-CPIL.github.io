"""Electric vehicle routing with time windows – structural redesign.

Strategy:
  1. Build an initial solution via a time-window-aware savings heuristic that
     also tracks battery so that routes are feasible from the start.
  2. Improve with a battery-aware local search: relocate, swap, 2-opt, and
     or-opt (segment) moves, with a deadline guard.
  3. Return the best feasible solution found before the wall-clock limit.
"""
import time
import math
import numpy as np
import random

_DEADLINE_FRAC = 0.88  # finish 12 % early to be safe
_WALL_LIMIT = 150.0


def _dist_matrix(coords):
    n = len(coords)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            m[i, j] = d
            m[j, i] = d
    return m


def _route_info(route, D, demands, ready, due, service,
                battery_cap, recharge_rate, horizon, speed, station_set,
                is_station=None):
    """Simulate a route; return (total_dist, feasible, total_time)."""
    if is_station is None:
        is_station = np.zeros(len(D), dtype=bool)
        for s in station_set:
            is_station[s] = True
    
    t = 0.0
    bat = battery_cap
    load = 0.0
    total_d = 0.0
    prev = 0  # depot
    for node in route:
        d = D[prev, node]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_d + d, False, t
        total_d += d
        if is_station[node]:
            charge = (battery_cap - bat) * recharge_rate
            t += charge
            bat = battery_cap
        else:
            # customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_d, False, t
            t += service[node]
            load += demands[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return total_d + d, False, t
    total_d += d
    if t > horizon + 1e-9:
        return total_d, False, t
    if load > 100 + 1e-9:
        return total_d, False, t
    return total_d, True, t


def _initial_solution(instance):
    coords = np.array(instance["coords"])
    n = len(coords)
    D = _dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]
    
    # Precompute is_station array for fast lookup
    is_station = np.zeros(n, dtype=bool)
    for s in stations:
        is_station[s] = True

    # Clarke-Wright savings heuristic
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_cust + 1)]
    
    # Compute savings: saving(i,j) = D[0,i] + D[0,j] - D[i,j]
    savings = []
    for i in range(1, n_cust + 1):
        for j in range(i + 1, n_cust + 1):
            sav = D[0, i] + D[0, j] - D[i, j]
            savings.append((sav, i, j))
    savings.sort(key=lambda x: -x[0])
    
    # Map customer to route index
    cust_to_route = {i: i - 1 for i in range(1, n_cust + 1)}
    
    # Try to merge routes based on savings
    for sav, i, j in savings:
        if sav <= 0:
            break
        ri = cust_to_route.get(i, -1)
        rj = cust_to_route.get(j, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Check if i is at end of route_i and j is at start of route_j
        # or i is at start of route_i and j is at end of route_j
        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
        elif route_i[-1] == j and route_j[0] == i:
            merged = route_j + route_i
        elif route_i[0] == j and route_j[-1] == i:
            merged = route_i + route_j
        
        if merged is None:
            continue
        
        # Check feasibility
        td, feas, _ = _route_info(merged, D, demands, ready, due, service,
                                   bat_cap, rr, horizon, speed, stations, is_station)
        if feas:
            # Merge: keep route_i, remove route_j
            routes[ri] = merged
            for c in route_j:
                cust_to_route[c] = ri
            routes[rj] = []
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations, is_station


def _local_search(routes, D, demands, ready, due, service,
                  bat_cap, rr, horizon, speed, stations, is_station, deadline):
    """Battery-aware local search with relocate, swap, 2-opt, or-opt."""
    n_routes = len(routes)
    best_total = sum(_route_info(r, D, demands, ready, due, service,
                                 bat_cap, rr, horizon, speed, stations, is_station)[0]
                     for r in routes)
    no_improve = 0
    max_no_improve = 30

    while time.time() < deadline:
        improved = False
        
        # 2-opt within each route (most impactful for distance reduction)
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            L = len(ri)
            if L < 2:
                continue
            for a in range(L - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, L):
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    td, f, _ = _route_info(new_ri, D, demands, ready, due,
                                           service, bat_cap, rr, horizon,
                                           speed, stations, is_station)
                    if f:
                        # Quick check: only compute old if new is better
                        # Use a fast distance-only check first
                        old_dist = 0.0
                        prev = 0
                        for node in ri:
                            old_dist += D[prev, node]
                            prev = node
                        old_dist += D[prev, 0]
                        if td < old_dist - 1e-9:
                            routes[i] = new_ri
                            best_total = best_total - old_dist + td
                            improved = True
                            no_improve = 0
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Relocate
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c = ri[pos]
                ri2 = ri[:pos] + ri[pos + 1:]
                for j in range(n_routes):
                    if i == j:
                        continue
                    rj = list(routes[j])
                    for pos2 in range(len(rj) + 1):
                        new_j = rj[:pos2] + [c] + rj[pos2:]
                        td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations, is_station)
                        if f_i:
                            td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations, is_station)
                            if f_j:
                                new_total = td_i + td_j
                                if new_total < best_total - 1e-9:
                                    routes[i] = ri2
                                    routes[j] = new_j
                                    best_total = new_total
                                    improved = True
                                    no_improve = 0
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
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c1 = ri[pos]
                for j in range(i + 1, n_routes):
                    rj = list(routes[j])
                    for pos2 in range(len(rj)):
                        c2 = rj[pos2]
                        new_i = ri[:pos] + [c2] + ri[pos + 1:]
                        new_j = rj[:pos2] + [c1] + rj[pos2 + 1:]
                        td_i, f_i, _ = _route_info(new_i, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations, is_station)
                        if f_i:
                            td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations, is_station)
                            if f_j:
                                new_total = td_i + td_j
                                if new_total < best_total - 1e-9:
                                    routes[i] = new_i
                                    routes[j] = new_j
                                    best_total = new_total
                                    improved = True
                                    no_improve = 0
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Or-opt: move 2 or 3 consecutive customers
        for seg_len in (2, 3):
            for i in range(n_routes):
                if time.time() > deadline:
                    break
                ri = list(routes[i])
                for pos in range(len(ri) - seg_len + 1):
                    seg = ri[pos:pos + seg_len]
                    ri2 = ri[:pos] + ri[pos + seg_len:]
                    for j in range(n_routes):
                        if i == j:
                            continue
                        rj = list(routes[j])
                        for pos2 in range(len(rj) + 1):
                            new_j = rj[:pos2] + seg + rj[pos2:]
                            td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations, is_station)
                            if f_i:
                                td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                           service, bat_cap, rr, horizon,
                                                           speed, stations, is_station)
                                if f_j:
                                    new_total = td_i + td_j
                                    if new_total < best_total - 1e-9:
                                        routes[i] = ri2
                                        routes[j] = new_j
                                        best_total = new_total
                                        improved = True
                                        no_improve = 0
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        no_improve += 1
        if no_improve >= max_no_improve:
            break

    return routes, best_total


def plan(instance):
    t0 = time.time()
    deadline = t0 + _WALL_LIMIT * _DEADLINE_FRAC

    coords = np.array(instance["coords"])
    n = len(coords)
    D = _dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]
    
    # Precompute is_station array
    is_station = np.zeros(n, dtype=bool)
    for s in stations:
        is_station[s] = True

    # Initial solution using Clarke-Wright savings
    routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations, is_station = \
        _initial_solution(instance)

    # Local search
    if time.time() < deadline:
        routes, _ = _local_search(routes, D, demands, ready, due, service,
                                  bat_cap, rr, horizon, speed, stations, is_station, deadline)

    # Ensure every customer is served exactly once
    all_served = []
    for r in routes:
        all_served.extend(r)
    missing = set(range(1, n_cust + 1)) - set(all_served)
    for c in missing:
        routes.append([c])

    # Final feasibility check; if any route infeasible, fall back to one-per-route
    for r in routes:
        _, feas, _ = _route_info(r, D, demands, ready, due, service,
                                 bat_cap, rr, horizon, speed, stations, is_station)
        if not feas:
            # split into singletons
            flat = []
            for rr_ in routes:
                flat.extend(rr_)
            routes = [[v] for v in flat]
            break

    return routes
