"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    start = time.time()
    deadline = start + 145.0  # safety margin vs 150s wall-clock

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n = coords.shape[0]
    # Precompute Euclidean distances.
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))

    # ---------- Fast feasibility check ----------
    def simulate(route):
        """Return (ok, total_dist, end_time). Battery/time windows/capacity/horizon."""
        load = demands[route].sum() if route else 0.0
        if load > capacity + 1e-9:
            return False, 0.0, 0.0
        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                return False, 0.0, 0.0
            battery -= d
            t += d / speed
            total += d
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        d = dist[prev, 0]
        if d > battery + 1e-9:
            return False, 0.0, 0.0
        t += d / speed
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, total, t

    # ---------- Route distance helper ----------
    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    # ---------- Clarke-Wright savings construction ----------
    def savings_construct(seed):
        """Build routes using Clarke-Wright savings heuristic."""
        rng = random.Random(seed)
        
        # Start with each customer as its own route
        # routes[i] = list of customer ids in route i
        # parent[i] = route index for customer i
        # route_load[r] = total demand in route r
        route_of = {v: v for v in customers}  # customer -> route_id (initially self)
        routes = {v: [v] for v in customers}  # route_id -> list of customers
        route_load = {v: demands[v] for v in customers}
        
        # Compute savings: s(i,j) = dist[0,i] + dist[0,j] - dist[i,j]
        # Only consider pairs that could be merged
        savings_list = []
        for i in customers:
            for j in customers:
                if i >= j:
                    continue
                s = dist[0, i] + dist[0, j] - dist[i, j]
                if s > 1e-9:
                    savings_list.append((-s, i, j))  # negative for min-heap
        
        heapq.heapify(savings_list)
        
        # Union-find style tracking
        # For each route, track endpoints
        # A route is a sequence; we need to know which end is which
        # route_ends[r] = (first_customer, last_customer)
        route_ends = {v: (v, v) for v in customers}
        
        merged = set()
        
        while savings_list:
            if time.time() > deadline - 20.0:
                break
            neg_s, i, j = heapq.heappop(savings_list)
            s = -neg_s
            
            if (i, j) in merged or (j, i) in merged:
                continue
            
            ri = route_of[i]
            rj = route_of[j]
            
            if ri == rj:
                continue
            
            # Check capacity
            if route_load[ri] + route_load[rj] > capacity + 1e-9:
                continue
            
            # Check if i and j are at the ends of their respective routes
            ei, fi = route_ends[ri]
            ej, fj = route_ends[rj]
            
            # Can merge if i is an end of ri and j is an end of rj
            # or i is end of ri and j is end of rj in reverse
            valid = False
            new_route = None
            new_first = None
            new_last = None
            
            # Case 1: i is first of ri, j is last of rj -> merge ri + rj
            if i == ei and j == fj:
                valid = True
                new_route = routes[ri] + routes[rj]
                new_first = ei
                new_last = fj
            # Case 2: i is last of ri, j is first of rj -> merge ri + rj
            elif i == fi and j == ei:
                valid = True
                new_route = routes[ri] + routes[rj]
                new_first = ei
                new_last = fj
            # Case 3: i is first of ri, j is first of rj -> merge ri + reversed(rj)
            elif i == ei and j == ej:
                valid = True
                new_route = routes[ri] + routes[rj][::-1]
                new_first = ei
                new_last = fj
            # Case 4: i is last of ri, j is last of rj -> merge reversed(ri) + rj
            elif i == fi and j == fj:
                valid = True
                new_route = routes[ri][::-1] + routes[rj]
                new_first = ei
                new_last = fj
            
            if not valid:
                continue
            
            # Check feasibility of merged route
            ok, tot, _ = simulate(new_route)
            if not ok:
                continue
            
            # Perform merge
            merged.add((i, j))
            merged.add((j, i))
            
            # Update route_of for all customers in rj
            for v in routes[rj]:
                route_of[v] = ri
            
            routes[ri] = new_route
            route_load[ri] += route_load[rj]
            route_ends[ri] = (new_first, new_last)
            
            del routes[rj]
            del route_load[rj]
            del route_ends[rj]
        
        # Collect non-empty routes
        result = [routes[r] for r in routes if routes[r]]
        return result

    # ---------- 2-opt with fast distance delta ----------
    def two_opt(route, max_iter=3000):
        if len(route) < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for i in range(L - 1):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                for j in range(i + 1, L):
                    c = best[j]
                    d = best[j + 1] if j + 1 < L else 0
                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, tot, _ = simulate(cand)
                        if ok:
                            best = cand
                            improved = True
                            break
                if improved:
                    break
        return best

    # ---------- Or-opt: move a segment of length 1-3 ----------
    def or_opt(route, max_iter=1500):
        if len(route) < 3:
            return route
        best = list(route)
        best_dist = route_dist(best)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for seg_len in (1, 2, 3):
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, tot, _ = simulate(cand)
                        if ok and tot < best_dist - 1e-9:
                            best = cand
                            best_dist = tot
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Relocate: move single customer between routes ----------
    def relocate_opt(routes, max_iter=200):
        """Try moving single customers between routes."""
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1)):
                        v = route1[i]
                        if route_load_check(route2, v) > capacity + 1e-9:
                            continue
                        new_route1 = route1[:i] + route1[i+1:]
                        # Find best position in route2
                        best_pos = -1
                        best_cost = float('inf')
                        L2 = len(route2)
                        for pos in range(L2 + 1):
                            cand = route2[:pos] + [v] + route2[pos:]
                            ok, tot, _ = simulate(cand)
                            if ok:
                                if tot < best_cost:
                                    best_cost = tot
                                    best_pos = pos
                        if best_pos >= 0:
                            new_route2 = route2[:best_pos] + [v] + route2[best_pos:]
                            ok1, d1, _ = simulate(new_route1)
                            if ok1:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + best_cost
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def route_load_check(route, v):
        return sum(demands[x] for x in route) + demands[v]

    # ---------- Swap: exchange two customers between routes ----------
    def swap_opt(routes, max_iter=100):
        """Try swapping single customers between routes."""
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                for r2 in range(r1 + 1, n_routes):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            # Check capacity after swap
                            load1 = sum(demands[x] for x in route1) - demands[v1] + demands[v2]
                            load2 = sum(demands[x] for x in route2) - demands[v2] + demands[v1]
                            if load1 > capacity + 1e-9 or load2 > capacity + 1e-9:
                                continue
                            new_route1 = route1[:i] + [v2] + route1[i+1:]
                            new_route2 = route2[:j] + [v1] + route2[j+1:]
                            ok1, d1, _ = simulate(new_route1)
                            ok2, d2, _ = simulate(new_route2)
                            if ok1 and ok2:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + d2
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Total distance ----------
    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    best_routes = None
    best_score = float('inf')

    # Multiple restarts with savings construction
    n_restarts = 12
    for seed in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        routes = savings_construct(seed)
        
        # Local search: 2-opt then or-opt on each route
        for r in range(len(routes)):
            if time.time() > deadline - 10.0:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])
        
        # Inter-route optimization
        if time.time() < deadline - 20.0:
            routes = relocate_opt(routes)
        if time.time() < deadline - 15.0:
            routes = swap_opt(routes)
        
        # Second pass of intra-route optimization
        for r in range(len(routes)):
            if time.time() > deadline - 5.0:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])
        
        score = total_distance(routes)
        if score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

    # Fallback: ensure every customer served
    if best_routes is None:
        best_routes = [[v] for v in customers]
    else:
        served = set()
        for r in best_routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            best_routes.append([v])

    return best_routes
# EVOLVE-BLOCK-END
