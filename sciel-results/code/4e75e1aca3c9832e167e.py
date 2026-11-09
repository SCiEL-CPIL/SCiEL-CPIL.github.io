"""Rewritten plan for EV routing with time windows and charging."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _euclid(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dist = math.sqrt(dx * dx + dy * dy)
            d[i, j] = dist
            d[j, i] = dist
    return d


def _route_time_and_battery(route, dist, coords, demands, ready, due, service,
                            stations, battery_max, recharge_rate, speed, horizon):
    """Return (total_distance, feasible) for a route (list of node ids, no depot at ends)."""
    if not route:
        return 0.0, True
    t = 0.0
    bat = battery_max
    total_dist = 0.0
    prev = 0
    for node in route:
        d = dist[prev][node]
        total_dist += d
        bat -= d
        t += d / speed
        if bat < -1e-9:
            return total_dist, False
        if node in stations:
            # recharge: time added = (battery - energy) * recharge_rate? 
            # The spec says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This is a bit ambiguous. Let me re-read.
            # "a station adds (battery - energy) * recharge_rate time and refills; stations may repeat;"
            # I think this means: at a station, you spend time to recharge. The time is proportional to how much energy you need to recharge.
            # Actually, "battery - energy" doesn't make sense if battery is current charge.
            # Let me interpret: the time to recharge is (battery_max - bat) * recharge_rate, i.e., you recharge to full, and the time is (amount needed) * rate.
            # But the spec says "(battery - energy) * recharge_rate". Maybe "energy" is the amount you need?
            # Let me assume: time_added = (battery_max - bat) * recharge_rate, and bat becomes battery_max.
            need = battery_max - bat
            if need > 0:
                t += need * recharge_rate
            bat = battery_max
        else:
            # customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False
            t += service[node]
        prev = node
    # return to depot
    d = dist[prev][0]
    total_dist += d
    bat -= d
    t += d / speed
    if bat < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False
    return total_dist, True


def _greedy_construct(route, dist, demands, ready, due, service, stations,
                      battery_max, recharge_rate, speed, horizon, capacity):
    """Given a route (list of customer nodes), try to insert remaining customers greedily."""
    # This is a helper, but for now we'll do a simple nearest neighbor with feasibility checks.
    pass


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    coords = instance["coords"]
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    speed = instance["speed"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dist = _euclid(coords)

    customers = list(range(1, n_customers + 1))
    # Sort by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda v: due[v])

    # Start with one route per customer
    routes = [[v] for v in customers_sorted]

    # Precompute feasibility of each single-customer route
    def is_feasible_route(route):
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False
        _, feas = _route_time_and_battery(route, dist, coords, demands, ready, due,
                                          service, stations, battery_max, recharge_rate,
                                          speed, horizon)
        return feas

    # Verify all initial routes are feasible
    for r in routes:
        if not is_feasible_route(r):
            # This shouldn't happen with one customer per route and full battery,
            # but let's handle it by returning as-is if infeasible
            pass

    # Now try to merge routes to reduce total distance
    # We'll do a simple 2-opt / insertion heuristic:
    # Repeatedly try to insert a customer from one route into another, if feasible.

    best_routes = routes[:]
    best_dist = sum(_route_time_and_battery(r, dist, coords, demands, ready, due,
                                            service, stations, battery_max, recharge_rate,
                                            speed, horizon)[0] for r in routes if r)

    # Merge phase: try to combine routes
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try all pairs of routes
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                # Try to insert each customer from routes[j] into routes[i]
                for k in range(len(routes[j])):
                    cust = routes[j][k]
                    # Try inserting cust into routes[i] at each position
                    for pos in range(len(routes[i]) + 1):
                        new_route = routes[i][:pos] + [cust] + routes[i][pos:]
                        if is_feasible_route(new_route):
                            # Check if this reduces total distance
                            old_dist_i = _route_time_and_battery(routes[i], dist, coords, demands,
                                                                 ready, due, service, stations,
                                                                 battery_max, recharge_rate,
                                                                 speed, horizon)[0]
                            old_dist_j = _route_time_and_battery(routes[j], dist, coords, demands,
                                                                 ready, due, service, stations,
                                                                 battery_max, recharge_rate,
                                                                 speed, horizon)[0]
                            new_dist_i = _route_time_and_battery(new_route, dist, coords, demands,
                                                                 ready, due, service, stations,
                                                                 battery_max, recharge_rate,
                                                                 speed, horizon)[0]
                            remaining_j = routes[j][:k] + routes[j][k + 1:]
                            if remaining_j:
                                new_dist_j = _route_time_and_battery(remaining_j, dist, coords, demands,
                                                                     ready, due, service, stations,
                                                                     battery_max, recharge_rate,
                                                                     speed, horizon)[0]
                            else:
                                new_dist_j = 0.0
                            old_total = old_dist_i + old_dist_j
                            new_total = new_dist_i + new_dist_j
                            if new_total < old_total - 1e-9:
                                routes[i] = new_route
                                routes[j] = remaining_j
                                if not routes[j]:
                                    routes[j] = []
                                changed = True
                                best_dist = sum(_route_time_and_battery(r, dist, coords, demands,
                                                                        ready, due, service, stations,
                                                                        battery_max, recharge_rate,
                                                                        speed, horizon)[0] for r in routes if r)
                                if best_dist < best_dist:
                                    best_routes = [r[:] for r in routes if r]
                                break
                    if changed and time.time() < deadline:
                        break
            if changed and time.time() < deadline:
                break

    # Return the best routes found
    # Ensure all customers are covered
    covered = set()
    for r in best_routes:
        for v in r:
            covered.add(v)
    missing = [v for v in customers if v not in covered]
    for v in missing:
        best_routes.append([v])

    return best_routes
# EVOLVE-BLOCK-END
