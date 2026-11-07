"""EV routing with time windows - constructive + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import time as _time

def _euclid(coords, a, b):
    return np.sqrt((coords[a,0]-coords[b,0])**2 + (coords[a,1]-coords[b,1])**2)

def _route_distance(coords, route):
    """Total distance including depot legs."""
    if not route:
        return 0.0
    d = _euclid(coords, 0, route[0])
    for i in range(len(route)-1):
        d += _euclid(coords, route[i], route[i+1])
    d += _euclid(coords, route[-1], 0)
    return d

def _simulate_route(instance, route):
    """
    Simulate a route. Returns (total_distance, feasible).
    If infeasible, returns (inf, False).
    """
    coords = np.array(instance["coords"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    if not route:
        return 0.0, True
    
    # Check capacity
    total_demand = sum(demands[c] for c in route if c not in stations)
    if total_demand > capacity:
        return float('inf'), False
    
    t = 0.0
    bat = float(battery_max)
    total_dist = 0.0
    cur = 0
    
    for node in route:
        d = _euclid(coords, cur, node)
        t += d / speed
        bat -= d
        total_dist += d
        if bat < -1e-9:
            return float('inf'), False
        
        if node in stations:
            # Charging: time = (battery - energy) * recharge_rate... wait
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # I think this means: recharge_time = (battery_max - current_battery) * recharge_rate
            # Actually re-reading: "adds (battery - energy) * recharge_rate time"
            # This is ambiguous. Let me re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # I think "energy" here means the energy needed to refill, so time = (battery_max - bat) * recharge_rate
            # Wait, "battery - energy" - maybe it's the deficit? Let me interpret as:
            # recharge_time = (battery_max - bat) * recharge_rate
            # Then bat = battery_max
            recharge_time = (battery_max - bat) * recharge_rate
            t += recharge_time
            bat = float(battery_max)
        else:
            # Customer
            t = max(t, float(ready[node]))
            if t > float(due[node]) + 1e-9:
                return float('inf'), False
            t += float(service[node])
        
        cur = node
    
    # Return to depot
    d = _euclid(coords, cur, 0)
    t += d / speed
    bat -= d
    total_dist += d
    if bat < -1e-9:
        return float('inf'), False
    if t > horizon + 1e-9:
        return float('inf'), False
    
    return total_dist, True

def _constructive(instance):
    """
    Nearest-neighbor with time window and battery feasibility, building multiple routes.
    """
    coords = np.array(instance["coords"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    customers = list(range(1, n_customers + 1))
    # Sort by due time (earliest first)
    customers.sort(key=lambda c: due[c])
    
    unassigned = set(customers)
    routes = []
    
    while unassigned:
        # Start new route from depot
        route = []
        cur = 0
        t = 0.0
        bat = float(battery_max)
        load = 0
        
        # Try to insert customers in order of due time
        for c in sorted(unassigned, key=lambda x: due[x]):
            # Check if we can go from cur to c
            d = _euclid(coords, cur, c)
            t_arrive = t + d / speed
            bat_arrive = bat - d
            
            if bat_arrive < -1e-9:
                continue
            if t_arrive > float(due[c]) + 1e-9:
                continue
            if load + demands[c] > capacity:
                continue
            
            # Check if we can return to depot after serving c
            t_after_service = max(t_arrive, float(ready[c])) + float(service[c])
            d_back = _euclid(coords, c, 0)
            t_back = t_after_service + d_back / speed
            bat_back = bat_arrive - d_back
            if bat_back < -1e-9:
                continue
            if t_back > horizon + 1e-9:
                continue
            
            # Accept
            route.append(c)
            t = t_after_service
            bat = bat_arrive
            load += demands[c]
            cur = c
            unassigned.discard(c)
        
        if route:
            routes.append(route)
    
    # Check feasibility of all routes
    all_feasible = True
    for r in routes:
        _, feas = _simulate_route(instance, r)
        if not feas:
            all_feasible = False
            break
    
    if not all_feasible:
        # Fallback: one customer per route
        routes = [[c] for c in customers]
    
    return routes

def _local_search(instance, routes, max_iter=500):
    """
    2-opt within routes + relocate between routes.
    """
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    
    # Compute current total distance
    def total_dist(routes):
        return sum(_route_distance(coords, r) for r in routes)
    
    current_dist = total_dist(routes)
    
    for it in range(max_iter):
        improved = False
        
        # 2-opt within each route
        for ri, route in enumerate(routes):
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Check feasibility
                    d_new, feas = _simulate_route(instance, new_route)
                    if feas:
                        # Check distance improvement
                        d_old_seg = _euclid(coords, route[i-1] if i > 0 else 0, route[i]) + \
                                   _euclid(coords, route[j], route[j+1] if j+1 < len(route) else 0)
                        d_new_seg = _euclid(coords, new_route[i-1] if i > 0 else 0, new_route[i]) + \
                                   _euclid(coords, new_route[j], new_route[j+1] if j+1 < len(route) else 0)
                        if d_new < current_dist + 1e-9:
                            routes[ri] = new_route
                            current_dist = total_dist(routes)
                            improved = True
        
        # Relocate: move a customer from one route to another
        for ri, route in enumerate(routes):
            for ci, c in enumerate(route):
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    # Try inserting c at each position in route rj
                    route_j = routes[rj]
                    for pos in range(len(route_j) + 1):
                        # Remove c from route ri
                        new_ri = route[:ci] + route[ci+1:]
                        # Insert c at pos in route rj
                        new_rj = route_j[:pos] + [c] + route_j[pos:]
                        
                        # Check feasibility
                        feas_ri = True
                        if new_ri:
                            _, feas_ri = _simulate_route(instance, new_ri)
                        feas_rj, feas_rj_ok = _simulate_route(instance, new_rj)
                        
                        if feas_ri and feas_rj_ok:
                            # Check distance
                            old_d = _route_distance(coords, route) + _route_distance(coords, route_j)
                            new_d = _route_distance(coords, new_ri) + _route_distance(coords, new_rj)
                            if new_d < current_dist + 1e-9:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                current_dist = total_dist(routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return routes

def plan(instance):
    start_time = _time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Constructive phase
    routes = _constructive(instance)
    
    elapsed = _time.time() - start_time
    if elapsed > deadline - start_time - 5:
        return routes
    
    # Local search
    max_iter = 200
    routes = _local_search(instance, routes, max_iter=max_iter)
    
    elapsed = _time.time() - start_time
    if elapsed > deadline - start_time - 2:
        return routes
    
    # Try a few more iterations if time permits
    routes = _local_search(instance, routes, max_iter=100)
    
    return routes
# EVOLVE-BLOCK-END
