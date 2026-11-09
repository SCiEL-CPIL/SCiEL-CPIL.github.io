# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import permutations


def _euclid(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _validate_route(route, instance, D):
    """Check if a route (list of node ids, no depot) is feasible."""
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    # Check demand
    total_demand = sum(demand[v] for v in route)
    if total_demand > capacity:
        return False

    # Simulate
    t = 0.0
    battery = battery_max
    # Start at depot (node 0)
    prev = 0

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return False

        if v in stations:
            # Recharge
            if battery < 0:
                return False
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
            # No service time for stations
        else:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False
            t += service[v]

        prev = v

    # Return to depot
    d = D[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False

    return True


def _route_distance(route, D):
    """Total Euclidean distance of a route including depot legs."""
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _solve_instance(instance, deadline):
    """Solve a single instance using multi-start VND with time/battery awareness."""
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = set(instance["stations"])
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    D = _build_dist_matrix(coords)

    # Pre-compute: for each customer, the nearest neighbor distances
    # Build initial solution: sort by due time, greedily build routes
    def _greedy_build():
        """Build initial solution by sorting customers by due time and greedily inserting."""
        unassigned = set(customers)
        routes = []
        # Sort by due time
        sorted_custs = sorted(customers, key=lambda v: due[v])
        
        for c in sorted_custs:
            if c not in unassigned:
                continue
            best_route = -1
            best_pos = -1
            best_dist = float('inf')
            
            for ri, route in enumerate(routes):
                # Try inserting c at each position
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    if _validate_route(new_route, instance, D):
                        new_dist = _route_distance(new_route, D)
                        old_dist = _route_distance(route, D)
                        improvement = old_dist - new_dist
                        if improvement > best_dist:
                            best_dist = improvement
                            best_route = ri
                            best_pos = pos
            
            # Also consider starting a new route
            if _validate_route([c], instance, D):
                new_dist = _route_distance([c], D)
                if best_route == -1 or best_dist < 0:
                    # New route might be better if no good insertion
                    if best_route == -1:
                        best_route = len(routes)
                        best_pos = 0
                    else:
                        # Compare: new route distance vs best insertion
                        if new_dist < _route_distance(routes[best_route], D) - best_dist:
                            best_route = len(routes)
                            best_pos = 0
            
            if best_route == -1:
                # Can't insert anywhere, must start new route (shouldn't happen with full battery)
                routes.append([c])
                unassigned.discard(c)
            elif best_route == len(routes):
                routes.append([c])
                unassigned.discard(c)
            else:
                routes[best_route].insert(best_pos, c)
                unassigned.discard(c)
        
        return routes

    # VND: Variable Neighborhood Descent
    def _vnd(routes, max_iter=50):
        """Apply VND with relocate and 2-opt moves."""
        current_routes = [list(r) for r in routes]
        current_dist = sum(_route_distance(r, D) for r in current_routes)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Neighborhood 1: Relocate
            for ri in range(len(current_routes)):
                if time.time() > deadline - 5:
                    break
                for pos in range(len(current_routes[ri])):
                    c = current_routes[ri][pos]
                    # Try removing and inserting into other routes
                    for rj in range(len(current_routes)):
                        if rj == ri:
                            continue
                        for new_pos in range(len(current_routes[rj]) + 1):
                            # Remove c from ri
                            temp_ri = current_routes[ri][:]
                            temp_ri.pop(pos)
                            # Insert c into rj
                            temp_rj = current_routes[rj][:]
                            temp_rj.insert(new_pos, c)
                            
                            if _validate_route(temp_ri, instance, D) and _validate_route(temp_rj, instance, D):
                                new_dist = sum(_route_distance(r, D) for r in current_routes) - _route_distance(current_routes[ri], D) - _route_distance(current_routes[rj], D) + _route_distance(temp_ri, D) + _route_distance(temp_rj, D)
                                if new_dist < current_dist - 1e-9:
                                    current_routes[ri] = temp_ri
                                    current_routes[rj] = temp_rj
                                    current_dist = new_dist
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
            
            # Neighborhood 2: 2-opt within each route
            for ri in range(len(current_routes)):
                if time.time() > deadline - 5:
                    break
                route = current_routes[ri]
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if _validate_route(new_route, instance, D):
                            new_dist = current_dist - _route_distance(route, D) + _route_distance(new_route, D)
                            if new_dist < current_dist - 1e-9:
                                current_routes[ri] = new_route
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Neighborhood 3: Swap between routes
            for ri in range(len(current_routes)):
                if time.time() > deadline - 5:
                    break
                for rj in range(ri + 1, len(current_routes)):
                    for pi in range(len(current_routes[ri])):
                        for pj in range(len(current_routes[rj])):
                            c1 = current_routes[ri][pi]
                            c2 = current_routes[rj][pj]
                            temp_ri = current_routes[ri][:]
                            temp_rj = current_routes[rj][:]
                            temp_ri[pi], temp_rj[pj] = c2, c1
                            if _validate_route(temp_ri, instance, D) and _validate_route(temp_rj, instance, D):
                                new_dist = current_dist - _route_distance(current_routes[ri], D) - _route_distance(current_routes[rj], D) + _route_distance(temp_ri, D) + _route_distance(temp_rj, D)
                                if new_dist < current_dist - 1e-9:
                                    current_routes[ri] = temp_ri
                                    current_routes[rj] = temp_rj
                                    current_dist = new_dist
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
        
        # Remove empty routes
        current_routes = [r for r in current_routes if r]
        return current_routes, current_dist

    # Multi-start: try several initial solutions
    best_routes = None
    best_dist = float('inf')
    
    # Start 1: Greedy by due time
    routes1 = _greedy_build()
    routes1, dist1 = _vnd(routes1)
    if dist1 < best_dist:
        best_dist = dist1
        best_routes = [list(r) for r in routes1]
    
    if time.time() > deadline - 10:
        return best_routes
    
    # Start 2: Greedy by nearest neighbor from depot
    def _nn_build():
        unassigned = set(customers)
        routes = []
        while unassigned:
            # Pick customer closest to depot among unassigned
            start = min(unassigned, key=lambda v: D[0, v])
            route = [start]
            unassigned.discard(start)
            last = start
            while unassigned:
                best_next = None
                best_d = float('inf')
                for c in unassigned:
                    d = D[last, c]
                    if d < best_d:
                        best_d = d
                        best_next = c
                if best_next is None:
                    break
                new_route = route + [best_next]
                if _validate_route(new_route, instance, D):
                    route = new_route
                    unassigned.discard(best_next)
                    last = best_next
                else:
                    break
            routes.append(route)
        return routes
    
    routes2 = _nn_build()
    routes2, dist2 = _vnd(routes2)
    if dist2 < best_dist:
        best_dist = dist2
        best_routes = [list(r) for r in routes2]
    
    if time.time() > deadline - 5:
        return best_routes
    
    # Start 3: Random restarts
    np.random.seed(42)
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        shuffled = customers[:]
        np.random.shuffle(shuffled)
        
        # Build initial solution with shuffled order
        unassigned = set(shuffled)
        routes = []
        for c in shuffled:
            best_route = -1
            best_pos = -1
            best_dist_improve = float('inf')
            
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    if _validate_route(new_route, instance, D):
                        new_d = _route_distance(new_route, D)
                        old_d = _route_distance(route, D)
                        imp = old_d - new_d
                        if imp > best_dist_improve:
                            best_dist_improve = imp
                            best_route = ri
                            best_pos = pos
            
            if best_route == -1:
                if _validate_route([c], instance, D):
                    routes.append([c])
                else:
                    # Force assign to some route (shouldn't happen)
                    routes.append([c])
            else:
                routes[best_route].insert(best_pos, c)
        
        routes, dist = _vnd(routes)
        if dist < best_dist:
            best_dist = dist
            best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
    
    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We need to solve 3 instances. But the problem says "returns routes for 3 fixed generated instances"
    # Wait, re-reading: "Write plan(instance), which returns routes for 3 fixed generated instances"
    # Actually, looking at the seed program, it takes a single instance and returns routes for that instance.
    # The task says "each with 25 customers and 4 charging stations" - so each instance is separate.
    # The evaluator likely calls plan() three times with three different instances, or once with all three.
    # Looking at the seed: it takes one instance dict and returns a list of routes.
    # So I'll assume plan is called once per instance.
    
    routes = _solve_instance(instance, deadline)
    return routes
# EVOLVE-BLOCK-END
