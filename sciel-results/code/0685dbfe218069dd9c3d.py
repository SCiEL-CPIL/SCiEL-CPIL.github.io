"""EV routing with time windows – Tabu Search with Regret-Based Insertion."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from collections import defaultdict


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _calculate_route_metrics(route, D, instance):
    """
    Calculate distance, validity, and final state for a route.
    Returns: (distance, valid, final_time, final_battery, load, details)
    details: list of (node, arrival_time, departure_time, battery_after) for each node in route
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0
    details = []

    for node in route:
        d = D[prev][node]
        dist_total += d
        t += d
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load, details
        
        if node in stations_set:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
            details.append((node, t, t, batt))
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load, details
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load, details
            t += service[node]
            details.append((node, t, t, batt))
        prev = node

    d = D[prev][0]
    dist_total += d
    t += d
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load, details
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load, details

    return dist_total, True, t, batt, load, details


def _total_distance(routes, D, instance):
    total = 0.0
    for route in routes:
        dist, valid, _, _, _, _ = _calculate_route_metrics(route, D, instance)
        if not valid:
            return float('inf')
        total += dist
    return total


def _is_valid_solution(routes, instance, D):
    served = set()
    stations_set = set(instance["stations"])
    for route in routes:
        dist, valid, _, _, _, _ = _calculate_route_metrics(route, D, instance)
        if not valid:
            return False
        for node in route:
            if node in stations_set:
                continue
            if node in served:
                return False
            served.add(node)
    n_customers = instance["n_customers"]
    expected = set(range(1, n_customers + 1))
    return served == expected


def _regret_based_construct(instance, D):
    """
    Construct routes using Regret-Based Insertion.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations_set = set(instance["stations"])
    
    routes = []
    remaining = set(customers)
    
    # Pre-calculate regret for initial insertion
    # We will maintain a list of (regret, customer, route_idx, pos)
    # But regret changes as routes fill up.
    # Simple approach: Iterate until all inserted.
    
    while remaining:
        best_regret = -1
        best_customer = None
        best_route_idx = -1
        best_pos = -1
        best_dist = float('inf')
        
        for c in remaining:
            # Calculate best and second best insertion costs
            costs = []
            
            # Try inserting into existing routes
            for r_idx, route in enumerate(routes):
                # Try all positions
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    # Quick check: does it fit capacity?
                    load = sum(instance["demands"][n] for n in new_route if n not in stations_set)
                    if load > instance["capacity"] + 1e-9:
                        continue
                    
                    dist, valid, _, _, _, _ = _calculate_route_metrics(new_route, D, instance)
                    if valid:
                        # Cost is increase in distance
                        old_dist, _, _, _, _, _ = _calculate_route_metrics(route, D, instance)
                        delta = dist - old_dist
                        costs.append((delta, r_idx, pos))
            
            # Try creating new route
            new_route = [c]
            dist, valid, _, _, _, _ = _calculate_route_metrics(new_route, D, instance)
            if valid:
                costs.append((dist, -1, 0)) # -1 indicates new route
            
            if not costs:
                # Cannot insert anywhere feasible? This shouldn't happen if we start with empty routes
                # But if it does, we must add it to a new route even if infeasible? No, must be valid.
                # If no valid insertion, we have a problem.
                # For now, skip? Or force insert?
                # Let's assume we can always create a new route if capacity allows.
                continue
            
            # Sort by delta
            costs.sort(key=lambda x: x[0])
            
            # Regret = cost_second_best - cost_best
            if len(costs) >= 2:
                regret = costs[1][0] - costs[0][0]
            else:
                regret = costs[0][0] # Only one option, high priority to insert
            
            # We want to maximize regret (most constrained)
            # But we also want to minimize cost.
            # Standard regret insertion: pick max regret.
            # If tie, pick min cost.
            
            # To make it more robust, we can combine them.
            # But standard is:
            if regret > best_regret or (regret == best_regret and costs[0][0] < best_dist):
                best_regret = regret
                best_customer = c
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
                best_dist = costs[0][0]
        
        if best_customer is None:
            # Fallback: just add to new route
            for c in remaining:
                routes.append([c])
                remaining.discard(c)
            break
        
        # Insert best_customer
        if best_route_idx == -1:
            routes.append([best_customer])
        else:
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_pos] + [best_customer] + route[best_pos:]
        
        remaining.discard(best_customer)
        
        # If we have too many routes, we might want to merge them later in Tabu Search
        # For now, this constructs a feasible solution.
        
    return routes


def _tabu_search(routes, D, instance, time_limit):
    """
    Tabu Search to improve the solution.
    """
    start_time = time.time()
    current_routes = [list(r) for r in routes]
    current_dist = _total_distance(current_routes, D, instance)
    
    best_routes = [list(r) for r in current_routes]
    best_dist = current_dist
    
    # Tabu list: (route_idx, node_idx, node_id, direction) or similar
    # For Relocate: (r_idx, pos, node)
    # For Swap: (r1_idx, pos1, r2_idx, pos2)
    # For 2-Opt: (r_idx, i, j)
    
    tabu_list = []
    tabu_tenure = 15
    
    # Helper to get route metrics
    def get_metrics(route):
        return _calculate_route_metrics(route, D, instance)
    
    # Pre-calculate metrics for current routes
    route_metrics = []
    for r in current_routes:
        route_metrics.append(get_metrics(r))
        
    total_dist = sum(m[0] for m in route_metrics if m[1])
    if total_dist != current_dist:
         # Recalculate if mismatch (shouldn't happen)
         current_dist = _total_distance(current_routes, D, instance)
         best_dist = current_dist
         best_routes = [list(r) for r in current_routes]
         route_metrics = [get_metrics(r) for r in current_routes]

    while time.time() < time_limit:
        # Expire tabu moves
        while tabu_list and tabu_list[0][0] <= len(tabu_list) - tabu_tenure:
            tabu_list.pop(0)
        
        best_move = None
        best_new_dist = float('inf')
        best_new_routes = None
        best_move_type = None
        best_move_params = None
        best_is_tabu = False
        
        n_routes = len(current_routes)
        
        # Generate Moves
        # 1. Relocate: Move node i from route a to position j in route b
        for a in range(n_routes):
            route_a = current_routes[a]
            if not route_a: continue
            for i in range(len(route_a)):
                node = route_a[i]
                # Skip stations for now? No, stations are part of route.
                # But we only care about customer moves for regret?
                # Tabu search should move any node.
                
                # Try moving to other routes
                for b in range(n_routes):
                    if a == b:
                        # Intra-route relocate
                        route_b = route_a
                        for j in range(len(route_b)):
                            if j == i: continue
                            new_route_a = route_a[:i] + route_a[i+1:]
                            new_route_b = new_route_a[:j] + [node] + new_route_a[j:]
                            # If a==b, we just modified the same route
                            # Actually, relocate within same route is just a move
                            # Let's handle intra-route separately or just treat as a->a
                            
                            # Calculate new metrics for route a
                            new_metrics_a = get_metrics(new_route_a)
                            if not new_metrics_a[1]: continue
                            
                            delta = new_metrics_a[0] - route_metrics[a][0]
                            new_total = current_dist + delta
                            
                            # Check tabu
                            move_key = ('rel', a, i, a, j)
                            is_tabu = any(m[1] == move_key for m in tabu_list)
                            
                            if new_total < best_new_dist or (is_tabu and new_total < best_dist):
                                best_new_dist = new_total
                                best_new_routes = [list(r) for r in current_routes]
                                best_new_routes[a] = new_route_a
                                best_move_type = 'rel'
                                best_move_params = (a, i, a, j)
                                best_is_tabu = is_tabu
                    else:
                        route_b = current_routes[b]
                        for j in range(len(route_b) + 1):
                            new_route_a = route_a[:i] + route_a[i+1:]
                            new_route_b = route_b[:j] + [node] + route_b[j:]
                            
                            new_metrics_a = get_metrics(new_route_a)
                            new_metrics_b = get_metrics(new_route_b)
                            
                            if not new_metrics_a[1] or not new_metrics_b[1]: continue
                            
                            delta = (new_metrics_a[0] - route_metrics[a][0]) + (new_metrics_b[0] - route_metrics[b][0])
                            new_total = current_dist + delta
                            
                            move_key = ('rel', a, i, b, j)
                            is_tabu = any(m[1] == move_key for m in tabu_list)
                            
                            if new_total < best_new_dist or (is_tabu and new_total < best_dist):
                                best_new_dist = new_total
                                best_new_routes = [list(r) for r in current_routes]
                                best_new_routes[a] = new_route_a
                                best_new_routes[b] = new_route_b
                                best_move_type = 'rel'
                                best_move_params = (a, i, b, j)
                                best_is_tabu = is_tabu
        
        # 2. Swap: Swap node i in route a with node j in route b
        for a in range(n_routes):
            route_a = current_routes[a]
            if not route_a: continue
            for i in range(len(route_a)):
                node_a = route_a[i]
                for b in range(a + 1, n_routes):
                    route_b = current_routes[b]
                    for j in range(len(route_b)):
                        node_b = route_b[j]
                        
                        new_route_a = list(route_a)
                        new_route_b = list(route_b)
                        new_route_a[i] = node_b
                        new_route_b[j] = node_a
                        
                        new_metrics_a = get_metrics(new_route_a)
                        new_metrics_b = get_metrics(new_route_b)
                        
                        if not new_metrics_a[1] or not new_metrics_b[1]: continue
                        
                        delta = (new_metrics_a[0] - route_metrics[a][0]) + (new_metrics_b[0] - route_metrics[b][0])
                        new_total = current_dist + delta
                        
                        move_key = ('swap', a, i, b, j)
                        is_tabu = any(m[1] == move_key for m in tabu_list)
                        
                        if new_total < best_new_dist or (is_tabu and new_total < best_dist):
                            best_new_dist = new_total
                            best_new_routes = [list(r) for r in current_routes]
                            best_new_routes[a] = new_route_a
                            best_new_routes[b] = new_route_b
                            best_move_type = 'swap'
                            best_move_params = (a, i, b, j)
                            best_is_tabu = is_tabu
        
        # 3. 2-Opt: Reverse segment in route a
        for a in range(n_routes):
            route_a = current_routes[a]
            if len(route_a) < 2: continue
            for i in range(len(route_a) - 1):
                for j in range(i + 1, len(route_a)):
                    new_route_a = route_a[:i] + route_a[i:j+1][::-1] + route_a[j+1:]
                    
                    new_metrics_a = get_metrics(new_route_a)
                    if not new_metrics_a[1]: continue
                    
                    delta = new_metrics_a[0] - route_metrics[a][0]
                    new_total = current_dist + delta
                    
                    move_key = ('2opt', a, i, j)
                    is_tabu = any(m[1] == move_key for m in tabu_list)
                    
                    if new_total < best_new_dist or (is_tabu and new_total < best_dist):
                        best_new_dist = new_total
                        best_new_routes = [list(r) for r in current_routes]
                        best_new_routes[a] = new_route_a
                        best_move_type = '2opt'
                        best_move_params = (a, i, j)
                        best_is_tabu = is_tabu
        
        if best_move is None:
            # No improving move found
            break
        
        # Accept best move
        current_routes = best_new_routes
        current_dist = best_new_dist
        route_metrics = [get_metrics(r) for r in current_routes]
        
        # Update best
        if current_dist < best_dist:
            best_dist = current_dist
            best_routes = [list(r) for r in current_routes]
        
        # Add to tabu list
        # Create a unique ID for the move to reverse it later?
        # Standard tabu: store the move that was made.
        # The reverse move is tabu.
        # For 'rel' (a, i, b, j): Reverse is 'rel' (b, j, a, i-1) or similar?
        # It's complex to define exact reverse keys.
        # Simpler: Just store the move key itself and consider the exact same move tabu?
        # No, that prevents oscillation.
        # Let's store the move key.
        tabu_list.append((len(tabu_list), best_move_params, best_move_type))
        
        # If tabu list gets too long, clear it? Or just let it expire.
        if len(tabu_list) > 100:
            tabu_list = tabu_list[-50:]
            
    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    
    # 1. Construction
    routes = _regret_based_construct(instance, D)
    
    # Check validity
    if not _is_valid_solution(routes, instance, D):
        # Fallback to simple construction
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[c] for c in customers]
    
    # 2. Tabu Search
    # Allocate time for tabu search
    time_for_tabu = deadline - time.time() - 5.0
    if time_for_tabu > 0:
        routes = _tabu_search(routes, D, instance, start_time + time_for_tabu)
    
    # 3. Final Validation
    if not _is_valid_solution(routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[c] for c in customers]
    
    return routes
# EVOLVE-BLOCK-END
