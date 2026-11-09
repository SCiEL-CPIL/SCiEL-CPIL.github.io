"""Rewritten EV routing solver with constructive heuristics and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def _euclidean(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.sqrt(dx * dx + dy * dy)

def _build_dist_matrix(coords, n_nodes):
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _euclidean(coords, i, j)
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _simulate_route(route, instance, dist, speed, battery_cap, recharge_rate, horizon):
    """
    Simulate a route (list of node ids, excluding depot at ends).
    Returns (total_distance, feasible, end_time, min_battery).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    
    if not route:
        return 0.0, True, 0.0, battery_cap
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > instance["capacity"]:
        return 0.0, False, 0.0, 0.0
    
    cur_node = 0
    cur_time = 0.0
    cur_battery = battery_cap
    total_dist = 0.0
    min_battery = battery_cap
    
    for v in route:
        d = dist[cur_node][v]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d
        
        if cur_battery < -1e-9:
            return total_dist, False, cur_time, cur_battery
        
        min_battery = min(min_battery, cur_battery)
        
        if v in stations:
            # Recharge: time added = (battery - energy) * recharge_rate? 
            # Wait, the contract says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This phrasing is weird. Usually it's (capacity - battery) * rate.
            # "refills" implies battery becomes capacity.
            # Let's assume the time cost is proportional to energy added.
            # energy_added = battery_cap - cur_battery
            # time_cost = energy_added * recharge_rate
            # But the text says "(battery - energy) * recharge_rate". 
            # If 'energy' means 'current battery level', then (battery_cap - cur_battery) makes sense.
            # Let's re-read: "adds (battery - energy) * recharge_rate time".
            # Maybe 'battery' is the cap and 'energy' is current?
            # Or maybe it's a typo for (capacity - current).
            # Given "refills", it goes to full.
            # Let's assume time = (battery_cap - cur_battery) * recharge_rate.
            time_cost = (battery_cap - cur_battery) * recharge_rate
            cur_time += time_cost
            cur_battery = battery_cap
        else:
            # Customer
            cur_time = max(cur_time, ready[v])
            if cur_time > due[v] + 1e-9:
                return total_dist, False, cur_time, cur_battery
            cur_time += service[v]
        
        cur_node = v
    
    # Return to depot
    d = dist[cur_node][0]
    cur_time += d / speed
    cur_battery -= d
    total_dist += d
    
    if cur_battery < -1e-9:
        return total_dist, False, cur_time, cur_battery
    
    min_battery = min(min_battery, cur_battery)
    
    if cur_time > horizon + 1e-9:
        return total_dist, False, cur_time, cur_battery
    
    return total_dist, True, cur_time, min_battery

def _insert_customer(route, customer, instance, dist, speed, battery_cap, recharge_rate, horizon):
    """
    Try to insert customer into route at best position.
    Returns (new_route, feasible, delta_dist) or None.
    """
    n = len(route)
    best_pos = -1
    best_delta = float('inf')
    best_new_route = None
    
    # Positions 0..n
    for pos in range(n + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        # Quick check: capacity
        # We assume route was feasible, adding one customer might break capacity.
        # We should check capacity outside or here. Let's assume caller checks capacity.
        
        total_dist, feasible, _, _ = _simulate_route(new_route, instance, dist, speed, battery_cap, recharge_rate, horizon)
        
        if feasible:
            # Calculate delta dist approximately or exactly?
            # Exact delta requires simulating old and new.
            # For speed, we can just use the new total dist vs old total dist?
            # But we don't have old total dist here easily without simulating.
            # Let's just pick the first feasible one or the one with min insertion cost.
            # Insertion cost = dist[i][c] + dist[c][j] - dist[i][j]
            i = route[pos-1] if pos > 0 else 0
            j = route[pos] if pos < n else 0
            delta = dist[i][customer] + dist[customer][j] - dist[i][j]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
                best_new_route = new_route
    
    if best_pos != -1:
        return best_new_route, True, best_delta
    return None

def _solve_instance(instance, time_limit):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    speed = instance["speed"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    # Node 0 is depot. Customers 1..n_customers. Stations are given.
    # Total nodes = 1 + n_customers + len(stations)
    n_nodes = 1 + n_customers + len(stations)
    
    dist = _build_dist_matrix(coords, n_nodes)
    
    customers = list(range(1, n_customers + 1))
    
    # Initial solution: Nearest Neighbor or Cheapest Insertion
    # Let's use a simple constructive heuristic:
    # Sort customers by due time (EDD) or use nearest neighbor.
    # Let's try a simple savings algorithm or nearest insertion.
    
    # Start with empty routes
    routes = []
    unvisited = set(customers)
    
    # Helper to check if a customer can be added to a route
    def can_add(route, customer):
        # Check capacity
        current_demand = sum(instance["demands"][v] for v in route if v <= n_customers)
        if current_demand + instance["demands"][customer] > capacity:
            return False
        # Try all positions
        n = len(route)
        for pos in range(n + 1):
            new_route = route[:pos] + [customer] + route[pos:]
            _, feasible, _, _ = _simulate_route(new_route, instance, dist, speed, battery_cap, recharge_rate, horizon)
            if feasible:
                return True
        return False
    
    # Constructive: Cheapest Insertion
    # Start with one route containing the customer with largest demand? Or just first?
    # Let's start with an empty route and insert customers one by one.
    # To be efficient, we'll maintain a list of routes.
    
    # Sort customers by due time to help feasibility
    sorted_customers = sorted(customers, key=lambda v: instance["due"][v])
    
    for c in sorted_customers:
        inserted = False
        # Try to insert into existing routes
        for i, route in enumerate(routes):
            res = _insert_customer(route, c, instance, dist, speed, battery_cap, recharge_rate, horizon)
            if res is not None:
                new_route, feasible, delta = res
                routes[i] = new_route
                inserted = True
                break
        
        if not inserted:
            # Start a new route
            # Check if single customer route is feasible
            _, feasible, _, _ = _simulate_route([c], instance, dist, speed, battery_cap, recharge_rate, horizon)
            if feasible:
                routes.append([c])
            else:
                # If even single customer is infeasible (shouldn't happen with battery 300 and coords <= 100), 
                # we might need to add a station? 
                # The problem says stations may repeat. 
                # If a customer is too far, we might need to charge on the way.
                # My _simulate_route doesn't insert stations automatically.
                # The input route is a list of node ids. If I put a station in the route, it will be handled.
                # But the constructive phase only inserts customers.
                # If a customer cannot be served in a new route, it's a problem.
                # With battery 300 and max distance ~141 (diagonal of 100x100), 
                # going depot -> customer -> depot is max ~282, which is < 300.
                # So a single customer should always be feasible without charging.
                # Unless service time pushes it over horizon? Horizon is 1000, max time ~300.
                # So it should be fine.
                routes.append([c])
    
    # Local Search: 2-Opt and Or-Opt
    start_time = time.time()
    
    # Precompute route costs
    def get_route_cost(route):
        d, feasible, _, _ = _simulate_route(route, instance, dist, speed, battery_cap, recharge_rate, horizon)
        return d if feasible else float('inf')
    
    current_cost = sum(get_route_cost(r) for r in routes)
    best_cost = current_cost
    best_routes = [r[:] for r in routes]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        
        # 2-Opt within each route
        for i, route in enumerate(routes):
            n = len(route)
            for a in range(n):
                for b in range(a + 1, n):
                    # Reverse segment a..b
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_cost = get_route_cost(new_route)
                    if new_cost < current_cost - 1e-9:
                        routes[i] = new_route
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_routes = [r[:] for r in routes]
        
        # Or-Opt (move 1 node)
        for i, route in enumerate(routes):
            n = len(route)
            for pos in range(n):
                c = route[pos]
                # Remove c
                route_without = route[:pos] + route[pos+1:]
                # Try inserting into other routes
                for j, other_route in enumerate(routes):
                    if i == j:
                        continue
                    res = _insert_customer(other_route, c, instance, dist, speed, battery_cap, recharge_rate, horizon)
                    if res is not None:
                        new_other_route, feasible, delta = res
                        # Calculate new total cost
                        old_cost_i = get_route_cost(route)
                        old_cost_j = get_route_cost(other_route)
                        new_cost_i = get_route_cost(route_without)
                        new_cost_j = get_route_cost(new_other_route)
                        
                        new_total = new_cost_i + new_cost_j
                        old_total = old_cost_i + old_cost_j
                        
                        if new_total < old_total - 1e-9:
                            routes[i] = route_without
                            routes[j] = new_other_route
                            current_cost = new_total
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_routes = [r[:] for r in routes]
                            break # Move c, so break inner loop
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    return best_routes

def plan(instance):
    # The task says "returns routes for 3 fixed generated instances".
    # But the signature is plan(instance).
    # And the description says "instance is a dict".
    # Wait, "Write plan(instance), which returns routes for 3 fixed generated instances".
    # This is ambiguous. Does plan take one instance and return routes for it?
    # Or does it take no args and return routes for 3 instances?
    # The current program:
    # def plan(instance):
    #     ...
    #     return [[v] for v in customers]
    # It returns a list of routes for the single instance passed.
    # "Scoring (lower is better): mean over instances of total Euclidean distance of all routes"
    # This implies the evaluator calls plan 3 times, once for each instance, and averages.
    # Or it calls plan once with a container?
    # The contract says: "instance is a dict: coords, ..."
    # So plan(instance) takes ONE instance.
    # The evaluator likely iterates over 3 instances, calls plan(inst), and averages the scores.
    
    # Time limit per instance?
    # "Each evaluation must finish within 120 s."
    # "the evaluator KILLS the program after 150s wall-clock."
    # If there are 3 instances, and total limit is 150s, we should use ~50s per instance.
    # Let's allocate time dynamically.
    
    # To be safe, let's assume plan is called once per instance.
    # We don't know how many instances will be processed in total within the 150s limit.
    # If the evaluator calls plan 3 times sequentially, and the total time must be < 150s,
    # we should limit each call to ~50s.
    
    # However, we don't have access to a global timer across calls if they are separate process invocations.
    # If they are in the same process, we could track it.
    # Let's assume a single call per instance and limit to 40s to be safe.
    
    time_limit = 40.0
    routes = _solve_instance(instance, time_limit)
    
    # Validate result
    # Every customer must be served exactly once.
    all_customers = set(range(1, instance["n_customers"] + 1))
    served = set()
    for route in routes:
        for v in route:
            if 1 <= v <= instance["n_customers"]:
                if v in served:
                    # Duplicate, invalid
                    pass
                served.add(v)
    
    if served != all_customers:
        # Fallback to trivial solution
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END
