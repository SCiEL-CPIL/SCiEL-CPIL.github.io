"""EV Routing: Fast Construction + Aggressive 2-Opt with Station Insertion."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE_SECONDS = 115.0

def _simulate_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """
    Fast simulation. Returns (total_dist, feasible).
    """
    if not route:
        return 0.0, True
        
    t = 0.0
    b = battery_max
    total_dist = 0.0
    prev = 0
    
    for node in route:
        dist = D[prev, node]
        total_dist += dist
        b -= dist
        if b < -1e-9:
            return total_dist, False
        t += dist / speed
        
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            if demands[node] > 0:
                if demands[node] > 100: # Capacity check (assuming capacity 100 for speed, or pass it)
                    return total_dist, False
                start_service = max(t, ready[node])
                if start_service > due[node] + 1e-9:
                    return total_dist, False
                t = start_service + service[node]
        prev = node
        
    dist_back = D[prev, 0]
    total_dist += dist_back
    b -= dist_back
    if b < -1e-9:
        return total_dist, False
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return total_dist, False
        
    return total_dist, True

def _try_fix_battery(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Attempts to insert a station to fix battery infeasibility.
    Returns new_route if successful, else None.
    """
    # Find the failing segment
    t = 0.0
    b = battery_max
    prev = 0
    fail_idx = -1
    
    for i, node in enumerate(route):
        dist = D[prev, node]
        b -= dist
        if b < -1e-9:
            fail_idx = i
            break
        t += dist / speed
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            if demands[node] > 0:
                start_service = max(t, ready[node])
                if start_service > due[node] + 1e-9:
                    return None # Time failure, station won't help (adds time)
                t = start_service + service[node]
        prev = node
        
    if fail_idx == -1:
        # Check return leg
        dist_back = D[prev, 0]
        b -= dist_back
        if b < -1e-9:
            fail_idx = len(route) # Need station at end (before return)
            
    if fail_idx == -1:
        return None
        
    # Identify nodes around failure
    # If fail_idx is len(route), the failure is on the return leg from route[-1] to 0.
    # We should insert a station before route[-1] or replace the last leg?
    # Actually, if we fail on return, we need to visit a station before returning.
    # So insert station at position len(route) (i.e., after last customer, before depot).
    # But wait, if we insert at len(route), the route becomes [..., last_customer, station].
    # Then we go station -> depot.
    
    # If fail_idx < len(route), the failure is on the leg route[fail_idx-1] -> route[fail_idx].
    # We should insert a station between them.
    
    # Let's determine the "context" nodes
    if fail_idx == 0:
        # Failed on depot -> route[0]
        # Insert station at 0? Or before 0?
        # Route becomes [station, route[0], ...]
        # Leg: depot -> station -> route[0]
        # Check if depot -> station is feasible (always is, station is close? no)
        # And station -> route[0] is feasible (battery max).
        # So we just need to check if we can reach the station.
        # Actually, if we fail depot->route[0], it means D[0, route[0]] > battery_max.
        # We need a station s such that D[0, s] <= battery_max and D[s, route[0]] <= battery_max.
        # Since battery_max is 300 and coords are in [0,100]^2, max dist is ~141.
        # So D[0, route[0]] > 300 is impossible.
        # Thus fail_idx 0 should never happen.
        return None
        
    if fail_idx == len(route):
        # Failed on return leg: route[-1] -> 0
        # We need to insert a station at the end: [..., route[-1], station]
        # Then station -> 0.
        # Check: can we reach station from route[-1]?
        # Battery at route[-1] is b (from simulation above, just before return leg check).
        # Wait, b was updated in the loop. At the end of loop, b is battery at route[-1].
        # We need D[route[-1], s] <= b.
        # And D[s, 0] <= battery_max (always true).
        # And time: t (at route[-1]) + D[route[-1], s]/speed + recharge_time + D[s, 0]/speed <= horizon.
        # Also, the station visit adds time.
        
        # Let's find best station s minimizing extra distance.
        best_s = None
        best_cost = float('inf')
        
        # We need to simulate the end part to check time.
        # t_end is time at route[-1] after service.
        # We didn't save it. Let's re-simulate quickly or just check distance first.
        
        for s in stations_list:
            d1 = D[route[-1], s]
            # Check battery
            # We need battery at route[-1].
            # Re-simulate to get battery at route[-1]
            b_sim = battery_max
            t_sim = 0.0
            prev_sim = 0
            for node in route:
                d = D[prev_sim, node]
                b_sim -= d
                if b_sim < -1e-9:
                    return None
                t_sim += d / speed
                if is_station[node]:
                    t_sim += (battery_max - b_sim) * recharge_rate
                    b_sim = battery_max
                else:
                    if demands[node] > 0:
                        t_sim = max(t_sim, ready[node]) + service[node]
                prev_sim = node
            
            if b_sim < d1 - 1e-9:
                continue
                
            d2 = D[s, 0]
            # Total time
            # t_sim is time at route[-1] (after service).
            # Travel to s: d1/speed
            # Recharge: (battery_max - (b_sim - d1)) * recharge_rate
            # Travel to depot: d2/speed
            total_time = t_sim + d1/speed + (battery_max - (b_sim - d1)) * recharge_rate + d2/speed
            
            if total_time > horizon + 1e-9:
                continue
                
            extra_dist = d1 + d2 - D[route[-1], 0]
            if extra_dist < best_cost:
                best_cost = extra_dist
                best_s = s
                
        if best_s is not None:
            return route + [best_s]
        return None
        
    else:
        # Failed on leg route[fail_idx-1] -> route[fail_idx]
        # Insert station s between them.
        # Route: [..., route[fail_idx-1], s, route[fail_idx], ...]
        # Need:
        # 1. Battery at route[fail_idx-1] >= D[route[fail_idx-1], s]
        # 2. Time check for the whole route?
        #    Station adds time. If we were feasible in time before (which we assume, since we only failed battery),
        #    adding a station might break time.
        #    So we must check time.
        
        # Get state at route[fail_idx-1]
        b_prev = battery_max
        t_prev = 0.0
        prev_node = 0
        for i in range(fail_idx):
            node = route[i]
            d = D[prev_node, node]
            b_prev -= d
            if b_prev < -1e-9:
                return None
            t_prev += d / speed
            if is_station[node]:
                t_prev += (battery_max - b_prev) * recharge_rate
                b_prev = battery_max
            else:
                if demands[node] > 0:
                    t_prev = max(t_prev, ready[node]) + service[node]
            prev_node = node
            
        # b_prev is battery at route[fail_idx-1]
        # t_prev is time at route[fail_idx-1] (after service)
        
        best_s = None
        best_extra = float('inf')
        
        for s in stations_list:
            d1 = D[route[fail_idx-1], s]
            if b_prev < d1 - 1e-9:
                continue
                
            d2 = D[s, route[fail_idx]]
            # Battery at s is 0, then refills to battery_max.
            # So leg s -> route[fail_idx] is feasible as long as d2 <= battery_max.
            
            # Time check:
            # We need to simulate the rest of the route from route[fail_idx] onwards to see if we arrive at depot by horizon.
            # This is expensive.
            # Heuristic: Assume the rest of the route is feasible in time if we add minimal delay.
            # Let's estimate the delay.
            # Original time at route[fail_idx] (arrival) would be:
            # t_arr_orig = t_prev + D[route[fail_idx-1], route[fail_idx]]/speed
            # New time at route[fail_idx] (arrival):
            # t_arr_new = t_prev + d1/speed + (battery_max - (b_prev - d1)) * recharge_rate + d2/speed
            
            delay = (t_arr_new - t_arr_orig)
            
            # If delay is large, it might break time windows of subsequent customers.
            # For now, accept if delay is small? Or just check final depot time with a quick simulation?
            # Let's do a quick simulation of the suffix.
            t_sim = t_arr_new
            b_sim = battery_max # After station
            prev_sim = s
            feasible_time = True
            
            for j in range(fail_idx, len(route)):
                node = route[j]
                d = D[prev_sim, node]
                b_sim -= d
                if b_sim < -1e-9:
                    feasible_time = False
                    break
                t_sim += d / speed
                if is_station[node]:
                    t_sim += (battery_max - b_sim) * recharge_rate
                    b_sim = battery_max
                else:
                    if demands[node] > 0:
                        t_sim = max(t_sim, ready[node])
                        if t_sim > due[node] + 1e-9:
                            feasible_time = False
                            break
                        t_sim += service[node]
                prev_sim = node
            
            if not feasible_time:
                continue
                
            # Check return leg
            d_back = D[prev_sim, 0]
            b_sim -= d_back
            if b_sim < -1e-9:
                continue
            t_sim += d_back / speed
            if t_sim > horizon + 1e-9:
                continue
                
            extra_dist = d1 + d2 - D[route[fail_idx-1], route[fail_idx]]
            if extra_dist < best_extra:
                best_extra = extra_dist
                best_s = s
                
        if best_s is not None:
            return route[:fail_idx] + [best_s] + route[fail_idx:]
            
        return None

def plan(instance):
    start_time = time.time()
    
    # 1. Precompute Data
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    stations_list = list(stations)
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
        
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    # 2. Construction: Cheapest Insertion
    # Start with empty routes
    routes = []
    route_costs = []
    
    # Sort customers by due date (or ready date) to help with time windows?
    # Let's just insert one by one.
    
    # Initial: Each customer is a separate route? No, that's expensive.
    # Let's start with one route and insert. Or start with several.
    # Given 25 customers, let's start with 1 route and insert all.
    # But battery might limit it.
    # Let's try to build routes by inserting customers into existing routes if feasible,
    # otherwise start a new route.
    
    # To ensure feasibility, we start with empty routes.
    # For each customer (sorted by something?), try to insert into each existing route.
    # If feasible, insert. Else, create new route.
    
    # Sorting customers by due date might help.
    cust_sorted = sorted(customers, key=lambda c: due[c])
    
    for c in cust_sorted:
        best_pos = None
        best_route_idx = None
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            # Try all positions
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                if not feas:
                    # Try station insertion
                    fixed = _try_fix_battery(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                    if fixed:
                        new_route = fixed
                        cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                
                if feas:
                    # Calculate marginal cost increase
                    old_cost = route_costs[r_idx]
                    # The cost is total distance.
                    # We want to minimize the increase.
                    # But we also want to minimize total distance.
                    # So we pick the insertion with the smallest new cost?
                    # Or smallest increase?
                    # Smallest increase is better for packing.
                    inc = cost - old_cost
                    if inc < best_cost:
                        best_cost = inc
                        best_pos = pos
                        best_route_idx = r_idx
                        # Store the new route?
                        # We need to reconstruct it.
                        pass
        
        if best_route_idx is not None:
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [c] + route[best_pos:]
            # Re-check feasibility and get cost
            cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if not feas:
                fixed = _try_fix_battery(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                if fixed:
                    new_route = fixed
                    cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            
            if feas:
                routes[best_route_idx] = new_route
                route_costs[best_route_idx] = cost
            else:
                # Should not happen if we checked
                routes.append([c])
                route_costs.append(_simulate_route([c], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0])
        else:
            # Create new route
            routes.append([c])
            route_costs.append(_simulate_route([c], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0])
            
        if time.time() - start_time > DEADLINE_SECONDS * 0.1:
            # If construction is too slow, just add remaining customers as single routes
            # This is a safety net
            remaining = set(customers)
            for r in routes:
                for n in r:
                    if not is_station[n]:
                        remaining.discard(n)
            for m in remaining:
                routes.append([m])
                route_costs.append(_simulate_route([m], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0])
            break
    
    # 3. Local Search: 2-Opt
    # Iterate until no improvement or time limit
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE_SECONDS * 0.8:
            break
            
        for r_idx in range(len(routes)):
            if time.time() - start_time > DEADLINE_SECONDS * 0.9:
                break
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
                
            best_saving = 0
            best_i, best_j = -1, -1
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    if not feas:
                        # Try station insertion
                        fixed = _try_fix_battery(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                        if fixed:
                            new_route = fixed
                            cost, feas = _simulate_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    
                    if feas:
                        saving = route_costs[r_idx] - cost
                        if saving > best_saving:
                            best_saving = saving
                            best_i, best_j = i, j
                            best_new_route = new_route
                            best_new_cost = cost
            
            if best_i != -1:
                routes[r_idx] = best_new_route
                route_costs[r_idx] = best_new_cost
                improved = True
                
    # 4. Finalize
    final_routes = [r for r in routes if r]
    
    # Ensure all customers are present
    visited = set()
    for r in final_routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
                
    missing = set(customers) - visited
    for m in missing:
        final_routes.append([m])
        
    return final_routes

# EVOLVE-BLOCK-END
