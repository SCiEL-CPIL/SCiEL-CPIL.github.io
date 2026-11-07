import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))
    
    # Sort customers by due date for insertion heuristic
    customers_sorted = sorted(customers, key=lambda c: due[c])
    
    # Initial solution: one route per customer (valid baseline)
    routes = [[c] for c in customers_sorted]
    
    # Helper to check if a route is valid and return its cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, t, d_total, load
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, t, d_total, load
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, t, d_total, load
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, t, d_total, load
        if t > horizon + _EPS:
            return False, 0.0, t, d_total, load
        d_total += d
        return True, d_total, t, d_total, load

    # Check initial validity
    total_dist = 0.0
    for r in routes:
        valid, d, t, _, _ = check_route(r)
        if not valid:
            # Fallback: ensure valid by splitting further? But initial is 1 cust/route, should be valid.
            # If not, we have a problem. But 1 cust/route with full battery is almost always valid.
            # Let's just proceed, the local search will fix it or we return invalid.
            # Actually, let's just use the initial routes as is if valid.
            pass
        total_dist += d

    # Local Search: 2-opt and Relocate
    # We will iterate through routes and try to improve them.
    # Given the small size (25 customers), we can afford a decent amount of search.
    
    # Helper to get route cost
    def get_route_cost(route):
        if not route:
            return 0.0
        valid, d, t, _, _ = check_route(route)
        if not valid:
            return float('inf')
        return d

    # Combine all customers into a single list for easier manipulation?
    # No, keeping separate routes is better for VRP.
    # Let's try to merge routes first to reduce number of trips, then optimize within routes.
    
    # Phase 1: Route Merging
    # Try to merge route j into route i if valid and reduces total distance (or just keeps it valid).
    # Actually, merging reduces the number of depot visits, which is good.
    # Let's sort routes by length or something? 
    # Simple greedy merging:
    # For each pair (i, j), try to insert all customers of j into i.
    # This is expensive O(N^2 * N^2). N=25. 25^4 = 390k ops. Very fast.
    
    merged_routes = []
    for r in routes:
        merged = False
        for i, mr in enumerate(merged_routes):
            # Try to insert each customer of r into mr
            # We can try inserting the whole sequence or individual customers.
            # Individual customers is more flexible.
            new_mr = list(mr)
            success = True
            for c in r:
                # Find best position for c in new_mr
                best_pos = -1
                best_cost = float('inf')
                # Try inserting at every position
                for pos in range(len(new_mr) + 1):
                    trial = new_mr[:pos] + [c] + new_mr[pos:]
                    valid, d, t, _, _ = check_route(trial)
                    if valid:
                        # Cost is distance. We want to minimize distance.
                        # But we are merging, so we just want to find a valid insertion.
                        # To minimize distance increase, we can check the delta distance.
                        # However, check_route computes total distance.
                        # Let's just pick the first valid one or the one with min distance?
                        # Min distance is better.
                        if d < best_cost:
                            best_cost = d
                            best_pos = pos
                if best_pos != -1:
                    new_mr = new_mr[:best_pos] + [c] + new_mr[best_pos:]
                else:
                    success = False
                    break
            if success:
                merged_routes[i] = new_mr
                merged = True
                break
        if not merged:
            merged_routes.append(list(r))
    
    routes = merged_routes

    # Phase 2: Local Search (2-opt and Relocate)
    # Repeat until no improvement or deadline.
    
    def two_opt(route):
        # Standard 2-opt for TSP-like part, but with time windows and battery.
        # We need to check validity after reversal.
        n = len(route)
        improved = False
        for i in range(n - 1):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, n):
                # Reverse segment i..j
                # Note: Reversing changes the direction of travel.
                # Distances are symmetric, so distance sum is same.
                # But time windows and battery might change.
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                valid, d, t, _, _ = check_route(new_route)
                if valid:
                    # Check if distance is actually improved?
                    # Since dist is symmetric, 2-opt on a symmetric matrix doesn't change total distance.
                    # Wait, in VRPTW, 2-opt is often used to improve time windows feasibility or to prepare for other moves.
                    # But here, if dist is symmetric, total distance of the route is the sum of edges.
                    # Reversing a segment changes the edges entering and leaving the segment.
                    # Edge (i-1, i) and (j, j+1) are replaced by (i-1, j) and (i, j+1).
                    # So distance CAN change.
                    old_d = get_route_cost(route)
                    new_d = get_route_cost(new_route)
                    if new_d < old_d - _EPS:
                        route = new_route
                        improved = True
        return route, improved

    def relocate(route, c, new_pos):
        # Remove c and insert at new_pos
        route.remove(c)
        route.insert(new_pos, c)
        valid, d, t, _, _ = check_route(route)
        if not valid:
            # Undo
            route.pop(new_pos)
            # Find original position? We don't know it.
            # Better to copy first.
            return False
        return True

    # Let's implement a proper 2-opt and Relocate loop.
    
    for iter_count in range(50): # Max iterations
        if time.time() > DEADLINE:
            break
        improved = False
        
        # 2-opt
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            new_r, imp = two_opt(r)
            if imp:
                routes[i] = new_r
                improved = True
        
        # Relocate
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            for j, c in enumerate(r):
                if time.time() > DEADLINE:
                    break
                # Try moving c to other positions in the same route
                for k in range(len(r)):
                    if k == j or k == j + 1:
                        continue
                    # Remove c
                    r.remove(c)
                    # Insert at k
                    # Note: indices shift after remove
                    # If k > j, k is now k-1
                    insert_idx = k if k < j else k - 1
                    r.insert(insert_idx, c)
                    valid, d, t, _, _ = check_route(r)
                    if valid:
                        # Check distance improvement
                        # We need old distance. Let's store it before modification.
                        # This is inefficient.
                        # Let's just accept if valid? No, we want to minimize distance.
                        # Let's compute old dist once per route per iteration.
                        pass
                    else:
                        # Undo
                        r.pop(insert_idx)
                        r.insert(j, c)
                
                # Try moving c to other routes
                for m, other_r in enumerate(routes):
                    if m == i:
                        continue
                    # Try inserting c into other_r at every position
                    best_pos = -1
                    best_d = float('inf')
                    old_d_i = get_route_cost(r)
                    old_d_m = get_route_cost(other_r)
                    old_total = old_d_i + old_d_m
                    
                    # Remove c from r
                    r.remove(c)
                    new_r = list(r)
                    
                    for pos in range(len(other_r) + 1):
                        trial_other = other_r[:pos] + [c] + other_r[pos:]
                        valid_r, d_r, t_r, _, _ = check_route(new_r)
                        valid_o, d_o, t_o, _, _ = check_route(trial_other)
                        if valid_r and valid_o:
                            new_total = d_r + d_o
                            if new_total < best_d:
                                best_d = new_total
                                best_pos = pos
                    
                    # Restore r
                    r.insert(j, c)
                    
                    if best_pos != -1 and best_d < old_total - _EPS:
                        # Perform move
                        r.remove(c)
                        routes[i] = list(r)
                        routes[m] = other_r[:best_pos] + [c] + other_r[best_pos:]
                        improved = True
                        break # Restart outer loop? Or just continue.
        
        if not improved:
            break

    # Final Validation and Return
    final_routes = []
    for r in routes:
        if r:
            final_routes.append(r)
        # If a route is empty, it's fine, we just don't include it.
        # But every customer must be served.
        # If a customer got lost, we need to add it back.
        # Let's check coverage.
        
    served = set()
    for r in final_routes:
        for c in r:
            served.add(c)
    
    missing = set(customers) - served
    if missing:
        # Add missing customers as single routes
        for c in missing:
            final_routes.append([c])
            
    return final_routes
