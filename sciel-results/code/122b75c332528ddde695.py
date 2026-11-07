"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    # Vectorized distance calculation
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    
    # Precompute demand array for faster access
    demands_arr = np.array(demands, dtype=float)
    
    customers = list(range(1, n_customers + 1))
    stations_set = set(stations)
    cust_set = set(customers)

    def simulate(route):
        """
        Fast simulation of a route.
        Returns (total_dist, is_feasible, total_time, end_battery)
        """
        if not route:
            return 0.0, True, 0.0, battery_cap

        t = 0.0
        bat = battery_cap
        total_dist = 0.0
        prev = 0

        # Local bindings for speed
        _D = D
        _ready = ready
        _due = due
        _service = service
        _bat_cap = battery_cap
        _speed = speed
        _rr = recharge_rate
        _cust_set = cust_set
        _station_set = stations_set

        for v in route:
            d = _D[prev, v]
            total_dist += d
            bat -= d
            t += d / _speed

            if bat < -1e-9:
                return total_dist, False, t, bat

            if v in _cust_set:
                if t < _ready[v]:
                    t = _ready[v]
                if t > _due[v] + 1e-9:
                    return total_dist, False, t, bat
                t += _service[v]
            elif v in _station_set:
                # Recharge
                energy_needed = _bat_cap - bat
                t += energy_needed * _rr
                bat = _bat_cap

            prev = v

        # Return to depot
        d = _D[prev, 0]
        total_dist += d
        bat -= d
        t += d / _speed

        if bat < -1e-9:
            return total_dist, False, t, bat
        if t > horizon + 1e-9:
            return total_dist, False, t, bat

        return total_dist, True, t, bat

    def get_demand(route):
        """Calculate total customer demand in a route."""
        total = 0.0
        for v in route:
            if v in cust_set:
                total += demands_arr[v]
        return total

    def try_insert_station(route):
        """
        Attempts to make a route battery-feasible by inserting the best station.
        Returns (new_route, new_dist, success)
        """
        best_route = None
        best_dist = float('inf')
        
        n = len(route)
        # Try inserting each station at each position
        # Optimization: Only try stations that are "near" the deficit or just try all if small
        for s in stations:
            for pos in range(n + 1):
                # Heuristic: If we already have a station, maybe we don't need to try all positions?
                # But for correctness and simplicity, we try all.
                # To speed up, we can check if inserting s at pos reduces the max battery deficit.
                # For now, simple simulation.
                new_route = route[:pos] + [s] + route[pos:]
                dist, feas, _, _ = simulate(new_route)
                if feas and dist < best_dist:
                    best_dist = dist
                    best_route = new_route
                    
        if best_route is not None:
            return best_route, best_dist, True
        return None, float('inf'), False

    # Phase 1: Construction (Clarke-Wright Savings)
    # Start with each customer as a separate route, ensuring battery feasibility
    routes = []
    route_dists = []
    route_demands = []
    
    for c in customers:
        r = [c]
        dist, feas, _, _ = simulate(r)
        if not feas:
            # Try to fix battery
            new_r, new_d, success = try_insert_station(r)
            if success:
                r = new_r
                dist = new_d
            else:
                # Should not happen if instance is solvable, but handle gracefully
                # If we can't even serve a single customer, the instance might be invalid or we need multiple stations?
                # The problem statement implies valid instances.
                r = [c]
                dist = float('inf')
        
        routes.append(r)
        route_dists.append(dist)
        route_demands.append(demands_arr[c])

    # Savings-based merging
    # We maintain a list of active routes
    # To speed up, we can use a priority queue or just iterate.
    # Given N=25, iterating is fine.
    
    changed = True
    while changed and (time.time() < deadline - 10.0):
        changed = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_new_route = None
        best_new_dist = 0.0
        best_new_demand = 0.0
        best_dir = 0 # 0: i+j, 1: i+rev(j), 2: rev(i)+j, 3: rev(i)+rev(j)

        n_routes = len(routes)
        for i in range(n_routes):
            if not routes[i]: continue
            ri = routes[i]
            di = route_dists[i]
            dei = route_demands[i]
            
            for j in range(i + 1, n_routes):
                if not routes[j]: continue
                rj = routes[j]
                dj = route_dists[j]
                dej = route_demands[j]
                
                # Capacity check
                if dei + dej > capacity:
                    continue

                # Try 4 directions
                candidates = [
                    ri + rj,
                    ri + rj[::-1],
                    ri[::-1] + rj,
                    ri[::-1] + rj[::-1]
                ]
                
                for k, cand in enumerate(candidates):
                    # Quick check: if demand is ok, simulate
                    dist, feas, _, _ = simulate(cand)
                    if not feas:
                        # Try to fix battery? 
                        # In construction, it's better to be strict to avoid complex initial states,
                        # but if a saving is large, maybe it's worth trying?
                        # For now, skip if not directly feasible.
                        continue
                    
                    saving = di + dj - dist
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_new_route = cand
                        best_new_dist = dist
                        best_new_demand = dei + dej
                        best_dir = k

        if best_saving > 1e-6:
            # Perform merge
            routes[best_i] = best_new_route
            route_dists[best_i] = best_new_dist
            route_demands[best_i] = best_new_demand
            
            # Clear route j
            routes[best_j] = []
            route_dists[best_j] = 0.0
            route_demands[best_j] = 0.0
            
            changed = True

    # Remove empty routes
    active_indices = [i for i in range(len(routes)) if routes[i]]
    routes = [routes[i] for i in active_indices]
    route_dists = [route_dists[i] for i in active_indices]
    route_demands = [route_demands[i] for i in active_indices]

    # Phase 2: Local Search
    # We will use a loop that tries different moves until no improvement or time runs out.
    
    def try_or_opt(route, route_dist, route_demand):
        """
        Try Or-opt moves (move 1, 2, 3 nodes) within the route.
        Returns (improved, new_route, new_dist)
        """
        n = len(route)
        if n < 3:
            return False, route, route_dist
            
        best_route = None
        best_dist = route_dist
        improved = False
        
        for seg_len in [1, 2, 3]:
            if n < seg_len + 1:
                continue
                
            for i in range(n - seg_len + 1):
                segment = route[i:i + seg_len]
                # Remove segment
                remaining = route[:i] + route[i + seg_len:]
                
                # If removing segment makes route empty (shouldn't happen if seg_len < n), handle
                if not remaining:
                    continue
                    
                # Try inserting segment at all positions in 'remaining'
                for j in range(len(remaining) + 1):
                    # Skip original position and adjacent positions to avoid trivial moves
                    if j == i:
                        continue
                    if j > i and j <= i + seg_len:
                        continue
                        
                    new_route = remaining[:j] + segment + remaining[j:]
                    
                    # Check capacity? Demand doesn't change, so it's fine.
                    dist, feas, _, _ = simulate(new_route)
                    
                    if feas and dist < best_dist - 1e-9:
                        best_dist = dist
                        best_route = new_route
                        improved = True
                        
        if improved:
            return True, best_route, best_dist
        return False, route, route_dist

    def try_2_opt(route, route_dist):
        """
        Try 2-opt moves within the route.
        """
        n = len(route)
        if n < 3:
            return False, route, route_dist
            
        best_route = None
        best_dist = route_dist
        improved = False
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i to j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                dist, feas, _, _ = simulate(new_route)
                
                if feas and dist < best_dist - 1e-9:
                    best_dist = dist
                    best_route = new_route
                    improved = True
                    
        if improved:
            return True, best_route, best_dist
        return False, route, route_dist

    def try_remove_station(route, route_dist):
        """
        Try removing a station if the route is still feasible without it.
        """
        improved = False
        new_route = route
        new_dist = route_dist
        
        for idx in range(len(route)):
            if route[idx] in stations_set:
                test_route = route[:idx] + route[idx+1:]
                dist, feas, _, _ = simulate(test_route)
                if feas and dist < new_dist - 1e-9:
                    new_route = test_route
                    new_dist = dist
                    improved = True
                    # Don't break, try to remove more stations? 
                    # Usually removing one might allow removing another.
                    # But we need to re-evaluate indices if we modify the list in place.
                    # Simpler: just return the first improvement found.
                    break
                    
        if improved:
            return True, new_route, new_dist
        return False, route, route_dist

    # Main Local Search Loop
    # To prevent infinite loops or oscillations, we use a simple "improved" flag.
    # We also need to handle inter-route moves.
    
    improved = True
    while improved and (time.time() < deadline - 5.0):
        improved = False
        
        # 1. Intra-route improvements
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            dist = route_dists[r_idx]
            demand = route_demands[r_idx]
            
            if not route:
                continue
                
            # Try Or-opt
            imp, new_route, new_dist = try_or_opt(route, dist, demand)
            if imp:
                routes[r_idx] = new_route
                route_dists[r_idx] = new_dist
                improved = True
                continue # Restart intra-route checks for this route? Or just move to next?
                # Usually, one improvement allows further improvements, so restarting the loop for this route is good.
                # But to keep it simple, we just mark improved and continue to next route.
                # A more aggressive approach would be to loop on a single route until local optimum.
                
            # Try 2-opt
            imp, new_route, new_dist = try_2_opt(route, dist)
            if imp:
                routes[r_idx] = new_route
                route_dists[r_idx] = new_dist
                improved = True
                continue
                
            # Try remove station
            imp, new_route, new_dist = try_remove_station(route, dist)
            if imp:
                routes[r_idx] = new_route
                route_dists[r_idx] = new_dist
                improved = True
                continue

        # 2. Inter-route moves (Relocation)
        if not improved and (time.time() < deadline - 5.0):
            for src_idx in range(len(routes)):
                if not routes[src_idx]:
                    continue
                src_route = routes[src_idx]
                src_dist = route_dists[src_idx]
                src_demand = route_demands[src_idx]
                
                for i in range(len(src_route)):
                    cust = src_route[i]
                    if cust in stations_set:
                        continue # Don't move stations between routes easily, they are specific to route structure
                    
                    # Remove customer from source
                    new_src_route = src_route[:i] + src_route[i+1:]
                    new_src_demand = src_demand - demands_arr[cust]
                    
                    # If source becomes empty, we can skip simulation if we are just moving
                    if new_src_route:
                        src_dist_after, src_feas, _, _ = simulate(new_src_route)
                        if not src_feas:
                            continue
                    else:
                        src_dist_after = 0.0
                        src_feas = True

                    for dst_idx in range(len(routes)):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        dst_dist = route_dists[dst_idx]
                        dst_demand = route_demands[dst_idx]
                        
                        if dst_demand + demands_arr[cust] > capacity:
                            continue

                        # Try inserting customer into destination at all positions
                        for j in range(len(dst_route) + 1):
                            new_dst_route = dst_route[:j] + [cust] + dst_route[j:]
                            
                            # Check if new destination is feasible
                            dst_dist_after, dst_feas, _, _ = simulate(new_dst_route)
                            if not dst_feas:
                                # Try to fix battery?
                                # Inter-route moves are expensive. If it's infeasible, skip.
                                continue
                                
                            total_old = src_dist + dst_dist
                            total_new = src_dist_after + dst_dist_after
                            
                            if total_new < total_old - 1e-6:
                                # Perform move
                                routes[src_idx] = new_src_route
                                route_dists[src_idx] = src_dist_after
                                route_demands[src_idx] = new_src_demand
                                
                                routes[dst_idx] = new_dst_route
                                route_dists[dst_idx] = dst_dist_after
                                route_demands[dst_idx] = dst_demand + demands_arr[cust]
                                
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        # 3. Inter-route 2-opt (Cross Exchange)
        if not improved and (time.time() < deadline - 5.0):
            for i in range(len(routes)):
                if not routes[i]: continue
                ri = routes[i]
                di = route_dists[i]
                dei = route_demands[i]
                
                for j in range(i + 1, len(routes)):
                    if not routes[j]: continue
                    rj = routes[j]
                    dj = route_dists[j]
                    dej = route_demands[j]
                    
                    # Try swapping segments
                    # p: cut in ri, q: cut in rj
                    # ri = A + B, rj = C + D
                    # New: A + D, C + B
                    # Demand check:
                    # dem(A) + dem(D) <= cap
                    # dem(C) + dem(B) <= cap
                    
                    # To speed up, we can precompute prefix demands?
                    # For small N, just summing is fine.
                    
                    for p in range(1, len(ri)):
                        A = ri[:p]
                        B = ri[p:]
                        dem_A = sum(demands_arr[v] for v in A if v in cust_set)
                        dem_B = sum(demands_arr[v] for v in B if v in cust_set)
                        
                        for q in range(1, len(rj)):
                            C = rj[:q]
                            D_part = rj[q:]
                            dem_C = sum(demands_arr[v] for v in C if v in cust_set)
                            dem_D = sum(demands_arr[v] for v in D_part if v in cust_set)
                            
                            new_dem_i = dem_A + dem_D
                            new_dem_j = dem_C + dem_B
                            
                            if new_dem_i > capacity or new_dem_j > capacity:
                                continue
                                
                            new_ri = A + D_part
                            new_rj = C + B
                            
                            dist_i, feas_i, _, _ = simulate(new_ri)
                            if not feas_i:
                                continue
                                
                            dist_j, feas_j, _, _ = simulate(new_rj)
                            if not feas_j:
                                continue
                                
                            old_total = di + dj
                            new_total = dist_i + dist_j
                            
                            if new_total < old_total - 1e-6:
                                routes[i] = new_ri
                                route_dists[i] = dist_i
                                route_demands[i] = new_dem_i
                                
                                routes[j] = new_rj
                                route_dists[j] = dist_j
                                route_demands[j] = new_dem_j
                                
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final cleanup and validation
    # Ensure all customers are served
    all_served = []
    for route in routes:
        for v in route:
            if v in cust_set:
                all_served.append(v)
                
    if sorted(all_served) != sorted(customers):
        # Fallback: return individual routes if something went wrong
        # This should not happen if construction was correct.
        return [[c] for c in customers]
        
    # Return only non-empty routes
    return [r for r in routes if r]

# EVOLVE-BLOCK-END
