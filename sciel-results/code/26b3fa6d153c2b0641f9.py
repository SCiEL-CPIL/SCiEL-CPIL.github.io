"""EV routing with time windows – Direct Multi-Route LNS with fast delta evaluation."""
import numpy as np
import time
import math
import random

def plan(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))
    n_cust = len(customers)

    def simulate_route(route):
        """Simulate a route. Returns (feasible, total_dist, end_time, end_bat)."""
        if not route:
            return True, 0.0, 0.0, battery_cap
        t = 0.0
        bat = battery_cap
        prev = 0
        total_d = 0.0
        load = 0.0
        for v in route:
            load += demands[v]
            if load > capacity:
                return False, 0.0, 0.0, 0.0
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0, 0.0, 0.0
            t += d / speed
            total_d += d
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[v]
            prev = v
        # Return to depot
        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0, 0.0, 0.0
        t += d / speed
        total_d += d
        if t > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0
        return True, total_d, t, bat

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    def check_route_feasibility(route):
        """Fast check for feasibility only."""
        if not route:
            return True
        t = 0.0
        bat = battery_cap
        prev = 0
        load = 0.0
        for v in route:
            load += demands[v]
            if load > capacity:
                return False
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False
            t += d / speed
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False
            t += service[v]
            prev = v
        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False
        t += d / speed
        if t > horizon + 1e-9:
            return False
        return True

    def try_insert(routes, v, verbose=False):
        """Try to insert customer v into the best feasible position in any route.
        Returns (success, new_routes, delta_dist) or (False, None, None)."""
        best_delta = float('inf')
        best_r_idx = -1
        best_pos = -1
        
        for ri, r in enumerate(routes):
            # Check capacity
            if sum(demands[x] for x in r) + demands[v] > capacity:
                continue
            
            # Try all positions
            for pos in range(len(r) + 1):
                # Construct new route for this trial
                new_r = r[:pos] + [v] + r[pos:]
                if not check_route_feasibility(new_r):
                    continue
                
                # Calculate delta distance
                old_d = 0.0
                if pos == 0:
                    old_d += dist[0, r[0]] if r else 0
                else:
                    old_d += dist[r[pos-1], r[pos]]
                if pos == len(r):
                    old_d += dist[r[-1], 0] if r else 0
                else:
                    old_d += dist[r[pos], r[pos+1]]
                
                new_d = 0.0
                if pos == 0:
                    new_d += dist[0, v]
                else:
                    new_d += dist[r[pos-1], v]
                if pos == len(r):
                    new_d += dist[v, 0]
                else:
                    new_d += dist[v, r[pos]]
                
                # If route was empty, we add a new loop
                if len(r) == 0:
                    old_d = 0.0
                    new_d = dist[0, v] + dist[v, 0]
                
                delta = new_d - old_d
                if delta < best_delta:
                    best_delta = delta
                    best_r_idx = ri
                    best_pos = pos
        
        if best_r_idx == -1:
            return False, None, None
        
        # Apply insertion
        new_routes = [r[:] for r in routes]
        r = new_routes[best_r_idx]
        new_r = r[:best_pos] + [v] + r[best_pos:]
        new_routes[best_r_idx] = new_r
        
        return True, new_routes, best_delta

    def relocate(routes, v, from_idx):
        """Move customer v from route from_idx to the best position in any other route.
        Returns (success, new_routes, delta_dist)."""
        # Remove v from source route
        src_r = routes[from_idx]
        src_r_without_v = src_r[:src_r.index(v)] + src_r[src_r.index(v)+1:]
        
        # Calculate cost of removal (distance saved in source route)
        old_d = 0.0
        idx_v = src_r.index(v)
        if idx_v == 0:
            old_d += dist[0, src_r[0]]
            old_d += dist[0, src_r[1]] if len(src_r) > 1 else dist[0, 0] # Wait, if len 1, new route is empty
            # Let's just calculate full distance of modified source route to be safe and simple for delta
            # Actually, let's just calculate delta properly
            # Old: ... A - v - B ...
            # New: ... A - B ...
            # Delta_remove = dist[A,B] - (dist[A,v] + dist[v,B])
            
            if len(src_r) == 1:
                # Route becomes empty
                old_d = dist[0, v] + dist[v, 0]
                new_d_src = 0.0
            else:
                if idx_v == 0:
                    old_d = dist[0, v] + dist[v, src_r[1]]
                    new_d_src = dist[0, src_r[1]]
                elif idx_v == len(src_r) - 1:
                    old_d = dist[src_r[-2], v] + dist[v, 0]
                    new_d_src = dist[src_r[-2], 0]
                else:
                    old_d = dist[src_r[idx_v-1], v] + dist[v, src_r[idx_v+1]]
                    new_d_src = dist[src_r[idx_v-1], src_r[idx_v+1]]
        
        # Now try to insert v into other routes
        best_total_delta = float('inf')
        best_target_idx = -1
        best_pos = -1
        best_new_r = None
        
        for ti in range(len(routes)):
            if ti == from_idx:
                continue
            
            tgt_r = routes[ti]
            if sum(demands[x] for x in tgt_r) + demands[v] > capacity:
                continue
            
            for pos in range(len(tgt_r) + 1):
                # Feasibility check of new target route
                new_tgt_r = tgt_r[:pos] + [v] + tgt_r[pos:]
                if not check_route_feasibility(new_tgt_r):
                    continue
                
                # Check feasibility of source route after removal
                if not check_route_feasibility(src_r_without_v):
                    continue
                
                # Calculate delta for insertion
                if len(tgt_r) == 0:
                    ins_d = dist[0, v] + dist[v, 0]
                else:
                    if pos == 0:
                        ins_d = dist[0, v] + dist[v, tgt_r[0]] - dist[0, tgt_r[0]]
                    elif pos == len(tgt_r):
                        ins_d = dist[tgt_r[-1], v] + dist[v, 0] - dist[tgt_r[-1], 0]
                    else:
                        ins_d = dist[tgt_r[pos-1], v] + dist[v, tgt_r[pos]] - dist[tgt_r[pos-1], tgt_r[pos]]
                
                total_delta = ins_d + (new_d_src - old_d)
                
                if total_delta < best_total_delta:
                    best_total_delta = total_delta
                    best_target_idx = ti
                    best_pos = pos
                    best_new_r = new_tgt_r
        
        if best_target_idx == -1:
            return False, None, None
        
        new_routes = [r[:] for r in routes]
        new_routes[from_idx] = src_r_without_v
        new_routes[best_target_idx] = best_new_r
        
        return True, new_routes, best_total_delta

    def two_opt(routes, ri, a, b):
        """Apply 2-opt within route ri. Returns (feasible, new_route, delta)."""
        r = routes[ri]
        if b - a < 1:
            return False, r, 0.0
        
        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
        if not check_route_feasibility(new_r):
            return False, r, 0.0
        
        # Calculate delta
        old_d = 0.0
        if a == 0:
            old_d += dist[0, r[a]]
        else:
            old_d += dist[r[a-1], r[a]]
        if b == len(r) - 1:
            old_d += dist[r[b], 0]
        else:
            old_d += dist[r[b], r[b+1]]
            
        new_d = 0.0
        if a == 0:
            new_d += dist[0, r[b]]
        else:
            new_d += dist[r[a-1], r[b]]
        if b == len(r) - 1:
            new_d += dist[r[a], 0]
        else:
            new_d += dist[r[a], r[b+1]]
            
        delta = new_d - old_d
        return True, new_r, delta

    def or_opt(routes, ri, i, seg_len):
        """Move a segment of length seg_len from position i to a new position.
        Returns (feasible, new_route, delta)."""
        r = routes[ri]
        if i + seg_len > len(r):
            return False, r, 0.0
        
        # Find best insertion point for the segment
        seg = r[i:i+seg_len]
        rest = r[:i] + r[i+seg_len:]
        
        best_delta = float('inf')
        best_pos = -1
        best_new_r = None
        
        # We don't insert at the original position (no-op)
        for pos in range(len(rest) + 1):
            if pos == i or pos == i + 1:
                continue
            new_r = rest[:pos] + seg + rest[pos:]
            if not check_route_feasibility(new_r):
                continue
            
            # Delta calculation is complex for Or-opt, so we calculate full diff
            # Old distance around i
            old_d = 0.0
            if i == 0:
                old_d += dist[0, r[0]]
            else:
                old_d += dist[r[i-1], r[i]]
            if i + seg_len == len(r):
                old_d += dist[r[-1], 0]
            else:
                old_d += dist[r[i+seg_len-1], r[i+seg_len]]
                
            # New distance around pos
            new_d = 0.0
            if pos == 0:
                new_d += dist[0, new_r[0]]
            else:
                new_d += dist[new_r[pos-1], new_r[pos]]
            if pos + seg_len == len(new_r):
                new_d += dist[new_r[-1], 0]
            else:
                new_d += dist[new_r[pos+seg_len-1], new_r[pos+seg_len]]
                
            delta = new_d - old_d
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
                best_new_r = new_r
        
        if best_pos == -1:
            return False, r, 0.0
        
        return True, best_new_r, best_delta

    # Initial solution: Greedy construction
    # Sort customers by angle from depot
    angles = [math.atan2(coords[v][1], coords[v][0]) for v in customers]
    sorted_customers = [v for _, v in sorted(zip(angles, customers))]
    
    routes = []
    unassigned = sorted_customers[:]
    
    # Simple clustering by angle to create initial routes
    # Divide into ~5-7 clusters
    n_clusters = 6
    if n_cust > 0:
        cluster_size = max(1, n_cust // n_clusters)
        for c in range(n_clusters):
            start = c * cluster_size
            end = start + cluster_size if c < n_clusters - 1 else n_cust
            cluster = sorted_customers[start:end]
            if not cluster:
                continue
            
            # Nearest neighbor construction for this cluster
            current_route = []
            remaining = cluster[:]
            # Start with customer closest to depot? Or just first one
            # Let's pick the one with earliest due time to be safe
            remaining.sort(key=lambda x: due[x])
            
            while remaining:
                if not current_route:
                    # Pick first feasible
                    for v in remaining:
                        if check_route_feasibility([v]):
                            current_route.append(v)
                            remaining.remove(v)
                            break
                    else:
                        # If none feasible individually, skip (shouldn't happen)
                        break
                else:
                    best_v = None
                    best_d = float('inf')
                    for v in remaining:
                        trial = current_route + [v]
                        if check_route_feasibility(trial):
                            d = dist[current_route[-1], v]
                            if d < best_d:
                                best_d = d
                                best_v = v
                    if best_v is not None:
                        current_route.append(best_v)
                        remaining.remove(best_v)
                    else:
                        # Try to insert into best position
                        best_pos = -1
                        best_delta = float('inf')
                        for pos in range(len(current_route) + 1):
                            trial = current_route[:pos] + [v] + current_route[pos:]
                            if check_route_feasibility(trial):
                                # Approx delta
                                if pos == 0:
                                    d = dist[0, v] + dist[v, current_route[0]] - dist[0, current_route[0]]
                                elif pos == len(current_route):
                                    d = dist[current_route[-1], v] + dist[v, 0] - dist[current_route[-1], 0]
                                else:
                                    d = dist[current_route[pos-1], v] + dist[v, current_route[pos]] - dist[current_route[pos-1], current_route[pos]]
                                if d < best_delta:
                                    best_delta = d
                                    best_pos = pos
                        if best_pos != -1:
                            current_route = current_route[:best_pos] + [best_v] + current_route[best_pos:]
                            remaining.remove(best_v)
                        else:
                            break
            
            if current_route:
                routes.append(current_route)
                
    # Assign any remaining customers (should be none if logic is sound, but safety)
    all_assigned = [v for r in routes for v in r]
    remaining = [v for v in customers if v not in all_assigned]
    for v in remaining:
        success, new_routes, _ = try_insert(routes, v)
        if success:
            routes = new_routes
        else:
            # Fallback: single route
            routes.append([v])

    # Ensure all customers are served exactly once
    all_assigned = [v for r in routes for v in r]
    if set(all_assigned) != set(customers) or len(all_assigned) != n_cust:
        # Reset to single customer routes
        routes = [[v] for v in customers]

    best_routes = [r[:] for r in routes]
    best_dist = total_distance(best_routes)
    
    # LNS Loop
    T = 50.0
    alpha = 0.9998
    iters = 0
    
    while time.time() < deadline:
        # Destroy: Remove 2-4 random customers
        n_remove = random.randint(2, 4)
        if len(best_routes) == 0:
            break
        
        # Pick routes to remove from
        # Prefer longer routes for variety
        route_weights = [len(r) for r in best_routes]
        if sum(route_weights) == 0:
            break
            
        removed = []
        temp_routes = [r[:] for r in best_routes]
        
        for _ in range(n_remove):
            if not any(temp_routes):
                break
            # Weighted random selection of a route with customers
            valid_ris = [i for i, r in enumerate(temp_routes) if len(r) > 0]
            if not valid_ris:
                break
            ri = random.choice(valid_ris)
            r = temp_routes[ri]
            vi = random.randint(0, len(r) - 1)
            v = r[vi]
            r.pop(vi)
            removed.append(v)
            if not r:
                temp_routes[ri] = []
        
        # Repair: Insert removed customers back
        new_routes = temp_routes
        repair_success = True
        # Sort removed by tightest due time? Or random?
        random.shuffle(removed)
        for v in removed:
            success, new_routes, _ = try_insert(new_routes, v)
            if not success:
                repair_success = False
                break
        
        if repair_success:
            new_dist = total_distance(new_routes)
            delta = new_dist - best_dist
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                best_routes = new_routes
                best_dist = new_dist
        
        T *= alpha
        iters += 1
        
        # Local Search Passes every 100 iterations
        if iters % 100 == 0:
            # 2-opt and Or-opt on each route
            improved = False
            for ri in range(len(best_routes)):
                r = best_routes[ri]
                if len(r) < 2:
                    continue
                
                # 2-opt
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        feasible, new_r, delta = two_opt(best_routes, ri, a, b)
                        if feasible and delta < -1e-9:
                            best_routes[ri] = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # Or-opt
                for i in range(len(r)):
                    for seg_len in [1, 2]:
                        feasible, new_r, delta = or_opt(best_routes, ri, i, seg_len)
                        if feasible and delta < -1e-9:
                            best_routes[ri] = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            
            # Relocate between routes
            for ri in range(len(best_routes)):
                r = best_routes[ri]
                for v in r[:]: # Iterate over copy
                    success, new_routes, delta = relocate(best_routes, v, ri)
                    if success and delta < -1e-9:
                        best_routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                best_dist = total_distance(best_routes)

    # Final verification
    all_served = [v for r in best_routes for v in r]
    if set(all_served) != set(customers) or len(all_served) != n_cust:
        best_routes = [[v] for v in customers]
        
    # Ensure no empty routes in output? The problem says "list of routes", empty routes are probably okay but let's filter them out to be clean
    best_routes = [r for r in best_routes if r]
    
    return best_routes
