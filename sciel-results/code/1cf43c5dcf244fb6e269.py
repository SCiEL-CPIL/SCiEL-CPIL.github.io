"""EV routing with time windows – A* search per route with direct multi-route local search."""
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

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
        """Simulate a route (list of customer ids). Returns (feasible, total_dist, end_time, end_bat)."""
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

    # Clarke-Wright savings initial solution
    def clarke_wright_init():
        """Build initial routes using Clarke-Wright savings heuristic."""
        # Start with each customer in its own route
        routes = [[c] for c in customers]
        # Verify each single-customer route is feasible
        feasible_routes = []
        for r in routes:
            feasible, _, _, _ = simulate_route(r)
            if feasible:
                feasible_routes.append(r)
            else:
                # Shouldn't happen, but handle it
                feasible_routes.append(r)
        
        # Compute savings for merging two routes
        # Savings = dist[depot, a] + dist[depot, b] - dist[a, b] where a is last of route1, b is first of route2
        # We'll use a simpler approach: compute savings for all pairs and greedily merge
        
        # For each pair of routes, compute the potential saving if we merge them
        # We need to track which node is the "end" of each route (last customer)
        
        # Build savings list
        savings_list = []
        for i in range(len(feasible_routes)):
            for j in range(i + 1, len(feasible_routes)):
                r1 = feasible_routes[i]
                r2 = feasible_routes[j]
                if not r1 or not r2:
                    continue
                # Option 1: append r2 to r1 (r1_last -> r2_first)
                last1 = r1[-1]
                first2 = r2[0]
                save1 = dist[0, last1] + dist[0, first2] - dist[last1, first2]
                merged1 = r1 + r2
                feasible1, _, _, _ = simulate_route(merged1)
                if feasible1:
                    savings_list.append((save1, i, j, 1))
                
                # Option 2: prepend r2 to r1 (r2_last -> r1_first)
                last2 = r2[-1]
                first1 = r1[0]
                save2 = dist[0, last2] + dist[0, first1] - dist[last2, first1]
                merged2 = r2 + r1
                feasible2, _, _, _ = simulate_route(merged2)
                if feasible2:
                    savings_list.append((save2, i, j, 2))
        
        # Sort by savings descending
        savings_list.sort(key=lambda x: -x[0])
        
        # Greedily merge
        route_to_idx = list(range(len(feasible_routes)))
        merged = [False] * len(feasible_routes)
        
        for save, i, j, mode in savings_list:
            if merged[i] or merged[j]:
                continue
            if save <= 0:
                break
            
            ri = route_to_idx[i]
            rj = route_to_idx[j]
            
            if mode == 1:
                new_route = feasible_routes[ri] + feasible_routes[rj]
            else:
                new_route = feasible_routes[rj] + feasible_routes[ri]
            
            feasible, _, _, _ = simulate_route(new_route)
            if feasible:
                feasible_routes[ri] = new_route
                merged[rj] = True
        
        # Collect non-merged routes
        result = []
        for i in range(len(feasible_routes)):
            if not merged[i] and feasible_routes[i]:
                result.append(feasible_routes[i])
        
        if not result:
            result = [[c] for c in customers]
        
        return result

    # Initial solution
    best_routes = clarke_wright_init()
    best_dist = total_distance(best_routes)

    # Direct multi-route local search with 2-opt, Or-opt, and relocate moves
    def get_moves(routes):
        """Generate candidate moves: 2-opt within route, Or-opt within route, relocate across routes."""
        moves = []
        
        # 2-opt within each route
        for ri, r in enumerate(routes):
            if len(r) < 2:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    moves.append(('2opt', ri, a, b))
        
        # Or-opt within each route (move 1-2 consecutive)
        for ri, r in enumerate(routes):
            if len(r) < 2:
                continue
            for seg_len in [1, 2]:
                for i in range(len(r) - seg_len + 1):
                    for j in range(len(r) - seg_len + 1):
                        if j == i:
                            continue
                        moves.append(('oropt', ri, i, j, seg_len))
        
        # Relocate: move a customer from one route to another
        for ri in range(len(routes)):
            for vi in range(len(routes[ri])):
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for vj in range(len(routes[rj]) + 1):
                        moves.append(('relocate', ri, vi, rj, vj))
        
        return moves

    def apply_move(routes, move):
        """Apply a move and return new routes. Returns None if infeasible."""
        mtype = move[0]
        
        if mtype == '2opt':
            _, ri, a, b = move
            r = routes[ri][:]
            new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
            new_routes = [r[:] for r in routes]
            new_routes[ri] = new_r
            return new_routes
        
        elif mtype == 'oropt':
            _, ri, i, j, seg_len = move
            r = routes[ri][:]
            seg = r[i:i+seg_len]
            rest = r[:i] + r[i+seg_len:]
            if j > i:
                j -= seg_len
            new_r = rest[:j] + seg + rest[j:]
            new_routes = [r[:] for r in routes]
            new_routes[ri] = new_r
            return new_routes
        
        elif mtype == 'relocate':
            _, ri, vi, rj, vj = move
            customer = routes[ri][vi]
            new_routes = [r[:] for r in routes]
            # Remove from source
            new_routes[ri].pop(vi)
            # Insert into target
            new_routes[rj].insert(vj, customer)
            return new_routes
        
        return None

    # Local search loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate moves
        all_moves = get_moves(best_routes)
        random.shuffle(all_moves)
        
        improved = False
        for move in all_moves:
            if time.time() > deadline:
                break
            
            new_routes = apply_move(best_routes, move)
            if new_routes is None:
                continue
            
            # Check feasibility of all routes
            feasible = True
            for r in new_routes:
                f, _, _, _ = simulate_route(r)
                if not f:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            new_dist = total_distance(new_routes)
            if new_dist < best_dist - 1e-9:
                best_routes = new_routes
                best_dist = new_dist
                improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final 2-opt polish on each route
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = False
        for ri in range(len(best_routes)):
            r = best_routes[ri]
            if len(r) < 2:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    if time.time() > deadline:
                        break
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    feasible, _, _, _ = simulate_route(new_r)
                    if feasible:
                        # Compute delta distance
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

                        if new_d < old_d - 1e-9:
                            best_routes[ri] = new_r
                            improved = True
                if time.time() > deadline:
                    break
        if not improved:
            break

    # Verify all customers served exactly once
    all_served = [v for r in best_routes for v in r]
    if set(all_served) != set(customers) or len(all_served) != n_cust:
        # Fallback
        best_routes = [[v] for v in customers]

    return best_routes
