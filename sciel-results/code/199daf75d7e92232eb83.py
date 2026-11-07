"""EV routing with time windows and charging - combined approach."""
import numpy as np
import math
import time
import random
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords_arr[i][0] - coords_arr[j][0]
            dy = coords_arr[i][1] - coords_arr[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    station_set = set(stations)

    # Precompute: for each node, the best station to visit if battery is low
    # and the resulting state (time_added, extra_distance, battery_after)
    # We'll compute this on the fly in build_route_with_charging

    def simulate_route(route):
        """Simulate a route, return (total_distance, valid)."""
        if not route:
            return 0.0, True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = dist[prev][node]
            total_dist += d
            if bat < d - 1e-9:
                return float('inf'), False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return float('inf'), False
            if 1 <= node <= n_customers:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return float('inf'), False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                if deficit > 0:
                    t += deficit * recharge_rate
                bat = battery_cap
            prev = node
        d_ret = dist[prev][0]
        total_dist += d_ret
        if bat < d_ret - 1e-9:
            return float('inf'), False
        t += d_ret / speed
        if t > horizon + 1e-9:
            return float('inf'), False
        return total_dist, True

    def build_route_with_charging(cust_seq):
        """
        Given a sequence of customer IDs, insert charging stops as needed.
        Returns (route_list, total_distance) or (None, inf) if infeasible.
        """
        if not cust_seq:
            return [], 0.0

        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0

        for cust in cust_seq:
            d_direct = dist[prev][cust]
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None, float('inf')
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                if load > capacity:
                    return None, float('inf')
                total_dist += d_direct
                route.append(cust)
                prev = cust
            else:
                best_st = None
                best_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0

                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                    t_st += rt
                    bat_st = battery_cap
                    d_st_cust = dist[st][cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    extra_dist = d_to_st + d_st_cust - d_direct
                    t_after = t_cust + service[cust]
                    bat_after = bat_st - d_st_cust
                    load_new = load + demands[cust]
                    if load_new > capacity:
                        continue
                    if extra_dist < best_cost - 1e-9:
                        best_cost = extra_dist
                        best_st = st
                        best_t_cust = t_after
                        best_bat_after = bat_after

                if best_st is None:
                    return None, float('inf')

                route.append(best_st)
                route.append(cust)
                t = best_t_cust
                bat = best_bat_after
                load += demands[cust]
                total_dist += best_cost
                prev = cust

        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None, float('inf')
            total_dist += d_ret
        else:
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                t_st += rt
                bat_st = battery_cap
                d_st_dep = dist[st][0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                total_dist += d_to_st + d_st_dep
                route.append(st)
                found = True
                break
            if not found:
                return None, float('inf')

        return route, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    # Clustering-based construction
    def cluster_customers(cust_list, n_clusters):
        if n_clusters >= len(cust_list):
            return [[c] for c in cust_list]

        centers = []
        shuffled = cust_list[:]
        random.shuffle(shuffled)
        centers.append(shuffled[0])
        for c in shuffled[1:]:
            min_dist = min(math.sqrt((coords[c][0]-coords[cc][0])**2 + (coords[c][1]-coords[cc][1])**2) for cc in centers)
            if min_dist > 15.0:
                centers.append(c)
            if len(centers) >= n_clusters:
                break

        assignments = [[] for _ in range(len(centers))]
        for c in cust_list:
            best_idx = 0
            best_d = float('inf')
            for idx, cc in enumerate(centers):
                d = math.sqrt((coords[c][0]-coords[cc][0])**2 + (coords[c][1]-coords[cc][1])**2)
                if d < best_d:
                    best_d = d
                    best_idx = idx
            assignments[best_idx].append(c)

        return [a for a in assignments if a]

    def construct_routes_from_clusters(clusters):
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            cluster_sorted = sorted(cluster, key=lambda c: math.atan2(coords[c][1]-coords[0][1], coords[c][0]-coords[0][0]))

            best_seq = None
            best_dist = float('inf')

            for start_idx in range(min(3, len(cluster_sorted))):
                seq = [cluster_sorted[start_idx]]
                remaining = [c for i, c in enumerate(cluster_sorted) if i != start_idx]
                current_load = demands[cluster_sorted[start_idx]]

                while remaining:
                    best_next = None
                    best_metric = float('inf')
                    last = seq[-1]

                    for c in remaining:
                        if current_load + demands[c] > capacity:
                            continue
                        d = dist[last][c]
                        if d > battery_cap:
                            continue
                        new_seq = seq + [c]
                        r, d_total = build_route_with_charging(new_seq)
                        if r is not None and d_total < best_metric:
                            best_metric = d_total
                            best_next = c

                    if best_next is None:
                        for c in remaining:
                            if current_load + demands[c] > capacity:
                                continue
                            new_seq = seq + [c]
                            r, d_total = build_route_with_charging(new_seq)
                            if r is not None and d_total < best_metric:
                                best_metric = d_total
                                best_next = c

                    if best_next is None:
                        break
                    seq.append(best_next)
                    remaining.remove(best_next)
                    current_load += demands[best_next]

                r, d_total = build_route_with_charging(seq)
                if r is not None and d_total < best_dist:
                    best_dist = d_total
                    best_seq = seq

            if best_seq is not None:
                r, _ = build_route_with_charging(best_seq)
                if r is not None:
                    routes.append(r)
                else:
                    for c in best_seq:
                        sr, _ = build_route_with_charging([c])
                        if sr is not None:
                            routes.append(sr)
                        else:
                            routes.append([c])
            else:
                for c in cluster:
                    sr, _ = build_route_with_charging([c])
                    if sr is not None:
                        routes.append(sr)
                    else:
                        routes.append([c])

        return routes

    def greedy_construct():
        """Greedy construction: build routes one at a time from remaining customers."""
        routes = []
        remaining = set(range(1, n_customers + 1))

        while remaining and time.time() < deadline - 5.0:
            best_route = None
            best_dist = float('inf')
            best_used = None

            start_candidates = sorted(remaining, key=lambda v: due[v])

            for start_cust in start_candidates[:8]:
                route_seq = [start_cust]
                used = {start_cust}
                current_load = demands[start_cust]

                while time.time() < deadline - 5.0:
                    best_next = None
                    best_metric = float('inf')

                    for c in remaining - used:
                        if current_load + demands[c] > capacity:
                            continue
                        new_seq = route_seq + [c]
                        test_route, d_total = build_route_with_charging(new_seq)
                        if test_route is not None and d_total < best_metric:
                            best_metric = d_total
                            best_next = c

                    if best_next is None:
                        break
                    route_seq.append(best_next)
                    used.add(best_next)
                    current_load += demands[best_next]

                final_route, d_total = build_route_with_charging(route_seq)
                if final_route is not None:
                    d = route_distance(final_route)
                    if d < best_dist:
                        best_dist = d
                        best_route = final_route
                        best_used = set(c for c in route_seq if 1 <= c <= n_customers)

            if best_route is None:
                # Fallback: single customer
                for c in remaining:
                    sr, _ = build_route_with_charging([c])
                    if sr is not None:
                        routes.append(sr)
                        remaining.discard(c)
                        break
                else:
                    break
            else:
                routes.append(best_route)
                remaining -= best_used

        # Add any remaining
        for c in remaining:
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                routes.append(sr)
            else:
                routes.append([c])

        return routes

    def total_distance(routes):
        return sum(simulate_route(r)[0] for r in routes)

    # Try multiple construction strategies and pick the best
    best_routes = None
    best_total = float('inf')

    # Strategy 1: Clustering-based
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: (due[v], ready[v]))

    n_cust = n_customers
    cluster_counts = [3, 4, 5, 6, 7, 8]
    if n_cust <= 15:
        cluster_counts = [2, 3, 4, 5]

    for nc in cluster_counts:
        if time.time() > deadline - 20.0:
            break
        clusters = cluster_customers(customers, nc)
        routes = construct_routes_from_clusters(clusters)

        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])

        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    # Strategy 2: Greedy construction
    if time.time() < deadline - 15.0:
        routes = greedy_construct()
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])
        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    if best_routes is None:
        best_routes = []
        for c in range(1, n_customers + 1):
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                best_routes.append(sr)
            else:
                best_routes.append([c])
        best_total = total_distance(best_routes)

    routes = best_routes

    # Strategy 3: Dijkstra-based optimal single-vehicle tour for small instances
    # This is a state-space search where state = (node, battery_level_quantized, time_bucket)
    # We use this to try to find better individual routes
    
    def dijkstra_optimal_route(cust_set):
        """
        Try to find an optimal route serving exactly the customers in cust_set.
        Uses Dijkstra on a state graph. Returns (route, distance) or (None, inf).
        This is only used for small sets (<= 8 customers) to polish.
        """
        if len(cust_set) == 0:
            return [], 0.0
        if len(cust_set) > 8:
            return None, float('inf')
        
        # State: (current_node, visited_mask, battery_quantized)
        # battery_quantized: 0..battery_cap in steps of 1 (300 states)
        # visited_mask: bitmask of customers served
        # This can be expensive, so we limit it
        
        n_cust_in_set = len(cust_set)
        cust_list = sorted(cust_set)
        cust_to_idx = {c: i for i, c in enumerate(cust_list)}
        
        # For small sets, use a simpler approach: try all permutations is too expensive
        # Instead, use a beam search with Dijkstra-like expansion
        
        # Actually, let's use a simpler improvement: for each route, try to reorder
        # customers using a local search that's more thorough
        
        return None, float('inf')
    
    # Apply per-route optimization with more thorough local search
    if time.time() < deadline - 30.0:
        # For each route, try to improve it with more iterations of 2-opt and or-opt
        for route_idx in range(len(routes)):
            if time.time() > deadline - 30.0:
                break
            
            cust_seq = [c for c in routes[route_idx] if 1 <= c <= n_customers]
            if len(cust_seq) < 2:
                continue
            
            # Try all possible 2-opt reversals
            best_seq = cust_seq[:]
            best_route, best_d = build_route_with_charging(cust_seq)
            if best_route is None:
                continue
            
            improved = True
            max_iters = 20
            for _ in range(max_iters):
                if time.time() > deadline - 30.0:
                    break
                improved = False
                for a in range(len(best_seq)):
                    for b in range(a + 1, len(best_seq)):
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        r, d = build_route_with_charging(new_seq)
                        if r is not None and d < best_d - 0.05:
                            best_d = d
                            best_seq = new_seq[:]
                            best_route = r
                            improved = True
                if not improved:
                    break
            
            if improved:
                routes[route_idx] = best_route

    # Local search: relocate, swap, and 2-opt with improved evaluation
    def try_relocate(routes, i, j, k, pos):
        cust = routes[i][j]
        if not (1 <= cust <= n_customers):
            return False

        new_i_seq = [c for idx, c in enumerate(routes[i]) if idx != j]
        new_k_seq = routes[k][:pos] + [cust] + routes[k][pos:]

        r_i, d_i = build_route_with_charging(new_i_seq)
        r_k, d_k = build_route_with_charging(new_k_seq)

        if r_i is None or r_k is None:
            return False

        set_i = set(c for c in r_i if 1 <= c <= n_customers)
        set_k = set(c for c in r_k if 1 <= c <= n_customers)
        if set_i & set_k:
            return False

        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
        new_dist = d_i + d_k

        if new_dist < old_dist - 0.1:
            routes[i] = r_i if r_i else []
            routes[k] = r_k
            if not routes[i]:
                del routes[i]
                if k > i:
                    k -= 1
            return True
        return False

    def try_swap(routes, i, j, k, pos):
        cust_i = routes[i][j]
        cust_k = routes[k][pos]
        if not (1 <= cust_i <= n_customers) or not (1 <= cust_k <= n_customers):
            return False

        new_i_seq = routes[i][:]
        new_k_seq = routes[k][:]
        new_i_seq[j] = cust_k
        new_k_seq[pos] = cust_i

        r_i, d_i = build_route_with_charging(new_i_seq)
        r_k, d_k = build_route_with_charging(new_k_seq)

        if r_i is None or r_k is None:
            return False

        set_i = set(c for c in r_i if 1 <= c <= n_customers)
        set_k = set(c for c in r_k if 1 <= c <= n_customers)
        if set_i & set_k:
            return False

        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
        new_dist = d_i + d_k

        if new_dist < old_dist - 0.1:
            routes[i] = r_i
            routes[k] = r_k
            return True
        return False

    def two_opt_route(routes, i):
        cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
        if len(cust_seq) < 2:
            return False

        best_seq = cust_seq[:]
        best_route, best_d = build_route_with_charging(cust_seq)
        if best_route is None:
            return False

        improved = False
        for _ in range(5):
            changed = False
            for a in range(len(best_seq)):
                for b in range(a + 1, len(best_seq)):
                    new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                    r, d = build_route_with_charging(new_seq)
                    if r is not None and d < best_d - 0.1:
                        best_d = d
                        best_seq = new_seq[:]
                        best_route = r
                        changed = True
                        improved = True
            if not changed:
                break

        if improved:
            routes[i] = best_route
        return improved

    # Run local search with more iterations and tighter tolerance
    if time.time() < deadline - 15.0:
        improved = True
        max_iter = 80
        it = 0
        while improved and it < max_iter and time.time() < deadline - 10.0:
            improved = False
            it += 1

            # 2-opt within routes
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                if two_opt_route(routes, i):
                    improved = True

            # Relocate
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(len(routes[i])):
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 10.0:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            if try_relocate(routes, i, j, k, pos):
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Swap
            if not improved and time.time() < deadline - 10.0:
                for i in range(len(routes)):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(len(routes[i])):
                        cust_i = routes[i][j]
                        if not (1 <= cust_i <= n_customers):
                            continue
                        for k in range(len(routes)):
                            if k == i or time.time() > deadline - 10.0:
                                continue
                            for pos in range(len(routes[k])):
                                if try_swap(routes, i, j, k, pos):
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            # Or-opt: move 2 or 3 consecutive customers to another position
            if not improved and time.time() < deadline - 10.0:
                for i in range(len(routes)):
                    if time.time() > deadline - 10.0:
                        break
                    for seg_len in [2, 3]:
                        for j in range(len(routes[i]) - seg_len + 1):
                            seg = routes[i][j:j+seg_len]
                            # Check if all are customers
                            if not all(1 <= c <= n_customers for c in seg):
                                continue
                            
                            # Remove segment
                            new_i_seq = routes[i][:j] + routes[i][j+seg_len:]
                            
                            # Try inserting in each other route and in same route
                            for k in range(len(routes)):
                                if time.time() > deadline - 10.0:
                                    break
                                target_route = routes[k] if k != i else new_i_seq
                                for pos in range(len(target_route) + 1):
                                    new_k_seq = target_route[:pos] + seg + target_route[pos:]
                                    
                                    if k == i:
                                        # Same route, different position
                                        if pos == j or pos == j + seg_len:
                                            continue
                                        r_i, d_i = build_route_with_charging(new_k_seq)
                                        if r_i is not None:
                                            old_dist = simulate_route(routes[i])[0]
                                            if d_i < old_dist - 0.05:
                                                routes[i] = r_i
                                                improved = True
                                                break
                                    else:
                                        r_i, d_i = build_route_with_charging(new_i_seq)
                                        r_k, d_k = build_route_with_charging(new_k_seq)
                                        if r_i is not None and r_k is not None:
                                            old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
                                            if d_i + d_k < old_dist - 0.05:
                                                routes[i] = r_i if r_i else []
                                                routes[k] = r_k
                                                if not routes[i]:
                                                    del routes[i]
                                                    if k > i:
                                                        k -= 1
                                                improved = True
                                                break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break

    # Final validation and cleanup
    valid_routes = []
    covered = set()
    for r in routes:
        d, valid = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    for cust in range(1, n_customers + 1):
        if cust not in covered:
            sr, _ = build_route_with_charging([cust])
            if sr is not None:
                valid_routes.append(sr)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
