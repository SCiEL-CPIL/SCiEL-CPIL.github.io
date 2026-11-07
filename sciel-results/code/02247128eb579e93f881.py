# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq

def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)

    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    # Fast simulation
    def simulate_vec(route):
        if not route:
            return 0.0, True
        
        _dist = dist
        _ready = ready
        _due = due
        _service = service
        _bat_cap = battery_cap
        _speed = speed
        _rr = recharge_rate
        _cust_set = cust_set
        _station_set = station_set
        
        t = 0.0
        bat = _bat_cap
        total_d = 0.0
        cur = 0
        
        for node in route:
            d = _dist[cur, node]
            total_d += d
            bat -= d
            t += d / _speed
            
            if bat < -1e-9:
                return total_d, False
            
            if node in _cust_set:
                if t < _ready[node]:
                    t = _ready[node]
                if t > _due[node] + 1e-9:
                    return total_d, False
                t += _service[node]
            elif node in _station_set:
                t += (_bat_cap - bat) * _rr
                bat = _bat_cap
                
            cur = node
            
        d = _dist[cur, 0]
        total_d += d
        bat -= d
        t += d / _speed
        
        if bat < -1e-9:
            return total_d, False
        if t > horizon + 1e-9:
            return total_d, False
            
        return total_d, True

    def route_demand(route):
        return sum(cust_demand[v] for v in route if v in cust_set)

    def fix_battery(route):
        d, valid = simulate_vec(route)
        if valid:
            return route, d
        
        # DP: find min-time charging plan. State = (node_idx, battery_level_bucket)
        # We track earliest arrival time at each (position, battery) state.
        # Position 0 = depot start, position len(route)+1 = depot end.
        # Nodes: [0] + route + [0]
        nodes = [0] + list(route) + [0]
        n_pos = len(nodes)
        
        # Precompute distances between consecutive nodes
        # We need to decide where to insert station visits.
        # Alternative: try inserting up to 2 stations at best positions.
        
        # First try: single station insertion (fast)
        best_route = None
        best_d = float('inf')
        n = len(route)
        
        for st in stations:
            for p in range(n + 1):
                cand = route[:p] + [st] + route[p:]
                cd, cvalid = simulate_vec(cand)
                if cvalid and cd < best_d:
                    best_d = cd
                    best_route = cand
                    
        if best_route is not None:
            return best_route, best_d
        
        # Try: two station insertions
        for st1 in stations:
            for p1 in range(n + 1):
                base1 = route[:p1] + [st1] + route[p1:]
                n1 = len(base1)
                for st2 in stations:
                    for p2 in range(n1 + 1):
                        cand = base1[:p2] + [st2] + base1[p2:]
                        cd, cvalid = simulate_vec(cand)
                        if cvalid and cd < best_d:
                            best_d = cd
                            best_route = cand
        
        if best_route is not None:
            return best_route, best_d
        
        # Try: three station insertions (only if route is long)
        if n > 5:
            for st1 in stations:
                for p1 in range(n + 1):
                    base1 = route[:p1] + [st1] + route[p1:]
                    n1 = len(base1)
                    for st2 in stations:
                        for p2 in range(n1 + 1):
                            base2 = base1[:p2] + [st2] + base1[p2:]
                            n2 = len(base2)
                            for st3 in stations:
                                for p3 in range(n2 + 1):
                                    cand = base2[:p3] + [st3] + base2[p3:]
                                    cd, cvalid = simulate_vec(cand)
                                    if cvalid and cd < best_d:
                                        best_d = cd
                                        best_route = cand
        if best_route is not None:
            return best_route, best_d
        
        return None, None

    def construct_savings_clean():
        active_routes = []
        for c in customers:
            r, d = fix_battery([c])
            if r is None:
                r = [c]
            active_routes.append([r, demands[c]])
            
        improved = True
        while improved:
            improved = False
            best_save = 1e-6
            best_i, best_j = -1, -1
            best_new_route = None
            best_new_demand = 0
            
            n_ar = len(active_routes)
            for i in range(n_ar):
                r_i, d_i = active_routes[i]
                d_i_val, _ = simulate_vec(r_i)
                
                for j in range(i+1, n_ar):
                    r_j, d_j = active_routes[j]
                    
                    if d_i + d_j > capacity:
                        continue
                        
                    d_j_val, _ = simulate_vec(r_j)
                    
                    cand = r_i + r_j
                    nr, nd = fix_battery(cand)
                    if nr is not None:
                        save = d_i_val + d_j_val - nd
                        if save > best_save:
                            best_save = save
                            best_i, best_j = i, j
                            best_new_route = nr
                            best_new_demand = d_i + d_j
                            
                    cand = r_j + r_i
                    nr, nd = fix_battery(cand)
                    if nr is not None:
                        save = d_i_val + d_j_val - nd
                        if save > best_save:
                            best_save = save
                            best_i, best_j = i, j
                            best_new_route = nr
                            best_new_demand = d_i + d_j
                            
            if best_i != -1:
                active_routes.pop(best_j)
                active_routes.pop(best_i)
                active_routes.append([best_new_route, best_new_demand])
                improved = True
                if time.time() > deadline - 30.0:
                    break
                    
        return [r[0] for r in active_routes]

    def two_opt(route, max_iters=50, tol=1e-7):
        n = len(route)
        if n < 2:
            return route
        
        best_route = route
        best_d, _ = simulate_vec(route)
        
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    nd, valid = simulate_vec(new_route)
                    if valid and nd < best_d - tol:
                        best_d = nd
                        best_route = new_route
                        improved = True
        return best_route

    def relocate(routes):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 100:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(len(routes[i])):
                    node = routes[i][j]
                    if node not in cust_set:
                        continue
                    rest_i = routes[i][:j] + routes[i][j+1:]
                    d_i_rest, v_i_rest = simulate_vec(rest_i)
                    if not v_i_rest:
                        continue
                        
                    for k in range(nR):
                        rk = routes[k]
                        if route_demand(rk) + demands[node] > capacity:
                            continue
                            
                        for p in range(len(rk) + 1):
                            if k == i and (p == j or p == j+1):
                                continue
                                
                            cand_k = rk[:p] + [node] + rk[p:]
                            d_k_new, v_k_new = simulate_vec(cand_k)
                            if not v_k_new:
                                continue
                                
                            if k == i:
                                old_d_i, _ = simulate_vec(routes[i])
                                new_total = d_k_new
                                old_total = old_d_i
                            else:
                                old_d_k, _ = simulate_vec(routes[k])
                                old_total = d_i_rest + old_d_k
                                new_total = d_i_rest + d_k_new
                                
                            if new_total < old_total - 1e-7:
                                routes[k] = cand_k
                                routes[i] = rest_i
                                routes = [r for r in routes if r]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def swap(routes):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 100:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(i+1, nR):
                    for pi in range(len(routes[i])):
                        ci = routes[i][pi]
                        if ci not in cust_set: continue
                        for pj in range(len(routes[j])):
                            cj = routes[j][pj]
                            if cj not in cust_set: continue
                            
                            di = route_demand(routes[i])
                            dj = route_demand(routes[j])
                            if di - demands[ci] + demands[cj] > capacity: continue
                            if dj - demands[cj] + demands[ci] > capacity: continue
                            
                            new_i = routes[i][:]
                            new_j = routes[j][:]
                            new_i[pi] = cj
                            new_j[pj] = ci
                            
                            di_new, vi = simulate_vec(new_i)
                            dj_new, vj = simulate_vec(new_j)
                            
                            if vi and vj:
                                old_di, _ = simulate_vec(routes[i])
                                old_dj, _ = simulate_vec(routes[j])
                                if di_new + dj_new < old_di + old_dj - 1e-7:
                                    routes[i] = new_i
                                    routes[j] = new_j
                                    improved = True
                                    break
                        if improved: break
                    if improved: break
                if improved: break
        return routes

    def cross_exchange(routes, max_depth=3):
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 20:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(i+1, nR):
                    for k in range(1, max_depth+1):
                        for pi in range(len(routes[i])):
                            for pj in range(len(routes[j])):
                                if pi + k > len(routes[i]) or pj + k > len(routes[j]):
                                    continue
                                # Check if segments contain only customers
                                seg_i = routes[i][pi:pi+k]
                                seg_j = routes[j][pj:pj+k]
                                if any(n not in cust_set for n in seg_i): continue
                                if any(n not in cust_set for n in seg_j): continue
                                
                                # Check capacity
                                dem_i = route_demand(routes[i])
                                dem_j = route_demand(routes[j])
                                dem_seg_i = sum(demands[n] for n in seg_i)
                                dem_seg_j = sum(demands[n] for n in seg_j)
                                if dem_i - dem_seg_i + dem_seg_j > capacity: continue
                                if dem_j - dem_seg_j + dem_seg_i > capacity: continue
                                
                                new_i = routes[i][:pi] + seg_j + routes[i][pi+k:]
                                new_j = routes[j][:pj] + seg_i + routes[j][pj+k:]
                                
                                di_new, vi = simulate_vec(new_i)
                                dj_new, vj = simulate_vec(new_j)
                                
                                if vi and vj:
                                    old_di, _ = simulate_vec(routes[i])
                                    old_dj, _ = simulate_vec(routes[j])
                                    if di_new + dj_new < old_di + old_dj - 1e-7:
                                        routes[i] = new_i
                                        routes[j] = new_j
                                        improved = True
                                        break
                        if improved: break
                    if improved: break
                if improved: break
        return routes

    def or_opt(routes, max_seg=3):
        """Move a segment of up to max_seg customers to a different position/route."""
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for i in range(nR):
                ri = routes[i]
                for seg_len in range(1, max_seg + 1):
                    for pi in range(len(ri) - seg_len + 1):
                        seg = ri[pi:pi + seg_len]
                        if any(n not in cust_set for n in seg):
                            continue
                        seg_dem = sum(demands[n] for n in seg)
                        rest_i = ri[:pi] + ri[pi + seg_len:]
                        d_i_rest, v_i_rest = simulate_vec(rest_i)
                        if not v_i_rest:
                            continue
                        # Try inserting into each route
                        for k in range(nR):
                            rk = routes[k]
                            if k == i:
                                # capacity of rest_i already checked implicitly
                                if route_demand(rest_i) > capacity:
                                    continue
                            else:
                                if route_demand(rk) + seg_dem > capacity:
                                    continue
                            for p in range(len(rk) + 1):
                                # skip trivial same-position
                                if k == i and (p == pi or p == pi + seg_len):
                                    continue
                                cand_k = rk[:p] + seg + rk[p:]
                                d_k_new, v_k_new = simulate_vec(cand_k)
                                if not v_k_new:
                                    continue
                                if k == i:
                                    old_total = d_i_rest
                                    new_total = d_k_new
                                else:
                                    old_d_k, _ = simulate_vec(rk)
                                    old_total = d_i_rest + old_d_k
                                    new_total = d_i_rest + d_k_new
                                if new_total < old_total - 1e-7:
                                    routes[k] = cand_k
                                    routes[i] = rest_i
                                    routes = [r for r in routes if r]
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def lns(routes, n_remove, max_iters=100):
        """LNS: remove n_remove customers, re-insert via cheapest insertion."""
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Pick a random subset of customers to remove
            all_custs = [n for r in routes for n in r if n in cust_set]
            if len(all_custs) < n_remove:
                n_remove = max(1, len(all_custs) // 2)
            if n_remove <= 0:
                break
            removed = random.sample(all_custs, min(n_remove, len(all_custs)))
            rem_set = set(removed)
            # Remove from routes
            new_routes = []
            for r in routes:
                nr = [n for n in r if n not in rem_set]
                if nr:
                    d, v = simulate_vec(nr)
                    if v:
                        new_routes.append(nr)
                    else:
                        # try to fix battery
                        fr, fd = fix_battery(nr)
                        if fr:
                            new_routes.append(fr)
                        else:
                            # drop stations and retry
                            nr2 = [n for n in nr if n in cust_set]
                            fr2, _ = fix_battery(nr2)
                            if fr2:
                                new_routes.append(fr2)
            if not new_routes:
                new_routes = [[0]]  # placeholder
            # Re-insert removed customers via cheapest insertion
            for c in removed:
                best_cost = float('inf')
                best_route_idx = -1
                best_pos = -1
                for ri_idx, r in enumerate(new_routes):
                    if route_demand(r) + demands[c] > capacity:
                        continue
                    for p in range(len(r) + 1):
                        cand = r[:p] + [c] + r[p:]
                        d, v = simulate_vec(cand)
                        if not v:
                            # try fix
                            fr, fd = fix_battery(cand)
                            if fr:
                                d = fd
                                v = True
                                cand = fr
                            else:
                                continue
                        if d < best_cost:
                            best_cost = d
                            best_route_idx = ri_idx
                            best_pos = p
                            best_cand = cand
                if best_route_idx >= 0:
                    new_routes[best_route_idx] = best_cand
                else:
                    # can't insert; create new route
                    fr, _ = fix_battery([c])
                    if fr:
                        new_routes.append(fr)
                    else:
                        new_routes.append([c])
            # Compute total
            total_new = sum(simulate_vec(r)[0] for r in new_routes if r)
            total_old = sum(simulate_vec(r)[0] for r in routes)
            if total_new < total_old - 1e-7:
                routes = new_routes
                improved = True
        return routes

    def two_opt_star(routes):
        """Swap suffixes between two routes (keeping depot at front)."""
        nR = len(routes)
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for i in range(nR):
                for j in range(i+1, nR):
                    ri = routes[i]
                    rj = routes[j]
                    if len(ri) < 1 or len(rj) < 1:
                        continue
                    old_di, vi = simulate_vec(ri)
                    old_dj, vj = simulate_vec(rj)
                    if not vi or not vj:
                        continue
                    for pi in range(1, len(ri)):
                        for pj in range(1, len(rj)):
                            # new_i = ri[:pi] + rj[pj:]
                            # new_j = rj[:pj] + ri[pi:]
                            new_i = ri[:pi] + rj[pj:]
                            new_j = rj[:pj] + ri[pi:]
                            
                            # Capacity check
                            if route_demand(new_i) > capacity or route_demand(new_j) > capacity:
                                continue
                            
                            di_new, vi_new = simulate_vec(new_i)
                            dj_new, vj_new = simulate_vec(new_j)
                            
                            if vi_new and vj_new:
                                if di_new + dj_new < old_di + old_dj - 1e-7:
                                    routes[i] = new_i
                                    routes[j] = new_j
                                    improved = True
                                    break
                    if improved:
                        break
            if improved:
                break
        return routes

    # --- Main Loop ---
    solutions = []
    
    s1 = construct_savings_clean()
    if s1:
        solutions.append(s1)
        
    s2 = []
    unvisited = set(customers)
    while unvisited:
        c = min(unvisited, key=lambda x: due[x])
        route = [c]
        unvisited.remove(c)
        
        while True:
            best_c = None
            best_d = float('inf')
            for c in unvisited:
                d, valid = simulate_vec(route + [c])
                if valid and d < best_d:
                    best_d = d
                    best_c = c
            if best_c is None:
                break
            route.append(best_c)
            unvisited.remove(best_c)
            
        fr, _ = fix_battery(route)
        if fr:
            s2.append(fr)
            
    if s2:
        solutions.append(s2)
        
    for _ in range(5):
        rnd = random.Random(int(time.time()*1000))
        s3 = []
        unvisited = set(customers)
        order = list(customers)
        rnd.shuffle(order)
        for c in order:
            if c in unvisited:
                route = [c]
                unvisited.remove(c)
                while unvisited:
                    best_c = None
                    best_d = float('inf')
                    for cc in unvisited:
                        d, valid = simulate_vec(route + [cc])
                        if valid and d < best_d:
                            best_d = d
                            best_c = cc
                    if best_c is None:
                        break
                    route.append(best_c)
                    unvisited.remove(best_c)
                fr, _ = fix_battery(route)
                if fr:
                    s3.append(fr)
        if s3:
            solutions.append(s3)

    global_best = None
    global_dist = float('inf')
    
    for sol in solutions:
        if time.time() > deadline - 20.0:
            break
            
        valid_sol = []
        served = set()
        for r in sol:
            d, v = simulate_vec(r)
            if v:
                valid_sol.append(r)
                for n in r:
                    if n in cust_set:
                        served.add(n)
                        
        if len(served) != len(customers):
            continue
            
        opt_routes = valid_sol[:]
        opt_routes = [two_opt(r) for r in opt_routes]
        
        improved = True
        it = 0
        while improved and it < 20 and time.time() < deadline - 5.0:
            improved = False
            it += 1
            new_routes = relocate(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
            new_routes = swap(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
            new_routes = cross_exchange(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
            new_routes = two_opt_star(opt_routes)
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
            new_routes = [two_opt(r) for r in opt_routes]
            if new_routes != opt_routes:
                opt_routes = new_routes
                improved = True
                
        # Final high-precision polish pass
        if time.time() < deadline - 5.0:
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                changed = False
                # More aggressive 2-opt with tighter tolerance
                new_routes = []
                for r in opt_routes:
                    nr = two_opt(r, max_iters=80, tol=1e-9)
                    if nr != r:
                        changed = True
                    new_routes.append(nr)
                opt_routes = new_routes
                
                # 2-opt-star with more iterations
                new_routes = two_opt_star(opt_routes)
                if new_routes != opt_routes:
                    opt_routes = new_routes
                    changed = True
                
                # Additional relocate and swap passes for fine-tuning
                new_routes = relocate(opt_routes)
                if new_routes != opt_routes:
                    opt_routes = new_routes
                    changed = True
                new_routes = swap(opt_routes)
                if new_routes != opt_routes:
                    opt_routes = new_routes
                    changed = True
                
                if not changed:
                    break
        
        total_d = sum(simulate_vec(r)[0] for r in opt_routes)
        if total_d < global_dist:
            global_dist = total_d
            global_best = opt_routes

    if global_best is None:
        global_best = [[c] for c in customers]
        
    served = []
    valid = True
    for r in global_best:
        d, v = simulate_vec(r)
        if not v:
            valid = False
            break
        for n in r:
            if n in cust_set:
                served.append(n)
                
    if not valid or len(served) != len(customers) or len(set(served)) != len(customers):
        global_best = [[c] for c in customers]

    return global_best

# EVOLVE-BLOCK-END
