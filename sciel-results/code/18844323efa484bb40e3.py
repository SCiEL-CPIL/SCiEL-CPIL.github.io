"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D

def _route_time_and_battery(route, D, instance):
    """Return (total_time, total_dist, min_battery, feasible) for a route."""
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    t = 0.0
    bat = battery_cap
    dist = 0.0
    prev = 0
    min_bat = battery_cap
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, dist, bat, False
        min_bat = min(min_bat, bat)
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = battery_cap
        else:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, dist, bat, False
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, dist, bat, False
    min_bat = min(min_bat, bat)
    if t > horizon + 1e-9:
        return t, dist, bat, False
    return t, dist, min_bat, True

def _route_feasible(route, D, instance):
    if not route:
        return True
    cap = instance.get("capacity", 100)
    demands = instance["demands"]
    if sum(demands[v] for v in route) > cap + 1e-9:
        return False
    _, _, _, feas = _route_time_and_battery(route, D, instance)
    return feas

def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total

def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)

def _construct_greedy(instance, D, deadline):
    """Constructs a feasible solution using greedy insertion."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    # Sort by due time
    customers.sort(key=lambda v: instance["due"][v])
    
    routes = []
    unassigned = set(customers)
    battery_cap = instance.get("battery", 300)
    speed = instance.get("speed", 1)
    recharge_rate = instance.get("recharge_rate", 0.5)
    cap = instance.get("capacity", 100)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    while unassigned and time.time() < deadline:
        route = []
        prev = 0
        t = 0.0
        bat = battery_cap
        demand = 0
        
        # Find best next customer
        while unassigned:
            best_v = None
            best_cost = float('inf')
            for v in unassigned:
                d = D[prev, v]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                if demand + demands[v] > cap + 1e-9:
                    continue
                if v in stations:
                    new_t += (battery_cap - new_bat) * recharge_rate
                    new_bat = battery_cap
                else:
                    new_t = max(new_t, ready[v])
                    if new_t > due[v] + 1e-9:
                        continue
                    new_t += service[v]
                
                # Check return
                ret_d = D[v, 0]
                ret_t = new_t + ret_d / speed
                ret_bat = new_bat - ret_d
                if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                    continue
                
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
            
            if best_v is None:
                break
            
            # Add best_v
            d = D[prev, best_v]
            t += d / speed
            bat -= d
            demand += demands[best_v]
            if best_v in stations:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            else:
                t = max(t, ready[best_v])
                t += service[best_v]
            
            route.append(best_v)
            prev = best_v
            unassigned.discard(best_v)
        
        if route:
            routes.append(route)
        else:
            # Force assign if stuck (shouldn't happen with good construction)
            if unassigned:
                v = min(unassigned, key=lambda x: due[x])
                routes.append([v])
                unassigned.discard(v)
            else:
                break

    return routes

def _2opt(route, D, instance):
    if len(route) < 2:
        return route
    best = route[:]
    best_dist = _route_dist(best, D)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0
                
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                
                if new_seg >= old_seg - 1e-9:
                    continue
                
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if _route_feasible(new_route, D, instance):
                    best = new_route
                    best_dist = _route_dist(best, D)
                    improved = True
                    break
            if improved:
                break
    return best

def _or_opt(route, D, instance):
    if len(route) < 3:
        return route
    best = route[:]
    best_dist = _route_dist(best, D)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            node = best[i]
            for j in range(len(best)):
                if j == i or j == i - 1:
                    continue
                new_route = best[:i] + best[i + 1:]
                ins_j = j if i > j else j - 1
                if ins_j < 0:
                    ins_j = 0
                if ins_j > len(new_route):
                    ins_j = len(new_route)
                candidate = new_route[:ins_j] + [node] + new_route[ins_j:]
                if candidate == best:
                    continue
                if _route_feasible(candidate, D, instance):
                    new_dist = _route_dist(candidate, D)
                    if new_dist < best_dist - 1e-9:
                        best = candidate
                        best_dist = new_dist
                        improved = True
                        break
            if improved:
                break
    return best

def _inter_route_relocate(routes, D, instance):
    """Move a customer from one route to another."""
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                if not routes[i]:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:]
                    new_i.pop(ai)
                    
                    # Try inserting into route j
                    best_pos = -1
                    best_dist = float('inf')
                    old_dist_i = _route_dist(routes[i], D)
                    old_dist_j = _route_dist(routes[j], D)
                    new_dist_i = _route_dist(new_i, D)
                    
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:]
                        new_j.insert(aj, v)
                        
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        
                        new_dist_j = _route_dist(new_j, D)
                        total_new = new_dist_i + new_dist_j
                        total_old = old_dist_i + old_dist_j
                        
                        if total_new < best_dist:
                            best_dist = total_new
                            best_pos = aj
                            best_new_i = new_i
                            best_new_j = new_j
                    
                    if best_pos != -1 and best_dist < old_dist_i + old_dist_j - 1e-9:
                        routes[i] = best_new_i
                        routes[j] = best_new_j
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes

def _parallel_tempering(routes, D, instance, deadline, n_chains=10):
    """
    Parallel Tempering Simulated Annealing.
    Maintains n_chains solutions at different temperatures.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Initialize chains
    # Chain 0 is the best solution found so far
    # Other chains are perturbations
    chains = [routes]
    for _ in range(n_chains - 1):
        perturbed = [r[:] for r in routes]
        # Randomly swap a few customers
        for _ in range(3):
            if len(perturbed) > 1:
                r1, r2 = random.sample(range(len(perturbed)), 2)
                if perturbed[r1] and perturbed[r2]:
                    i1 = random.randint(0, len(perturbed[r1]) - 1)
                    i2 = random.randint(0, len(perturbed[r2]) - 1)
                    perturbed[r1][i1], perturbed[r2][i2] = perturbed[r2][i2], perturbed[r1][i1]
        chains.append(perturbed)
    
    # Temperatures
    T_min = 1.0
    T_max = 100.0
    temps = [T_max * (T_min / T_max) ** (i / (n_chains - 1)) for i in range(n_chains)]
    # Reverse so index 0 is coldest (best), index n-1 is hottest
    temps = temps[::-1]
    
    best_routes = chains[0]
    best_dist = _total_dist(best_routes, D)
    current_dists = [_total_dist(c, D) for c in chains]
    
    start_time = time.time()
    
    while time.time() < deadline:
        for i in range(n_chains):
            if time.time() >= deadline:
                break
            
            # Generate a neighbor for chain i
            new_routes = [r[:] for r in chains[i]]
            
            # Random move: 2-opt, Or-opt, or Inter-route Relocate
            move_type = random.choice(['2opt', 'oropt', 'relocate'])
            
            if move_type == '2opt':
                r_idx = random.randint(0, len(new_routes) - 1)
                route = new_routes[r_idx]
                if len(route) >= 2:
                    i1 = random.randint(0, len(route) - 2)
                    i2 = random.randint(i1 + 1, len(route) - 1)
                    new_routes[r_idx] = route[:i1] + route[i1:i2+1][::-1] + route[i2+1:]
                    
            elif move_type == 'oropt':
                r_idx = random.randint(0, len(new_routes) - 1)
                route = new_routes[r_idx]
                if len(route) >= 3:
                    i1 = random.randint(0, len(route) - 1)
                    i2 = random.randint(0, len(route) - 1)
                    if i2 == i1 or i2 == i1 - 1:
                        continue
                    node = route[i1]
                    new_route = route[:i1] + route[i1+1:]
                    ins_j = i2 if i1 > i2 else i2 - 1
                    if ins_j < 0: ins_j = 0
                    if ins_j > len(new_route): ins_j = len(new_route)
                    new_routes[r_idx] = new_route[:ins_j] + [node] + new_route[ins_j:]
                    
            else: # relocate
                if len(new_routes) >= 2:
                    r1, r2 = random.sample(range(len(new_routes)), 2)
                    if new_routes[r1]:
                        i1 = random.randint(0, len(new_routes[r1]) - 1)
                        v = new_routes[r1].pop(i1)
                        i2 = random.randint(0, len(new_routes[r2]))
                        new_routes[r2].insert(i2, v)
            
            # Check feasibility
            if not all(_route_feasible(r, D, instance) for r in new_routes):
                continue
            
            new_dist = _total_dist(new_routes, D)
            delta = new_dist - current_dists[i]
            
            # Metropolis acceptance
            if delta < 0 or random.random() < math.exp(-delta / temps[i]):
                chains[i] = new_routes
                current_dists[i] = new_dist
                
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_routes = [r[:] for r in new_routes]
            
            # Exchange with neighbor (i+1)
            if i < n_chains - 1:
                # Chain i is colder (lower index in temps list after reversal? No, temps[0] is coldest)
                # Wait, temps = [T_max ... T_min] reversed -> [T_min ... T_max]
                # So index 0 is coldest.
                # We exchange between i and i+1.
                T_exchange = temps[i+1] # Use higher temperature for acceptance?
                # Standard PTSA: accept swap if E_i < E_j or with prob exp(-(E_i - E_j)/T_j)
                # Here i is colder, i+1 is hotter.
                # We want to move good solutions to cold, bad to hot.
                # If cold (i) is better than hot (i+1), swap is good.
                # If hot (i+1) is better than cold (i), swap with prob.
                
                if current_dists[i] < current_dists[i+1]:
                    # Cold is better, swap to keep it cold? No, usually we swap if the higher temp chain has a better solution.
                    # Let's use standard: accept swap with prob 1 if E_cold < E_hot, else exp(-(E_cold - E_hot)/T_hot)
                    # Actually, simpler: swap if it improves the "energy" of the system or with prob.
                    # Let's try: if dist[i] < dist[i+1], swap is beneficial for cooling.
                    # If dist[i] > dist[i+1], hot chain found a better solution, move it to cold.
                    
                    # Let's just swap if dist[i+1] < dist[i] (hot found better)
                    # Or use Metropolis on the swap.
                    # Swap probability: 1 if dist[i] < dist[i+1], else exp(-(dist[i] - dist[i+1])/temps[i+1])
                    if dist_i < dist_j: # Pseudocode, need actual vars
                        pass
                    
                    # Correct logic:
                    # Let A be chain i (cold), B be chain i+1 (hot).
                    # Accept swap if E_A < E_B or random < exp(-(E_A - E_B)/T_B)
                    # This ensures good solutions move to cold, bad to hot.
                    
                    if current_dists[i] < current_dists[i+1]:
                        # A is better. Keep A at i, B at i+1. No swap needed?
                        # Actually, we want to *try* to move B's diversity to A?
                        # Standard PTSA: swap if E_i < E_j (A better than B) is NOT the criterion.
                        # Criterion: Accept swap if E_i < E_j? No.
                        # If we swap, A gets B's solution, B gets A's.
                        # We accept if the new configuration is "better" in terms of annealing.
                        # Usually: Accept if E_i < E_j is false?
                        # Let's use: Accept swap with probability 1 if E_i < E_j? No, that keeps them sorted.
                        # We want to *randomize* order to escape local optima.
                        # Standard: Accept swap if E_i < E_j is NOT required.
                        # Accept swap if random < exp(-(E_i - E_j)/T_j) where T_j is the *higher* temperature.
                        # If E_i < E_j (Cold better), delta = E_i - E_j < 0. exp(pos) > 1. Always accept.
                        # If E_i > E_j (Hot better), delta > 0. exp(neg) < 1. Sometimes accept.
                        # This moves good solutions to cold, bad to hot.
                        
                        delta_swap = current_dists[i] - current_dists[i+1]
                        if delta_swap < 0 or random.random() < math.exp(-delta_swap / temps[i+1]):
                            # Swap
                            chains[i], chains[i+1] = chains[i+1], chains[i]
                            current_dists[i], current_dists[i+1] = current_dists[i+1], current_dists[i]
    
    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    coords = instance["coords"]
    D = _dist_matrix(coords)
    
    # 1. Construction
    routes = _construct_greedy(instance, D, deadline)
    
    # 2. Local Search
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, instance)
        routes[r_idx] = _or_opt(routes[r_idx], D, instance)
    
    routes = _inter_route_relocate(routes, D, instance)
    
    # 3. Parallel Tempering
    if time.time() < deadline - 10:
        routes = _parallel_tempering(routes, D, instance, deadline - 5, n_chains=10)
    
    return routes

# EVOLVE-BLOCK-END
