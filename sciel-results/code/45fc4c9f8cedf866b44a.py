"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _euclidean(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            d[i, j] = dist
            d[j, i] = dist
    return d


def _build_dist_matrix(instance):
    coords = instance["coords"]
    return _euclidean(coords)


def _route_cost(route, dist, instance):
    """
    Return (total_distance, is_feasible, total_time, final_battery) for a route.
    route is a list of node ids (customers/stations), depot implicit at ends.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_dist = 0.0
    total_demand = 0
    time_cur = 0.0
    battery = battery_max

    # Start at depot (node 0)
    prev = 0
    for node in route:
        d = dist[prev][node]
        total_dist += d
        time_cur += d / speed
        battery -= d

        if battery < -1e-9:
            return total_dist, False, time_cur, battery

        if node in stations:
            # Recharge: time added = (battery_max - battery) * recharge_rate
            # But battery is current charge before recharge. The rule says:
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # I think "energy" here means the current battery level? Or the deficit?
            # "refills" means battery becomes battery_max.
            # The time added is (battery_max - battery) * recharge_rate? 
            # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # This is ambiguous. Usually recharge time = (max - current) / rate.
            # Here rate is recharge_rate. If recharge_rate is energy per time, then time = (max-bat)/rate.
            # If recharge_rate is time per energy, then time = (max-bat)*rate.
            # The variable name is recharge_rate = 0.5.
            # Let's assume time = (battery_max - battery) * recharge_rate.
            recharge_time = (battery_max - battery) * recharge_rate
            time_cur += recharge_time
            battery = battery_max
        else:
            # Customer
            total_demand += demands[node]
            if total_demand > capacity:
                return total_dist, False, time_cur, battery

            if time_cur < ready[node]:
                time_cur = ready[node]

            if time_cur > due[node] + 1e-9:
                return total_dist, False, time_cur, battery

            time_cur += service[node]

        prev = node

    # Return to depot
    d = dist[prev][0]
    total_dist += d
    time_cur += d / speed
    battery -= d

    if battery < -1e-9:
        return total_dist, False, time_cur, battery

    if time_cur > horizon + 1e-9:
        return total_dist, False, time_cur, battery

    return total_dist, True, time_cur, battery


def _evaluate_routes(routes, dist, instance):
    """
    Return (total_distance, is_feasible) for a list of routes.
    """
    total_dist = 0.0
    for route in routes:
        d, feasible, _, _ = _route_cost(route, dist, instance)
        if not feasible:
            return float('inf'), False
        total_dist += d
    return total_dist, True


def _construct_initial_solution(instance, dist):
    """
    Greedy construction: sort customers by due time, insert into best feasible position.
    """
    customers = list(range(1, instance["n_customers"] + 1))
    customers.sort(key=lambda v: instance["due"][v])

    routes = []
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')

        # Try inserting into each existing route
        for r_idx, route in enumerate(routes):
            # Try each position in the route
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                d, feasible, _, _ = _route_cost(new_route, dist, instance)
                if feasible and d < best_cost:
                    best_cost = d
                    best_route_idx = r_idx
                    best_pos = pos

        # Try creating a new route
        d, feasible, _, _ = _route_cost([cust], dist, instance)
        if feasible and d < best_cost:
            best_cost = d
            best_route_idx = -2  # sentinel for new route
            best_pos = 0

        if best_route_idx == -2:
            routes.append([cust])
        elif best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, cust)
        else:
            # Should not happen if single customer is always feasible
            routes.append([cust])

    return routes


def _local_search(routes, dist, instance, max_iters=5000, time_limit=None):
    """
    2-opt and relocate local search.
    """
    start_time = time.time()
    current_routes = [r[:] for r in routes]
    current_cost, feasible = _evaluate_routes(current_routes, dist, instance)
    if not feasible:
        return current_routes, float('inf')

    n_customers = instance["n_customers"]
    customers = set(range(1, n_customers + 1))

    # Precompute some things for speed? Maybe not necessary for small n.

    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        improved = False

        # Relocate moves
        for r_idx in range(len(current_routes)):
            route = current_routes[r_idx]
            for pos in range(len(route)):
                cust = route[pos]
                # Remove cust
                new_route_r = route[:pos] + route[pos + 1:]
                # Try inserting into other routes
                for t_idx in range(len(current_routes)):
                    if t_idx == r_idx:
                        # Try reinserting into same route at different pos
                        for ins_pos in range(len(new_route_r) + 1):
                            if ins_pos == pos:
                                continue
                            new_route_r_ins = new_route_r[:ins_pos] + [cust] + new_route_r[ins_pos:]
                            # Check feasibility of new_route_r_ins
                            d_new, feasible_new, _, _ = _route_cost(new_route_r_ins, dist, instance)
                            if feasible_new:
                                # Check if total cost improves
                                # This is expensive to evaluate all routes.
                                # Approximation: only check if the changed route improves?
                                # No, must be exact.
                                # Let's just evaluate the whole set. It's small (25 customers).
                                test_routes = [r[:] for r in current_routes]
                                test_routes[r_idx] = new_route_r_ins
                                test_cost, test_feasible = _evaluate_routes(test_routes, dist, instance)
                                if test_feasible and test_cost < current_cost - 1e-9:
                                    current_routes[r_idx] = new_route_r_ins
                                    current_cost = test_cost
                                    improved = True
                                    break
                        if improved:
                            break
                    else:
                        t_route = current_routes[t_idx]
                        for ins_pos in range(len(t_route) + 1):
                            new_t_route = t_route[:ins_pos] + [cust] + t_route[ins_pos:]
                            # Check feasibility of new_t_route and new_route_r
                            d_t, feas_t, _, _ = _route_cost(new_t_route, dist, instance)
                            d_r, feas_r, _, _ = _route_cost(new_route_r, dist, instance)
                            if feas_t and feas_r:
                                test_routes = [r[:] for r in current_routes]
                                test_routes[r_idx] = new_route_r
                                test_routes[t_idx] = new_t_route
                                test_cost, test_feasible = _evaluate_routes(test_routes, dist, instance)
                                if test_feasible and test_cost < current_cost - 1e-9:
                                    current_routes[r_idx] = new_route_r
                                    current_routes[t_idx] = new_t_route
                                    current_cost = test_cost
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

        if not improved:
            break

    return current_routes, current_cost


def _simulated_annealing(routes, dist, instance, time_limit=5.0):
    """
    Simple simulated annealing with relocate and 2-opt moves.
    """
    start_time = time.time()
    current_routes = [r[:] for r in routes]
    current_cost, feasible = _evaluate_routes(current_routes, dist, instance)
    if not feasible:
        return current_routes, float('inf')

    best_routes = [r[:] for r in current_routes]
    best_cost = current_cost

    T = 50.0
    T_min = 0.01
    cooling = 0.995

    n_customers = instance["n_customers"]

    while T > T_min:
        if (time.time() - start_time) > time_limit:
            break

        # Generate a random move
        move_type = random.random()
        if move_type < 0.5:
            # Relocate
            r_idx = random.randint(0, len(current_routes) - 1)
            route = current_routes[r_idx]
            if not route:
                T *= cooling
                continue
            pos = random.randint(0, len(route) - 1)
            cust = route[pos]
            new_route_r = route[:pos] + route[pos + 1:]

            if len(current_routes) > 1:
                t_idx = random.randint(0, len(current_routes) - 1)
                if t_idx == r_idx:
                    t_idx = (t_idx + 1) % len(current_routes)
            else:
                t_idx = r_idx

            t_route = current_routes[t_idx]
            ins_pos = random.randint(0, len(t_route))
            new_t_route = t_route[:ins_pos] + [cust] + t_route[ins_pos:]

            test_routes = [r[:] for r in current_routes]
            test_routes[r_idx] = new_route_r
            test_routes[t_idx] = new_t_route
        else:
            # 2-opt within a route
            r_idx = random.randint(0, len(current_routes) - 1)
            route = current_routes[r_idx]
            if len(route) < 2:
                T *= cooling
                continue
            i, j = sorted(random.sample(range(len(route)), 2))
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            test_routes = [r[:] for r in current_routes]
            test_routes[r_idx] = new_route

        new_cost, new_feasible = _evaluate_routes(test_routes, dist, instance)

        if new_feasible:
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_routes = test_routes
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_routes = [r[:] for r in current_routes]
        else:
            # If infeasible, we can still accept with high probability if T is high?
            # But infeasible routes have cost inf.
            # Let's treat infeasible as a penalty.
            # For simplicity, reject infeasible moves unless T is very high.
            # Actually, better to just reject.
            pass

        T *= cooling

    return best_routes, best_cost


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    dist = _build_dist_matrix(instance)

    # 1. Initial construction
    routes = _construct_initial_solution(instance, dist)

    # 2. Local search
    routes, cost = _local_search(routes, dist, instance, max_iters=2000, time_limit=10.0)

    # 3. Simulated Annealing with multiple restarts
    best_routes = routes
    best_cost = cost

    while (time.time() - start_time) < deadline:
        # Perturb best solution
        perturbed = [r[:] for r in best_routes]
        # Randomly swap two customers or relocate
        all_customers = [c for r in perturbed for c in r]
        if len(all_customers) >= 2:
            c1, c2 = random.sample(all_customers, 2)
            # Find their positions
            for i, r in enumerate(perturbed):
                if c1 in r:
                    pos1 = r.index(c1)
                    r[pos1] = c2
                if c2 in r:
                    pos2 = r.index(c2)
                    r[pos2] = c1

        # Run SA from perturbed solution
        sa_routes, sa_cost = _simulated_annealing(perturbed, dist, instance, time_limit=5.0)

        if sa_cost < best_cost:
            best_cost = sa_cost
            best_routes = sa_routes

        # Check time
        if (time.time() - start_time) > deadline - 5.0:
            break

    # Final local search on best
    best_routes, best_cost = _local_search(best_routes, dist, instance, max_iters=1000, time_limit=5.0)

    return best_routes

# EVOLVE-BLOCK-END
