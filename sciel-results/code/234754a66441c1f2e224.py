"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix vectorized
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    def simulate_route(route):
        """
        Simulate a route (list of customer node ids).
        Returns (feasible, total_distance, final_time, final_battery).
        Depot is implicit at both ends.
        """
        if not route:
            return True, 0.0, 0.0, battery

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d
            if bat < -1e-9:
                return False, 0.0, t, bat
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, t, bat
            t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d
        if bat < -1e-9:
            return False, total_dist, t, bat
        if t > horizon + 1e-9:
            return False, total_dist, t, bat

        return True, total_dist, t, bat

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

    def is_feasible(routes):
        all_visited = []
        for r in routes:
            feasible, _, _, _ = simulate_route(r)
            if not feasible:
                return False
            all_visited.extend(r)
        if len(all_visited) != n_customers:
            return False
        if set(all_visited) != set(customers):
            return False
        return True

    def evaluate(routes):
        """Return (feasible, total_dist)."""
        all_visited = []
        total_d = 0.0
        for r in routes:
            feasible, d, _, _ = simulate_route(r)
            if not feasible:
                return False, float('inf')
            all_visited.extend(r)
            total_d += d
        if len(all_visited) != n_customers or set(all_visited) != set(customers):
            return False, float('inf')
        return True, total_d

    def construct_randomized(n_restarts=1, seed_offset=0):
        """
        Construct a solution using randomized insertion.
        Returns a list of routes.
        """
        rng = random.Random(seed_offset)
        remaining = set(customers)
        routes = []

        # Sort by due time for priority, add some randomness
        priority = sorted(remaining, key=lambda v: due[v] + rng.random() * 20)

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0

            while remaining:
                best_v = None
                best_score = float('inf')

                for v in remaining:
                    d = dist[prev, v]
                    bat_after = bat - d
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue

                    t_after = t + d / speed
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue

                    # Randomized score: distance + small random perturbation
                    score = d + rng.random() * 5.0
                    if score < best_score:
                        best_score = score
                        best_v = v

                if best_v is None:
                    break

                v = best_v
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                route.append(v)
                remaining.discard(v)
                prev = v

            if route:
                # Check if we can return to depot
                d_ret = dist[prev, 0]
                if bat - d_ret >= -1e-9 and t + d_ret / speed <= horizon + 1e-9:
                    routes.append(route)
                else:
                    # Try to salvage: remove last customer that makes it infeasible
                    while route and not (bat - d_ret >= -1e-9 and t + d_ret / speed <= horizon + 1e-9):
                        last = route.pop()
                        remaining.add(last)
                        # Recompute from scratch for this route is complex; just break
                        break
                    if route:
                        routes.append(route)
                    else:
                        remaining.add(route[0]) if route else None
                        break

        # Verify all customers are assigned; if not, add them as single-customer routes
        assigned = set()
        for r in routes:
            assigned.update(r)
        for v in customers:
            if v not in assigned:
                routes.append([v])

        return routes

    def tabu_search(initial_routes, max_iters=2000, tabu_tenure=7, seed_offset=0):
        """
        Tabu search on the set of routes.
        Moves: relocate a customer from one route to another.
        Returns best feasible solution found.
        """
        rng = random.Random(seed_offset + 1000)
        current_routes = [list(r) for r in initial_routes]
        feasible, current_dist = evaluate(current_routes)
        if not feasible:
            return current_routes, float('inf')

        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist

        # Tabu list: (customer, from_route_idx, to_route_idx, pos) -> iteration until which it's tabu
        tabu_list = {}
        current_iter = 0

        for it in range(max_iters):
            current_iter += 1
            if time.time() > deadline - 5:
                break

            # Generate neighborhood: relocate each customer to each position in each other route
            best_move = None
            best_move_dist = current_dist
            best_move_info = None

            # To speed up, sample a subset of moves
            n_moves_to_try = min(50, sum(len(r) for r in current_routes) * 3)
            # Build list of (customer_idx_in_routes, route_idx, pos)
            all_positions = []
            for ri, r in enumerate(current_routes):
                for pi, v in enumerate(r):
                    all_positions.append((ri, pi, v))

            if not all_positions:
                break

            # Randomly sample moves
            sampled = rng.sample(all_positions, min(n_moves_to_try, len(all_positions)))

            for (from_ri, from_pi, v) in sampled:
                if from_ri not in current_routes or from_pi >= len(current_routes[from_ri]):
                    continue
                if current_routes[from_ri][from_pi] != v:
                    continue

                # Skip if tabu
                tabu_key = (v, from_ri)
                if tabu_key in tabu_list and tabu_list[tabu_key] > current_iter:
                    continue

                # Try inserting v into each other route at each position
                for to_ri in range(len(current_routes)):
                    if to_ri == from_ri:
                        continue
                    for to_pi in range(len(current_routes[to_ri]) + 1):
                        # Construct new routes
                        new_routes = []
                        for ri in range(len(current_routes)):
                            if ri == from_ri:
                                new_r = list(current_routes[ri])
                                if from_pi < len(new_r) and new_r[from_pi] == v:
                                    new_r.pop(from_pi)
                                new_routes.append(new_r)
                            elif ri == to_ri:
                                new_r = list(current_routes[ri])
                                new_r.insert(to_pi, v)
                                new_routes.append(new_r)
                            else:
                                new_routes.append(list(current_routes[ri]))

                        # Evaluate
                        f, d = evaluate(new_routes)
                        if f and d < best_move_dist - 1e-9:
                            best_move_dist = d
                            best_move = new_routes
                            best_move_info = (v, from_ri, to_ri, to_pi)

            if best_move is not None:
                current_routes = best_move
                current_dist = best_move_dist

                # Update tabu list
                v, from_ri, to_ri, to_pi = best_move_info
                tabu_list[(v, from_ri)] = current_iter + tabu_tenure

                # Clean up old tabu entries
                if len(tabu_list) > 200:
                    tabu_list = {k: val for k, val in tabu_list.items() if val > current_iter}

                if current_dist < best_dist - 1e-9:
                    best_dist = current_dist
                    best_routes = [list(r) for r in current_routes]
            else:
                # No improving move; accept a random feasible move to escape
                if all_positions:
                    from_ri, from_pi, v = rng.choice(all_positions)
                    if from_ri >= len(current_routes) or from_pi >= len(current_routes[from_ri]):
                        continue
                    if current_routes[from_ri][from_pi] != v:
                        continue

                    to_ri = rng.choice(range(len(current_routes)))
                    if to_ri == from_ri:
                        continue
                    to_pi = rng.randint(0, len(current_routes[to_ri]))

                    new_routes = []
                    for ri in range(len(current_routes)):
                        if ri == from_ri:
                            new_r = list(current_routes[ri])
                            if from_pi < len(new_r) and new_r[from_pi] == v:
                                new_r.pop(from_pi)
                            new_routes.append(new_r)
                        elif ri == to_ri:
                            new_r = list(current_routes[ri])
                            new_r.insert(to_pi, v)
                            new_routes.append(new_r)
                        else:
                            new_routes.append(list(current_routes[ri]))

                    f, d = evaluate(new_routes)
                    if f:
                        current_routes = new_routes
                        current_dist = d
                        tabu_list[(v, from_ri)] = current_iter + tabu_tenure

        return best_routes, best_dist

    # Multi-start tabu search
    best_overall_routes = None
    best_overall_dist = float('inf')

    n_restarts = 0
    seed_counter = 0

    while time.time() < deadline - 10:
        seed_counter += 1
        n_restarts += 1

        # Construct initial solution
        init_routes = construct_randomized(seed_offset=seed_counter)

        # Evaluate
        feasible, d = evaluate(init_routes)
        if not feasible:
            # Try to fix by making single-customer routes for unassigned
            assigned = set()
            for r in init_routes:
                assigned.update(r)
            fixed_routes = []
            for r in init_routes:
                if r:
                    fixed_routes.append(r)
            for v in customers:
                if v not in assigned:
                    fixed_routes.append([v])
            init_routes = fixed_routes
            feasible, d = evaluate(init_routes)
            if not feasible:
                continue

        if d < best_overall_dist:
            best_overall_dist = d
            best_overall_routes = [list(r) for r in init_routes]

        # Run tabu search with time-limited iterations
        # Estimate how many iterations we can afford
        time_remaining = deadline - 10 - time.time()
        if time_remaining < 2:
            break

        # Run tabu search
        ts_routes, ts_dist = tabu_search(init_routes, max_iters=500, tabu_tenure=6, seed_offset=seed_counter)

        if ts_dist < best_overall_dist:
            best_overall_dist = ts_dist
            best_overall_routes = [list(r) for r in ts_routes]

    # Final safety: if we have no valid solution, use single-customer routes
    if best_overall_routes is None or not is_feasible(best_overall_routes):
        best_overall_routes = [[v] for v in customers]

    return best_overall_routes
# EVOLVE-BLOCK-END
