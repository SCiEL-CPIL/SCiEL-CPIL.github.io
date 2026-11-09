"""EV routing with time windows – simulated annealing over permutation + feasibility splitter."""
import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

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
    # Distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))
    n_cust = len(customers)

    # Precompute for each customer: earliest departure from depot that still allows service
    # and check if single-customer route is feasible
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

    def split_permutation(perm):
        """Split a permutation of customers into feasible routes.
        Greedy: walk through perm, start a new route when current one becomes infeasible.
        Returns list of routes or None if any customer can't be placed.
        """
        routes = []
        current = []
        for v in perm:
            trial = current + [v]
            feasible, _, _, _ = simulate_route(trial)
            if feasible:
                current = trial
            else:
                if current:
                    routes.append(current)
                # Try single customer route
                single = [v]
                feasible_s, _, _, _ = simulate_route(single)
                if not feasible_s:
                    return None
                current = single
        if current:
            routes.append(current)
        return routes

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

    # Initial solution: sort by angle from depot
    depot = np.array([0.0, 0.0])
    angles = [math.atan2(coords[v][1] - depot[1], coords[v][0] - depot[0]) for v in customers]
    initial_perm = [v for _, v in sorted(zip(angles, customers))]

    routes = split_permutation(initial_perm)
    if routes is None:
        # Fallback: one customer per route
        routes = [[v] for v in customers]
    best_perm = initial_perm[:]
    best_routes = routes
    best_dist = total_distance(routes)

    # Simulated annealing
    # Neighborhood: swap two positions, 2-opt (reverse segment), Or-opt (move 1-2 consecutive)
    def get_neighbor(perm):
        n = len(perm)
        move_type = random.random()
        if move_type < 0.4:
            # Swap
            i, j = random.sample(range(n), 2)
            new_perm = perm[:]
            new_perm[i], new_perm[j] = perm[j], perm[i]
            return new_perm
        elif move_type < 0.75:
            # 2-opt
            i, j = sorted(random.sample(range(n), 2))
            new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
            return new_perm
        else:
            # Or-opt: move 1 or 2 consecutive elements
            seg_len = 1 if random.random() < 0.5 else 2
            i = random.randint(0, n - seg_len)
            j = random.randint(0, n - seg_len)
            if j == i:
                return perm[:]
            # Remove segment at i
            seg = perm[i:i+seg_len]
            rest = perm[:i] + perm[i+seg_len:]
            # Insert at j (adjust if j > i)
            if j > i:
                j -= seg_len
            new_perm = rest[:j] + seg + rest[j:]
            return new_perm

    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995  # cooling rate
    T = T_start
    iters = 0
    max_iters = 500000

    while time.time() < deadline:
        new_perm = get_neighbor(best_perm)
        new_routes = split_permutation(new_perm)
        if new_routes is None:
            # Reject infeasible
            new_dist = best_dist + 100.0
            delta = new_dist - best_dist
        else:
            new_dist = total_distance(new_routes)
            delta = new_dist - best_dist

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            best_perm = new_perm
            best_routes = new_routes
            best_dist = new_dist

        T *= alpha
        iters += 1
        if T < T_end:
            break

    # Final local search: simple 2-opt passes on each route
    for _ in range(3):
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
                    feasible, nd, _, _ = simulate_route(new_r)
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

    # Rebuild best_dist
    best_dist = total_distance(best_routes)

    # Verify all customers served exactly once
    all_served = [v for r in best_routes for v in r]
    if set(all_served) != set(customers) or len(all_served) != n_cust:
        # Fallback
        best_routes = [[v] for v in customers]

    return best_routes
