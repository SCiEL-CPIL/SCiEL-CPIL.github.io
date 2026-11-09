"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _solve_with_savings(instance, dist, time_limit=20):
    """Solve using Clarke-Wright savings heuristic."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    # Sort by savings descending
    savings.sort(reverse=True)
    
    # Track which route each customer belongs to
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    # Try to merge routes
    for s, ci, cj in savings:
        if time.time() > deadline - 2:
            break
        
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Try both orderings
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist, _ = _check_route(merged, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            for c in best_new_route:
                route_of[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


class _RouteState:
    """Track route state for fast feasibility evaluation."""
    def __init__(self, route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
        self.route = route
        self.dist = dist
        self.instance = instance
        self.battery_max = battery_max
        self.capacity = capacity
        self.speed = speed
        self.recharge_rate = recharge_rate
        self.horizon = horizon
        self.n_customers = instance["n_customers"]
        self.demands = instance["demands"]
        self.ready = instance["ready"]
        self.due = instance["due"]
        self.service = instance["service"]
        self._precompute()
    
    def _precompute(self):
        """Precompute prefix sums for fast evaluation."""
        self.route = list(self.route)
        n = len(self.route)
        self.total_demand = 0
        self.total_dist = 0.0
        self.total_time = 0.0
        self.battery_min = self.battery_max
        
        battery = self.battery_max
        t = 0.0
        prev = 0
        
        for v in self.route:
            d = self.dist[prev][v]
            battery -= d
            t += d / self.speed
            self.total_dist += d
            
            if battery < self.battery_min:
                self.battery_min = battery
            
            if v <= self.n_customers:
                self.total_demand += self.demands[v]
                if t < self.ready[v]:
                    t = self.ready[v]
                t += self.service[v]
            else:
                energy_needed = self.battery_max - battery
                t += energy_needed * self.recharge_rate
                battery = self.battery_max
            
            prev = v
        
        d = self.dist[prev][0]
        battery -= d
        t += d / self.speed
        self.total_dist += d
        self.total_time = t
        
        if battery < self.battery_min:
            self.battery_min = battery
        
        self._feasible = (self.total_demand <= self.capacity and 
                         self.battery_min >= -1e-9 and 
                         self.total_time <= self.horizon + 1e-9)
    
    def is_feasible(self):
        return self._feasible
    
    def get_distance(self):
        return self.total_dist


def _parallel_tempering_sa(instance, dist, time_limit=30):
    """Parallel tempering simulated annealing for VRPTW with battery constraints."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Initialize with savings heuristic
    routes = _solve_with_savings(instance, dist, 3)
    
    # Create initial states
    chains = []
    for r in routes:
        state = _RouteState(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        chains.append(state)
    
    # Parallel tempering parameters
    n_chains = 4
    temperature = 50.0
    cooling_rate = 0.995
    
    # Temperature schedule for parallel tempering
    chain_temps = [temperature * (1.5 ** i) for i in range(n_chains)]
    
    best_routes = [list(c.route) for c in chains]
    best_dist = sum(c.get_distance() for c in chains)
    
    # Customer to route index mapping
    cust_route = {}
    for ri, r in enumerate(routes):
        for c in r:
            if c <= n_customers:
                cust_route[c] = ri
    
    step_count = 0
    last_improve = start_time
    
    while time.time() < deadline - 1:
        step_count += 1
        
        # Pick a random chain
        chain_idx = random.randint(0, len(chains) - 1)
        state = chains[chain_idx]
        
        if not state.is_feasible():
            # If infeasible, try to fix by removing a customer
            if state.route:
                c = random.choice(state.route)
                if c <= n_customers:
                    new_route = [x for x in state.route if x != c]
                    new_state = _RouteState(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    chains[chain_idx] = new_state
                    state = new_state
                    if c in cust_route:
                        del cust_route[c]
                    continue
            else:
                continue
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # 2-opt within a route
            if len(state.route) < 2:
                continue
            
            i = random.randint(0, len(state.route) - 2)
            j = random.randint(i + 1, len(state.route) - 1)
            
            new_route = state.route[:i] + state.route[i:j+1][::-1] + state.route[j+1:]
            new_state = _RouteState(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            
            if new_state.is_feasible():
                old_dist = state.get_distance()
                new_dist = new_state.get_distance()
                delta = new_dist - old_dist
                
                # Acceptance based on temperature
                temp = chain_temps[chain_idx % n_chains]
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    chains[chain_idx] = new_state
                    state = new_state
                    if new_dist < old_dist:
                        best_dist = min(best_dist, sum(c.get_distance() for c in chains))
                        best_routes[chain_idx] = list(new_route)
                        last_improve = time.time()
                        
        elif move_type < 0.6:
            # Relocate customer between routes
            if len(chains) < 2:
                continue
            
            # Find a customer to move
            all_customers = [c for c in state.route if c <= n_customers]
            if not all_customers:
                continue
            
            c = random.choice(all_customers)
            
            # Remove from current route
            new_route_i = [x for x in state.route if x != c]
            new_state_i = _RouteState(new_route_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            
            # Try to insert into another random route
            target_idx = random.choice([x for x in range(len(chains)) if x != chain_idx])
            target_state = chains[target_idx]
            
            # Find best insertion position
            best_pos = -1
            best_total_dist = float('inf')
            
            for pos in range(len(target_state.route) + 1):
                new_route_j = target_state.route[:pos] + [c] + target_state.route[pos:]
                new_state_j = _RouteState(new_route_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                
                if new_state_i.is_feasible() and new_state_j.is_feasible():
                    total_d = new_state_i.get_distance() + new_state_j.get_distance()
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_pos = pos
                        best_state_i = new_state_i
                        best_state_j = new_state_j
            
            if best_pos != -1:
                old_total = state.get_distance() + target_state.get_distance()
                new_total = best_total_dist
                delta = new_total - old_total
                
                temp = chain_temps[chain_idx % n_chains]
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    chains[chain_idx] = best_state_i
                    chains[target_idx] = best_state_j
                    
                    # Update customer mapping
                    if c in cust_route:
                        del cust_route[c]
                    cust_route[c] = target_idx
                    
                    if new_total < old_total:
                        best_dist = min(best_dist, sum(c.get_distance() for c in chains))
                        best_routes[chain_idx] = list(best_state_i.route)
                        best_routes[target_idx] = list(best_state_j.route)
                        last_improve = time.time()
                        
        elif move_type < 0.8:
            # Swap customers between routes
            if len(chains) < 2:
                continue
            
            all_c1 = [c for c in state.route if c <= n_customers]
            if not all_c1:
                continue
            
            target_idx = random.choice([x for x in range(len(chains)) if x != chain_idx])
            target_state = chains[target_idx]
            all_c2 = [c for c in target_state.route if c <= n_customers]
            if not all_c2:
                continue
            
            c1 = random.choice(all_c1)
            c2 = random.choice(all_c2)
            
            # Swap
            new_route_i = [c2 if x == c1 else x for x in state.route]
            new_route_j = [c1 if x == c2 else x for x in target_state.route]
            
            new_state_i = _RouteState(new_route_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            new_state_j = _RouteState(new_route_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            
            if new_state_i.is_feasible() and new_state_j.is_feasible():
                old_total = state.get_distance() + target_state.get_distance()
                new_total = new_state_i.get_distance() + new_state_j.get_distance()
                delta = new_total - old_total
                
                temp = chain_temps[chain_idx % n_chains]
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    chains[chain_idx] = new_state_i
                    chains[target_idx] = new_state_j
                    
                    # Update customer mapping
                    if c1 in cust_route:
                        cust_route[c1] = target_idx
                    if c2 in cust_route:
                        cust_route[c2] = chain_idx
                    
                    if new_total < old_total:
                        best_dist = min(best_dist, sum(c.get_distance() for c in chains))
                        best_routes[chain_idx] = list(new_state_i.route)
                        best_routes[target_idx] = list(new_state_j.route)
                        last_improve = time.time()
                        
        else:
            # Or-Opt: move 1-3 consecutive nodes
            if len(state.route) < 2:
                continue
            
            L = random.choice([1, 2, 3])
            if len(state.route) < L + 1:
                continue
            
            i = random.randint(0, len(state.route) - L)
            moving = state.route[i:i+L]
            rest = state.route[:i] + state.route[i+L:]
            
            j = random.randint(0, len(rest))
            if j == i or j == i + L:
                continue
            
            new_route = rest[:j] + moving + rest[j:]
            new_state = _RouteState(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            
            if new_state.is_feasible():
                old_dist = state.get_distance()
                new_dist = new_state.get_distance()
                delta = new_dist - old_dist
                
                temp = chain_temps[chain_idx % n_chains]
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    chains[chain_idx] = new_state
                    state = new_state
                    if new_dist < old_dist:
                        best_dist = min(best_dist, sum(c.get_distance() for c in chains))
                        best_routes[chain_idx] = list(new_route)
                        last_improve = time.time()
        
        # Parallel tempering: swap chains with some probability
        if random.random() < 0.05 and len(chains) > 1:
            i, j = random.sample(range(len(chains)), 2)
            chains[i], chains[j] = chains[j], chains[i]
        
        # Cool down
        chain_temps = [t * cooling_rate for t in chain_temps]
        chain_temps = [max(t, 0.5) for t in chain_temps]
    
    # Return best solution
    final_routes = []
    for c in chains:
        if c.route:
            final_routes.append(list(c.route))
    
    # Ensure all customers are covered
    all_served = set()
    for r in final_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        final_routes.append([c])
    
    return final_routes


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 3
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _solve_instance(instance, time_limit=25):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Use parallel tempering SA
    sa_time = max(10, time_limit - 5)
    routes = _parallel_tempering_sa(instance, dist, sa_time)
    
    # Final polish with 2-opt on each route
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        for i in range(len(routes)):
            if time.time() > deadline - 1:
                break
            routes[i] = _two_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline - 1)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    n_customers = instance["n_customers"]
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
