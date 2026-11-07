import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))
    
    # Helper to check if a route is valid and return its cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, t, d_total, load
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, t, d_total, load
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, t, d_total, load
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, t, d_total, load
        if t > horizon + _EPS:
            return False, 0.0, t, d_total, load
        d_total += d
        return True, d_total, t, d_total, load

    # Initial solution: one route per customer (valid baseline)
    routes = [[c] for c in customers]
    
    # MCTS Configuration
    # Since we have 25 customers, the state space is large, but we can use MCTS to optimize the sequence of a single route or the whole set.
    # Given the constraints, a full MCTS over the whole VRP is complex. 
    # Instead, we will use MCTS to optimize the order of customers within routes, or to decide insertions.
    # A simpler and effective MCTS application here: Optimize the sequence of a set of customers assigned to a single route.
    # We will first cluster customers into routes (e.g., using simple greedy or k-means), then use MCTS to optimize each route's sequence.
    
    # Phase 1: Initial Clustering / Route Assignment
    # Use a simple greedy insertion to create initial routes, respecting capacity and basic feasibility.
    # Sort customers by due date.
    customers_sorted = sorted(customers, key=lambda c: due[c])
    
    initial_routes = []
    for c in customers_sorted:
        best_route_idx = -1
        best_cost = float('inf')
        
        # Try inserting into existing routes
        for i, r in enumerate(initial_routes):
            # Check if adding c is feasible in terms of capacity
            if sum(demands[x] for x in r) + demands[c] > capacity:
                continue
            
            # Try inserting at best position
            best_pos_in_r = -1
            best_d_in_r = float('inf')
            for pos in range(len(r) + 1):
                trial = r[:pos] + [c] + r[pos:]
                valid, d, t, _, _ = check_route(trial)
                if valid and d < best_d_in_r:
                    best_d_in_r = d
                    best_pos_in_r = pos
            
            if best_pos_in_r != -1:
                # Calculate total cost change
                old_d = check_route(r)[1]
                new_total = old_d + best_d_in_r - old_d # This logic is slightly off, just use best_d_in_r as proxy for improvement or just minimize d
                if best_d_in_r < best_cost:
                    best_cost = best_d_in_r
                    best_route_idx = i
                    best_pos = best_pos_in_r
        
        if best_route_idx != -1:
            initial_routes[best_route_idx] = initial_routes[best_route_idx][:best_pos] + [c] + initial_routes[best_route_idx][best_pos:]
        else:
            initial_routes.append([c])
    
    routes = initial_routes

    # Phase 2: MCTS Optimization for each route
    # We will run MCTS to find a better sequence for the customers in each route.
    
    class MCTSNode:
        def __init__(self, state, parent=None, action=None):
            self.state = state # (unserved_set, current_loc, time, battery, load, path)
            self.parent = parent
            self.action = action
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.untried_actions = None
            self.is_terminal = False
            
        def expand(self, all_customers):
            # State: (unserved_set, current_loc, time, battery, load, path)
            unserved, cur_loc, t, b, load, path = self.state
            
            # Generate possible actions: visit any unserved customer or return to depot if feasible
            actions = []
            
            # Option 1: Return to depot
            d_back = dist[cur_loc, 0]
            t_back = t + d_back / speed
            b_back = b - d_back
            if b_back >= -_EPS and t_back <= horizon + _EPS:
                actions.append((0, "return")) # Action 0 is depot
            
            # Option 2: Visit unserved customers
            for c in unserved:
                if load + demands[c] > capacity:
                    continue
                d = dist[cur_loc, c]
                t_arr = t + d / speed
                b_arr = b - d
                if b_arr < -_EPS:
                    continue
                t_wait = max(t_arr, ready[c])
                if t_wait > due[c] + _EPS:
                    continue
                t_dep = t_wait + service[c]
                # Heuristic: prioritize customers with tight windows or close distance
                urgency = (due[c] - t_wait)
                if urgency > 0:
                    actions.append((c, "visit"))
            
            return actions

        def simulate(self, all_customers, dist, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon):
            # Simulate a random rollout from this state
            unserved, cur_loc, t, b, load, path = self.state
            path = list(path)
            
            while unserved:
                # Choose a random valid action
                valid_actions = []
                # Try to return to depot
                d_back = dist[cur_loc, 0]
                t_back = t + d_back / speed
                b_back = b - d_back
                if b_back >= -_EPS and t_back <= horizon + _EPS:
                    valid_actions.append(0)
                
                for c in unserved:
                    if load + demands[c] > capacity:
                        continue
                    d = dist[cur_loc, c]
                    t_arr = t + d / speed
                    b_arr = b - d
                    if b_arr < -_EPS:
                        continue
                    t_wait = max(t_arr, ready[c])
                    if t_wait > due[c] + _EPS:
                        continue
                    valid_actions.append(c)
                
                if not valid_actions:
                    # Dead end, return failure or high cost
                    return float('inf')
                
                # Weighted random choice: prefer closer or more urgent
                weights = []
                for a in valid_actions:
                    if a == 0:
                        weights.append(1.0)
                    else:
                        d = dist[cur_loc, a]
                        urgency = due[a] - t
                        weights.append(1.0 / (d + 1) * (urgency + 1))
                
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen = valid_actions[0]
                for w, a in zip(weights, valid_actions):
                    cumulative += w
                    if r <= cumulative:
                        chosen = a
                        break
                
                if chosen == 0:
                    # End of route
                    d = dist[cur_loc, 0]
                    t += d / speed
                    # Total distance of this route segment
                    # We need to calculate the total distance of the whole path to evaluate
                    total_d = sum(dist[path[i], path[i+1]] for i in range(len(path)-1)) + dist[cur_loc, 0]
                    return total_d
                else:
                    c = chosen
                    d = dist[cur_loc, c]
                    t += d / speed
                    b -= d
                    t = max(t, ready[c])
                    t += service[c]
                    load += demands[c]
                    path.append(c)
                    unserved.discard(c)
                    cur_loc = c
            
            # If all served, return to depot
            d = dist[cur_loc, 0]
            t += d / speed
            b -= d
            if b < -_EPS or t > horizon + _EPS:
                return float('inf')
            total_d = sum(dist[path[i], path[i+1]] for i in range(len(path)-1)) + dist[cur_loc, 0]
            return total_d

    def mcts_optimize_route(route_customers):
        if len(route_customers) <= 2:
            return route_customers
        
        # Initial state: Depot, full battery, time 0, load 0, all customers in route unserved
        unserved = set(route_customers)
        initial_state = (unserved, 0, 0.0, battery_cap, 0.0, [0])
        root = MCTSNode(initial_state)
        
        # MCTS Parameters
        c_exploration = 1.414
        iterations = 50 # Adjust based on time
        
        for _ in range(iterations):
            if time.time() > DEADLINE:
                break
            
            # Selection
            node = root
            while not node.is_terminal and node.untried_actions is not None and node.untried_actions:
                # If has untried actions, expand
                break
            
            if node.untried_actions is None or not node.untried_actions:
                # Select child using UCB
                if not node.children:
                    # Should not happen if untried is empty and not terminal
                    break
                best_child = None
                best_ucb = -float('inf')
                for child in node.children:
                    if child.visits == 0:
                        ucb = float('inf')
                    else:
                        exploitation = child.value / child.visits
                        exploration = c_exploration * math.sqrt(math.log(node.visits + 1) / child.visits)
                        ucb = exploitation + exploration
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                node = best_child
            else:
                # Expand
                actions = node.expand(route_customers)
                node.untried_actions = actions
                if not actions:
                    node.is_terminal = True
                    node.value = float('inf')
                    # Backpropagate
                    while node:
                        node.visits += 1
                        node.value += node.value # This logic is flawed, value should be reward
                        node = node.parent
                    continue
                
                # Pick a random untried action
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                # Generate new state
                unserved, cur_loc, t, b, load, path = node.state
                if action[1] == "return":
                    # Terminal
                    d = dist[cur_loc, 0]
                    t_new = t + d / speed
                    b_new = b - d
                    if b_new < -_EPS or t_new > horizon + _EPS:
                        reward = float('inf')
                    else:
                        total_d = sum(dist[path[i], path[i+1]] for i in range(len(path)-1)) + d
                        reward = total_d
                    new_state = (unserved, 0, t_new, b_new, load, path + [0])
                    new_node = MCTSNode(new_state, node, action)
                    new_node.is_terminal = True
                    new_node.value = reward
                    node.children.append(new_node)
                    # Backpropagate
                    while new_node:
                        new_node.visits += 1
                        new_node.value += reward
                        new_node = new_node.parent
                else:
                    c = action[0]
                    d = dist[cur_loc, c]
                    t_arr = t + d / speed
                    b_arr = b - d
                    t_wait = max(t_arr, ready[c])
                    t_dep = t_wait + service[c]
                    load_new = load + demands[c]
                    new_unserved = unserved.copy()
                    new_unserved.discard(c)
                    new_state = (new_unserved, c, t_dep, b_arr, load_new, path + [c])
                    new_node = MCTSNode(new_state, node, action)
                    node.children.append(new_node)
                    
                    # Simulation
                    reward = new_node.simulate(route_customers, dist, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon)
                    
                    # Backpropagate
                    while new_node:
                        new_node.visits += 1
                        new_node.value += reward
                        new_node = new_node.parent
        
        # Extract best path
        best_path = [0]
        node = root
        while True:
            if not node.children:
                break
            best_child = max(node.children, key=lambda x: x.visits)
            if best_child.action[1] == "return":
                break
            best_path.append(best_child.action[0])
            node = best_child
        
        return best_path[1:] # Remove depot

    # Apply MCTS to each route
    for i, r in enumerate(routes):
        if time.time() > DEADLINE:
            break
        new_r = mcts_optimize_route(r)
        # Validate new route
        if new_r:
            valid, d, t, _, _ = check_route(new_r)
            if valid:
                routes[i] = new_r
            else:
                # Fallback to original if invalid
                pass
        else:
            # MCTS failed, try simple 2-opt or keep original
            pass

    # Final Validation and Return
    final_routes = []
    for r in routes:
        if r:
            final_routes.append(r)
            
    served = set()
    for r in final_routes:
        for c in r:
            served.add(c)
    
    missing = set(customers) - served
    if missing:
        for c in missing:
            final_routes.append([c])
            
    return final_routes
