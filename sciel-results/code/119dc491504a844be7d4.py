"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """Fast simulation of the ACO to evaluate a heuristic's quality."""
    n = len(demands)
    rng = np.random.RandomState(seed)
    pheromone = np.ones_like(distance_matrix)
    best_cost = float('inf')
    demands_arr = demands.astype(float)
    
    # Precompute feasible neighbors for each node to speed up search? 
    # No, capacity depends on path. We must check on the fly.
    
    for it in range(n_iters):
        iter_best = float('inf')
        for ant in range(n_ants):
            route_cost = 0.0
            current = 0
            remaining_cap = capacity
            visited = set()
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= remaining_cap:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    route_cost += distance_matrix[current, 0]
                    break
                
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = pheromone[current, j] * heuristic[current, j]
                
                total = probs.sum()
                if total < 1e-30:
                    # Pick random feasible
                    j = feasible[rng.randint(len(feasible))]
                else:
                    probs /= total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                    j = feasible[idx]
                
                route_cost += distance_matrix[current, j]
                visited.add(j)
                remaining_cap -= demands_arr[j]
                current = j
            
            # Add final return to depot
            route_cost += distance_matrix[current, 0]
            if route_cost < iter_best:
                iter_best = route_cost
        
        if iter_best < best_cost:
            best_cost = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        if iter_best > 0:
            pheromone += 1.0 / iter_best
    
    return best_cost

def _clarke_wright_savings(distance_matrix, demands, capacity, coordinates=None):
    """
    Generate a set of routes using Clarke-Wright Savings heuristic.
    Returns a list of routes (each route is a list of node indices, starting/ending with 0).
    """
    n = len(demands)
    demands_arr = demands.astype(float)
    
    # 1. Initial state: each customer is its own route [0, i, 0]
    # We represent routes as lists of customer indices (excluding depot for logic, but we need to track order)
    # Actually, CW works on edges. Let's stick to the standard algorithm.
    
    # Calculate savings for all pairs i, j
    # Savings(i, j) = d(0, i) + d(0, j) - d(i, j)
    d0 = distance_matrix[0, :]
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i+1, n):
            s = d0[i] + d0[j] - distance_matrix[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    pairs = []
    for i in range(1, n):
        for j in range(i+1, n):
            pairs.append((savings[i, j], i, j))
    pairs.sort(reverse=True)
    
    # Data structures for CW
    # route_of[i]: index of the route containing customer i
    # left[i], right[i]: neighbors in the route
    # demand_sum[r]: total demand of route r
    # cost[r]: cost of route r (excluding depot connections? No, usually total tour cost)
    
    # Let's use a simpler representation:
    # routes is a list of lists of customer nodes in order.
    # We start with n-1 routes: [i] for i in 1..n-1
    routes = [[i] for i in range(1, n)]
    route_index = {i: i-1 for i in range(1, n)} # Customer i is in route index i-1
    route_demand = [demands_arr[i] for i in range(1, n)]
    
    # To find neighbors efficiently, we maintain for each customer its left and right neighbor in the route
    # left[i], right[i]. For a single node route [i], left[i]=0, right[i]=0 (depot)
    left = [0] * n
    right = [0] * n
    
    # Initialize
    for i in range(1, n):
        left[i] = 0
        right[i] = 0
    
    # Process savings
    for s, i, j in pairs:
        if s <= 0:
            continue
        
        ri = route_index[i]
        rj = route_index[j]
        
        # Check if i and j are in the same route
        if ri == rj:
            continue
        
        # Check if i and j are endpoints of their respective routes
        # An endpoint is a node with one neighbor being 0 (depot)
        # left[i]==0 or right[i]==0
        # left[j]==0 or right[j]==0
        
        i_is_endpoint = (left[i] == 0 or right[i] == 0)
        j_is_endpoint = (left[j] == 0 or right[j] == 0)
        
        if not (i_is_endpoint and j_is_endpoint):
            continue
        
        # Check capacity constraint
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        # Merge routes
        # Determine which side of i and j connects to the depot
        # Case 1: left[i]==0 and left[j]==0 -> Connect right[i] to left[j]? No.
        # The route is a sequence. 
        # If left[i]==0, i is the "start" (after depot). right[i] is next.
        # If right[i]==0, i is the "end" (before depot). left[i] is prev.
        
        # We want to connect i and j.
        # The new link is (i, j) or (j, i).
        # We break the links (0, i) or (i, 0) and (0, j) or (j, 0).
        
        # Let's identify the "outer" neighbors (the ones connected to depot or other nodes)
        # and the "inner" neighbors.
        
        # Actually, it's easier to reconstruct the route list from the linked list structure.
        # But linked list merging is O(1) if we know the pointers.
        
        # Let's determine the direction.
        # We are merging route ri and rj.
        # We need to connect i and j.
        
        # Which side of i is free?
        if left[i] == 0:
            i_free_side = 'left'
            i_other_neighbor = right[i]
        else:
            i_free_side = 'right'
            i_other_neighbor = left[i]
            
        if left[j] == 0:
            j_free_side = 'left'
            j_other_neighbor = right[j]
        else:
            j_free_side = 'right'
            j_other_neighbor = left[j]
            
        # We connect i and j.
        # The new link is between i and j.
        # The "free" sides are the ones that were connected to depot.
        # So we replace (0, i) with (j, i) or (i, j) depending on orientation?
        # Wait, the savings formula d(0,i)+d(0,j)-d(i,j) implies we remove edges (0,i) and (0,j) and add (i,j).
        # So i and j become adjacent.
        
        # We need to update the linked list.
        # If i's free side is left, it means the route goes 0 -> i -> ...
        # If j's free side is left, it means the route goes 0 -> j -> ...
        
        # Let's just rebuild the affected routes from the linked list to be safe, or carefully update pointers.
        # Given n=50, rebuilding is cheap.
        
        # Let's extract the two routes as lists.
        def get_route_list(start_node):
            # start_node is a customer node
            # We need to traverse from the "start" (leftmost, connected to depot on left) to "end" (rightmost, connected to depot on right)
            # Actually, the route is a cycle 0->...->0.
            # In the list representation [c1, c2, ...], c1 is after 0, c_last is before 0.
            # So left[c1]=0, right[c_last]=0.
            
            # Find the start node (left[x]==0)
            # Traverse from start to end.
            node = start_node
            # Go left until we hit 0
            while left[node] != 0:
                node = left[node]
            # Now node is the start
            path = []
            curr = node
            while curr != 0:
                path.append(curr)
                curr = right[curr]
            return path
            
        route_i = get_route_list(i)
        route_j = get_route_list(j)
        
        # We need to merge them such that i and j are adjacent.
        # The edge (i,j) is added.
        # The edges (0, start_i) and (0, start_j) are removed? No, (0, end_i) and (0, end_j) are removed?
        # The savings removes the two depot connections that are "broken".
        # In the standard CW, we remove (0, i) and (0, j) from the *conceptual* star, but in the route, we remove the depot link from the *endpoints* that are being connected.
        
        # Wait, the standard CW assumes we connect i and j.
        # If i is an endpoint, its depot neighbor is either left or right.
        # If j is an endpoint, its depot neighbor is either left or right.
        
        # Let's determine the new route order.
        # If i's depot neighbor is left (i is start), and j's depot neighbor is right (j is end):
        # Route i: 0 -> i -> ... -> end_i -> 0
        # Route j: 0 -> start_j -> ... -> j -> 0
        # Connect i and j? No, we connect the endpoints that are "free" in the sense of the star.
        # Actually, the algorithm connects i and j directly.
        # So we join the two sequences.
        
        # Let's identify which end of route_i is i and which end of route_j is j.
        # i is either first or last in route_i.
        # j is either first or last in route_j.
        
        i_at_start = (left[i] == 0)
        i_at_end = (right[i] == 0)
        j_at_start = (left[j] == 0)
        j_at_end = (right[j] == 0)
        
        # The new route is formed by concatenating route_i and route_j, possibly reversing one.
        # We want i and j to be adjacent in the middle.
        
        if i_at_start and j_at_start:
            # 0->i... and 0->j...
            # Connect i and j? This would mean 0->i->j->...->end_j->0 and ...->end_i->0?
            # No, we remove depot links.
            # If we connect i and j, and both were starts, we are connecting the two "starts".
            # This implies we reverse one route?
            # Let's look at the geometry.
            # Route A: 0 - i - A2 - ... - Aend - 0
            # Route B: 0 - j - B2 - ... - Bend - 0
            # If we connect i and j, we get 0 - i - j - B2 ... Bend - 0 ... Aend - A2 - i? No.
            # The standard merge connects the two nodes.
            # If i is start of A and j is start of B:
            # We can form 0 - i - j - B2 ... Bend - 0 ... Aend - A2 - 0? No, that breaks A.
            # We must keep the internal order of the routes.
            # Route A: i -> A2 -> ... -> Aend
            # Route B: j -> B2 -> ... -> Bend
            # If we connect i and j, we can do:
            # 0 -> i -> j -> B2 ... Bend -> 0 ... Aend -> A2 -> 0? No.
            # We have to pick one direction.
            # Usually, we connect i and j.
            # If i is start of A and j is end of B:
            # 0 -> i -> A2 ... Aend -> j -> B... -> 0? No.
            
            # Let's use a robust method:
            # 1. Identify the "open" ends.
            # 2. The new route is the concatenation of the two paths between the connected nodes and the depot.
            
            # Path from depot to i in Route A:
            if i_at_start:
                path_ai = [i] + route_i[1:] # i is first
                # Actually route_i is [i, A2, ..., Aend]
            else:
                # i is last
                path_ai = list(reversed(route_i)) # [i, Aprev, ..., Astart]
                # Wait, if i is last, route_i is [Astart, ..., i]
                # Path from depot (start) to i is Astart -> ... -> i
                # Path from i to depot (end) is i -> ... -> Astart? No.
                
            # Let's just define the segment from Depot to Node and Node to Depot.
            # For Route A and node i:
            if i_at_start:
                seg_ai_to_depot = [] # i is connected to depot on left
                seg_depot_to_ai = route_i # i, A2, ..., Aend. The rest goes to depot.
                # Actually, the route is 0 -> i -> A2 ... -> Aend -> 0
                # The segment from i to the other end (Aend) is route_i[1:]
                # The segment from depot to i is just i (if we consider i as start)
                
            # This is getting complicated. Let's use a simpler heuristic construction for the "Golden Routes".
            # Instead of perfect CW, let's use a Greedy Nearest Neighbor with Savings bias, or just multiple NN starts.
            # Or, simpler: Use the Savings values to build a heuristic matrix directly.
            # eta[i,j] = 1 / (dist[i,j] + epsilon) * (1 + savings[i,j] / max_savings)
            
            # But the prompt asks for a *different strategy*.
            # "Route-membership indicator" is the strategy.
            # Let's implement a simpler constructive heuristic to get good routes:
            # 1. Sort customers by angle from depot.
            # 2. Sweep line / Partition into clusters based on angle and capacity.
            # 3. For each cluster, solve TSP (or use NN) to get a route.
            # This is "Sweep Heuristic". It's fast and produces good spatial clusters.
            
            pass # Skip complex CW implementation for now, use Sweep.
        
    # Let's replace the CW logic with a Sweep Heuristic to generate Golden Routes.
    
    # 1. Compute angles
    angles = np.arctan2(coordinates[1:, 1], coordinates[1:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi) # 0 to 2pi
    
    # 2. Sort customers by angle
    sorted_indices = np.argsort(angles)
    sorted_customers = sorted_indices + 1 # 1-based
    
    # 3. Partition into routes
    # Greedy packing in angular order
    routes = []
    current_route = []
    current_demand = 0
    
    for c in sorted_customers:
        if current_demand + demands_arr[c] > capacity:
            if current_route:
                routes.append(current_route)
            current_route = []
            current_demand = 0
        
        current_route.append(c)
        current_demand += demands_arr[c]
        
    if current_route:
        routes.append(current_route)
        
    # 4. Optimize each route slightly (2-Opt or just NN order within cluster)
    # For simplicity and speed, let's just use the angular order as the route order.
    # Or apply a simple 2-Opt improvement on each route.
    
    improved_routes = []
    for route in routes:
        if len(route) <= 1:
            improved_routes.append(route)
            continue
            
        # 2-Opt
        best_route = route
        best_cost = _route_cost([0] + best_route + [0], distance_matrix)
        
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route)):
                for j in range(i+1, len(best_route)):
                    # Reverse segment i..j
                    new_route = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    new_cost = _route_cost([0] + new_route + [0], distance_matrix)
                    if new_cost < best_cost - 1e-9:
                        best_cost = new_cost
                        best_route = new_route
                        improved = True
        improved_routes.append(best_route)
        
    return improved_routes

def _route_cost(route, distance_matrix):
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i], route[i+1]]
    return cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix using a Route-Membership Indicator strategy.
    1. Generate good routes using Sweep Heuristic + 2-Opt.
    2. Create a matrix where edges in these routes have high attractiveness.
    3. Blend with inverse distance to ensure connectivity.
    4. Validate and tune via ACO simulation.
    """
    n = len(demands)
    t_start = time.time()
    deadline = t_start + 250.0
    
    # 1. Generate Golden Routes
    golden_routes = _clarke_wright_savings(distance_matrix, demands, capacity, coordinates)
    
    # 2. Build Initial Heuristic Matrix
    # Base: Inverse Distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Route Membership Indicator
    # Count how many times an edge appears in the golden routes
    edge_count = np.zeros((n, n))
    for route in golden_routes:
        full_route = [0] + route + [0]
        for k in range(len(full_route) - 1):
            u = full_route[k]
            v = full_route[k+1]
            edge_count[u, v] += 1
            edge_count[v, u] += 1
            
    # Attractiveness = Base * (1 + weight * edge_count)
    # We want to strongly favor edges in good routes.
    # Let's try a few weights.
    
    best_heuristic = base_heuristic.copy()
    best_score = float('inf')
    
    # Candidate weights for route membership
    weights = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    # Also try pure inverse distance with different exponents
    alphas = [1.0, 1.5, 2.0]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        h_base = 1.0 / (distance_matrix ** alpha + 1e-9)
        for w in weights:
            if time.time() > deadline:
                break
            try:
                if w == 0:
                    h = h_base
                else:
                    h = h_base * (1.0 + w * edge_count)
                
                h = np.where(h > 0, h, 1e-9)
                
                # Evaluate with ACO simulation
                # Use fewer ants/iters for speed during search
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h, 
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = h.copy()
            except Exception:
                continue

    # If we have time, try to refine the specific edges or mix with savings
    if time.time() < deadline - 30:
        # Try blending with Savings
        d0 = distance_matrix[0, :]
        savings = np.zeros((n, n))
        for i in range(1, n):
            for j in range(i+1, n):
                s = d0[i] + d0[j] - distance_matrix[i, j]
                savings[i, j] = s
                savings[j, i] = s
        
        max_savings = savings.max()
        if max_savings > 0:
            norm_savings = savings / max_savings
            
            for blend in [0.0, 0.5, 1.0, 2.0]:
                if time.time() > deadline:
                    break
                try:
                    h = best_heuristic * (1.0 + blend * norm_savings)
                    h = np.where(h > 0, h, 1e-9)
                    score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h, 
                                           n_ants=10, n_iters=10, seed=42)
                    if score < best_score:
                        best_score = score
                        best_heuristic = h.copy()
                except Exception:
                    continue

    # Final Validation
    best_heuristic = np.asarray(best_heuristic, dtype=float)
    if best_heuristic.shape != distance_matrix.shape:
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
