"""
Structural Change: Min-Cost Flow with Lagrangian Dual Prices and Simulated Annealing Polish.

Key structural ideas:
1. Solve Lagrangian dual to get facility prices that capture capacity scarcity
2. Build a min-cost flow network (source->customers->facilities->sink) where
   customer->facility edge costs include the dual prices
3. The flow solver guarantees integrality and capacity constraints simultaneously
4. Polish with simulated annealing to escape local optima found by the flow solution
5. Multi-start with different dual price initializations for diversity
"""

import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import heapq

DEADLINE = 115.0  # Safety margin before 150s kill


def _total_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Calculate the total cost of an assignment."""
    used = np.zeros(len(capacities), dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = float(open_cost[used].sum())
    cust = np.arange(len(assignment))
    assign_sum = float(assign_cost[cust, assignment].sum())
    return open_sum + assign_sum


def _valid(assignment, demands, capacities):
    """Check if assignment is valid (integer indices, capacity constraints)."""
    if len(assignment) != len(demands):
        return False
    if np.any(assignment < 0) or np.any(assignment >= len(capacities)):
        return False
    if np.any(np.asarray(assignment, dtype=float) % 1 != 0):
        return False
    loads = np.zeros(len(capacities))
    np.add.at(loads, assignment, demands)
    return bool(np.all(loads <= capacities + 1e-9))


def _solve_lagrangian_dual(demands, capacities, assign_cost, lam_init, max_iter=500, time_limit=5.0):
    """
    Solve the Lagrangian dual to get facility prices (lambda).
    Returns lambda array.
    """
    n_f = len(capacities)
    n_c = len(demands)
    lam = lam_init.copy()
    best_dual = -np.inf
    t_start = time.time()
    best_lam = lam.copy()
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
            
        # Subproblem: assign each customer to argmin_j (c_ij + lam_j)
        adjusted_costs = assign_cost + lam[None, :]
        best_facilities = np.argmin(adjusted_costs, axis=1)
        best_costs = adjusted_costs[np.arange(n_c), best_facilities]
        
        # Calculate load
        loads = np.zeros(n_f)
        np.add.at(loads, best_facilities, demands)
        
        # Dual value: sum of min costs + sum(lambda * capacity)
        dual_val = float(np.sum(best_costs)) + float(np.dot(lam, capacities))
        
        if dual_val > best_dual:
            best_dual = dual_val
            best_lam = lam.copy()
            
        # Subgradient: s_j = capacity_j - load_j
        subgrad = capacities - loads
        
        # Update lambda with adaptive step size
        norm_s = np.linalg.norm(subgrad)
        if norm_s < 1e-10:
            break
            
        step = 2.0 / (1.0 + 0.05 * it)
        lam = np.maximum(lam + step * subgrad / max(norm_s, 1e-6), 0.0)
        
        if it % 100 == 99:
            lam += np.random.uniform(-0.5, 0.5, size=n_f)
            lam = np.maximum(lam, 0.0)
        
    return best_lam


def _min_cost_flow_assign(demands, capacities, assign_cost, lam):
    """
    Solve the customer-facility assignment as a min-cost flow problem.
    
    Network:
    - Source -> Customer_i: capacity = demands[i], cost = 0
    - Customer_i -> Facility_j: capacity = demands[i], cost = assign_cost[i,j] + lam[j]
    - Facility_j -> Sink: capacity = capacities[j], cost = 0
    
    This ensures each customer is assigned to exactly one facility (flow conservation)
    and facility capacities are respected.
    
    We use a successive shortest path algorithm with potentials.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    # Build adjacency for the flow network
    # Nodes: 0=source, 1..n_c=customers, n_c+1..n_c+n_f=facilities, n_c+n_f+1=sink
    src = 0
    cust_start = 1
    fac_start = cust_start + n_c
    sink = fac_start + n_f
    total_nodes = sink + 1
    
    # Adjacency list: for each node, list of (neighbor, capacity, cost, reverse_index)
    adj = [[] for _ in range(total_nodes)]
    edge_costs = []  # Store (u, v, capacity, cost) for tracking
    
    def add_edge(u, v, cap, cost):
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])
    
    # Source -> Customers
    for i in range(n_c):
        add_edge(src, cust_start + i, int(demands[i]), 0)
    
    # Customers -> Facilities
    for i in range(n_c):
        for j in range(n_f):
            cost = assign_cost[i, j] + lam[j]
            add_edge(cust_start + i, fac_start + j, int(demands[i]), cost)
    
    # Facilities -> Sink
    for j in range(n_f):
        add_edge(fac_start + j, sink, int(capacities[j]), 0)
    
    # Successive shortest path with Dijkstra + potentials
    INF = float('inf')
    flow = 0
    total_demand = int(sum(demands))
    
    # Initialize potentials with Bellman-Ford (or just zeros if no negative edges)
    # Since all costs are non-negative (assign_cost >= 0, lam >= 0), we can start with 0
    potential = np.zeros(total_nodes)
    
    # Track assignment
    assignment = np.full(n_c, -1, dtype=int)
    assigned_flow = np.zeros(n_c)
    
    # For each customer, we need to send demands[i] units of flow
    # We'll do this by finding shortest paths from source to sink
    
    while flow < total_demand:
        # Find shortest path from src to sink
        dist = np.full(total_nodes, INF)
        prev_node = np.full(total_nodes, -1, dtype=int)
        prev_edge = np.full(total_nodes, -1, dtype=int)
        dist[src] = 0
        
        # Dijkstra with reduced costs
        visited = np.zeros(total_nodes, dtype=bool)
        pq = [(0, src)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            
            for e_idx, (v, cap, cost, rev_idx) in enumerate(adj[u]):
                if cap <= 0:
                    continue
                reduced_cost = cost + potential[u] - potential[v]
                new_dist = d + reduced_cost
                if new_dist < dist[v] - 1e-9:
                    dist[v] = new_dist
                    prev_node[v] = u
                    prev_edge[v] = e_idx
                    heapq.heappush(pq, (new_dist, v))
        
        if dist[sink] >= INF:
            break  # No more augmenting paths
        
        # Update potentials
        for i in range(total_nodes):
            if dist[i] < INF:
                potential[i] += dist[i]
        
        # Find bottleneck capacity along the path
        path_cap = total_demand - flow
        v = sink
        while v != src:
            u = prev_node[v]
            e_idx = prev_edge[v]
            path_cap = min(path_cap, adj[u][e_idx][1])
            v = u
        
        if path_cap <= 0:
            break
        
        # Augment flow along the path
        v = sink
        while v != src:
            u = prev_node[v]
            e_idx = prev_edge[v]
            adj[u][e_idx][1] -= path_cap
            # Reverse edge
            rev_idx = adj[u][e_idx][3]
            adj[v][rev_idx][1] += path_cap
            
            # Track which customer-facility edge this flow went through
            if cust_start <= u < fac_start and fac_start <= v < sink:
                cust_idx = u - cust_start
                fac_idx = v - fac_start
                assignment[cust_idx] = fac_idx
                assigned_flow[cust_idx] += path_cap
            
            v = u
        
        flow += path_cap
    
    # Check if all customers got assigned
    for i in range(n_c):
        if assignment[i] == -1:
            # Fallback: assign to facility with most remaining capacity
            loads = np.zeros(n_f)
            for k in range(n_c):
                if assignment[k] >= 0:
                    loads[assignment[k]] += demands[k]
            remaining = capacities - loads
            assignment[i] = int(np.argmax(remaining))
    
    return assignment


def _greedy_construct(demands, capacities, open_cost, assign_cost, lam=None):
    """
    Greedy construction: assign customers in order of increasing demand (hardest first).
    For each customer, pick the facility that minimizes (assign_cost + marginal open cost).
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    if lam is None:
        lam = np.zeros(n_f)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    in_use = np.zeros(n_f, dtype=bool)
    
    # Order customers by decreasing demand (hardest to place first)
    order = np.argsort(-demands)
    
    for i in order:
        d = demands[i]
        best_f = -1
        best_cost = np.inf
        
        for j in range(n_f):
            if loads[j] + d > capacities[j] + 1e-9:
                continue
            
            # Cost: assignment cost + marginal open cost
            cost = assign_cost[i, j] + lam[j]
            if not in_use[j]:
                cost += open_cost[j]
            
            if cost < best_cost:
                best_cost = cost
                best_f = j
        
        if best_f == -1:
            # Fallback: pick facility with most remaining space
            remaining = capacities - loads
            best_f = int(np.argmax(remaining))
        
        assignment[i] = best_f
        loads[best_f] += d
        in_use[best_f] = True
    
    return assignment


def _simulated_annealing(assignment, demands, capacities, open_cost, assign_cost, 
                          t_start, t_end, initial_temp=100.0, cooling_rate=0.995):
    """
    Simulated annealing to escape local optima.
    Tries random 1-moves and 2-swaps, accepting worse solutions with probability exp(-delta/T).
    """
    n_c = len(assignment)
    n_f = len(capacities)
    
    assignment = assignment.copy()
    loads = np.zeros(n_f)
    np.add.at(loads, assignment, demands)
    
    in_use = np.zeros(n_f, dtype=bool)
    for f in assignment:
        in_use[f] = True
    
    current_cost = _total_cost(assignment, demands, capacities, open_cost, assign_cost)
    best_assign = assignment.copy()
    best_cost = current_cost
    
    temp = initial_temp
    rng = np.random.default_rng(42)
    
    while time.time() < t_end:
        # Choose move type: 1-move (70%) or 2-swap (30%)
        if rng.random() < 0.7:
            # 1-move: pick random customer, try moving to random other facility
            i = rng.integers(0, n_c)
            cur_f = assignment[i]
            d = demands[i]
            
            # Pick a random alternative facility
            alternatives = [j for j in range(n_f) if j != cur_f and loads[j] + d <= capacities[j] + 1e-9]
            if not alternatives:
                continue
            new_f = alternatives[rng.integers(0, len(alternatives))]
            
            # Calculate cost delta
            delta_assign = assign_cost[i, new_f] - assign_cost[i, cur_f]
            
            delta_open = 0.0
            if not in_use[new_f]:
                delta_open += open_cost[new_f]
            if loads[cur_f] == d:
                delta_open -= open_cost[cur_f]
            
            delta = delta_assign + delta_open
            
            # Acceptance criterion
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-6)):
                # Apply move
                loads[cur_f] -= d
                loads[new_f] += d
                assignment[i] = new_f
                in_use[new_f] = True
                if loads[cur_f] == 0:
                    in_use[cur_f] = False
                
                current_cost += delta
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_assign = assignment.copy()
        else:
            # 2-swap: pick two random customers at different facilities
            i = rng.integers(0, n_c)
            k = rng.integers(0, n_c)
            if i == k:
                continue
            fi = assignment[i]
            fk = assignment[k]
            if fi == fk:
                continue
            di = demands[i]
            dk = demands[k]
            
            # Check capacity after swap
            if loads[fi] - di + dk > capacities[fi] + 1e-9:
                continue
            if loads[fk] - dk + di > capacities[fk] + 1e-9:
                continue
            
            # Calculate cost delta
            delta_assign = (assign_cost[i, fk] + assign_cost[k, fi]) - (assign_cost[i, fi] + assign_cost[k, fk])
            delta = delta_assign
            
            # Acceptance criterion
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-6)):
                # Apply swap
                loads[fi] = loads[fi] - di + dk
                loads[fk] = loads[fk] - dk + di
                assignment[i] = fk
                assignment[k] = fi
                current_cost += delta
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_assign = assignment.copy()
        
        # Cool down
        temp *= cooling_rate
    
    return best_assign


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Efficient local search with 1-moves and 2-swaps.
    """
    n_c = len(assignment)
    n_f = len(capacities)
    
    if n_c == 0:
        return assignment
        
    assignment = assignment.copy()
    loads = np.zeros(n_f)
    np.add.at(loads, assignment, demands)
    
    in_use = np.zeros(n_f, dtype=bool)
    for f in assignment:
        in_use[f] = True
        
    t_start = time.time()
    
    max_passes = 30
    for pass_num in range(max_passes):
        if time.time() >= deadline:
            break
            
        improved = False
        
        # --- 1-Move Search ---
        for i in range(n_c):
            if time.time() >= deadline:
                break
                
            cur_f = assignment[i]
            d = demands[i]
            cur_cost = assign_cost[i, cur_f]
            
            best_f = -1
            best_delta = 0.0
            
            for j in range(n_f):
                if j == cur_f:
                    continue
                if loads[j] + d > capacities[j] + 1e-9:
                    continue
                
                delta_assign = assign_cost[i, j] - cur_cost
                delta_open = 0.0
                if not in_use[j]:
                    delta_open += open_cost[j]
                if loads[cur_f] == d:
                    delta_open -= open_cost[cur_f]
                
                delta = delta_assign + delta_open
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = j
            
            if best_f >= 0 and best_delta < -1e-9:
                loads[cur_f] -= d
                loads[best_f] += d
                assignment[i] = best_f
                in_use[best_f] = True
                if loads[cur_f] == 0:
                    in_use[cur_f] = False
                improved = True
        
        # --- 2-Swap Search ---
        if time.time() < deadline:
            for i in range(n_c):
                if time.time() >= deadline:
                    break
                fi = assignment[i]
                di = demands[i]
                
                for k in range(i + 1, n_c):
                    fk = assignment[k]
                    if fi == fk:
                        continue
                    dk = demands[k]
                    
                    if loads[fi] - di + dk > capacities[fi] + 1e-9:
                        continue
                    if loads[fk] - dk + di > capacities[fk] + 1e-9:
                        continue
                    
                    delta = (assign_cost[i, fk] + assign_cost[k, fi]) - (assign_cost[i, fi] + assign_cost[k, fk])
                    
                    if delta < -1e-9:
                        loads[fi] = loads[fi] - di + dk
                        loads[fk] = loads[fk] - dk + di
                        assignment[i] = fk
                        assignment[k] = fi
                        improved = True
                        break
                
                if improved and time.time() >= deadline:
                    break
        
        if not improved:
            break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_c = len(demands)
    n_f = len(capacities)
    t0 = time.time()
    
    best_assign = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions
    initial_solutions = []
    
    # Solution 1: Greedy with zero lambda
    sol1 = _greedy_construct(demands, capacities, open_cost, assign_cost, lam=np.zeros(n_f))
    initial_solutions.append(sol1)
    
    # Solution 2: Min-cost flow with zero lambda
    if time.time() - t0 < DEADLINE - 30:
        sol2 = _min_cost_flow_assign(demands, capacities, assign_cost, lam=np.zeros(n_f))
        initial_solutions.append(sol2)
    
    # Solution 3: Lagrangian dual + min-cost flow
    if time.time() - t0 < DEADLINE - 25:
        lam3 = _solve_lagrangian_dual(demands, capacities, assign_cost, 
                                       lam_init=np.zeros(n_f), 
                                       max_iter=200, time_limit=5.0)
        sol3 = _min_cost_flow_assign(demands, capacities, assign_cost, lam=lam3)
        initial_solutions.append(sol3)
    
    # Solution 4: Lagrangian dual with different init + greedy
    if time.time() - t0 < DEADLINE - 20:
        lam4 = _solve_lagrangian_dual(demands, capacities, assign_cost, 
                                       lam_init=open_cost * 0.3, 
                                       max_iter=150, time_limit=4.0)
        sol4 = _greedy_construct(demands, capacities, open_cost, assign_cost, lam=lam4)
        initial_solutions.append(sol4)
    
    # Solution 5: Random perturbation greedy
    if time.time() - t0 < DEADLINE - 15:
        lam5 = np.random.uniform(0, 15, size=n_f)
        sol5 = _greedy_construct(demands, capacities, open_cost, assign_cost, lam=lam5)
        initial_solutions.append(sol5)
    
    # Phase 2: Polish each initial solution with local search, keep best
    for idx, sol in enumerate(initial_solutions):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        if not _valid(sol, demands, capacities):
            continue
        
        sol_cost = _total_cost(sol, demands, capacities, open_cost, assign_cost)
        if sol_cost < best_cost:
            best_cost = sol_cost
            best_assign = sol.copy()
        
        # Local search polish
        remaining = DEADLINE - 5 - idx * 2
        if time.time() < remaining:
            polished = _local_search(sol, demands, capacities, open_cost, assign_cost, 
                                      t0 + remaining)
            if _valid(polished, demands, capacities):
                pol_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
                if pol_cost < best_cost:
                    best_cost = pol_cost
                    best_assign = polished.copy()
    
    # Phase 3: Simulated annealing on the best solution
    if best_assign is not None and time.time() - t0 < DEADLINE - 5:
        sa_start = time.time()
        sa_end = t0 + DEADLINE - 2
        sa_result = _simulated_annealing(best_assign, demands, capacities, open_cost, 
                                          assign_cost, sa_start, sa_end,
                                          initial_temp=50.0, cooling_rate=0.998)
        if _valid(sa_result, demands, capacities):
            sa_cost = _total_cost(sa_result, demands, capacities, open_cost, assign_cost)
            if sa_cost < best_cost:
                best_cost = sa_cost
                best_assign = sa_result.copy()
    
    # Final fallback
    if best_assign is None or not _valid(best_assign, demands, capacities):
        best_assign = _greedy_construct(demands, capacities, open_cost, assign_cost, lam=np.zeros(n_f))
    
    return best_assign.astype(int)
