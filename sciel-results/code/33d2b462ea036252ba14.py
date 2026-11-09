import numpy as np
import time
import random
import scipy.optimize as so

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP + cluster-based local search.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init(seed):
        """Greedy initialization with randomization."""
        random.seed(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        order = np.argsort(-demands)
        
        for i in order:
            fac_costs = assign_cost[i]
            noisy_costs = fac_costs + np.random.uniform(0, 1, n_fac) * np.std(fac_costs)
            fac_order_noisy = np.argsort(noisy_costs)
            
            assigned = False
            for f in fac_order_noisy:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                space = capacities - load
                max_space = np.max(space)
                candidates = np.where(space == max_space)[0]
                f = random.choice(candidates.tolist())
                res[i] = f
                load[f] += demands[i]
                
        return res

    def cluster_local_search(res, load, max_iter=2000):
        """
        Local search with cluster moves (move groups of customers together).
        """
        improved = True
        it = 0
        best_res = res.copy()
        best_score = compute_score(res)
        best_load = load.copy()
        
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            # Try cluster moves: pick a random cluster of 2-5 customers and try to move them
            cluster_size = random.randint(2, 5)
            cluster = np.random.choice(n_cust, size=cluster_size, replace=False)
            
            # Current facility for each customer in cluster
            current_facs = res[cluster]
            
            # Try all possible new facility assignments for the cluster
            # For efficiency, only try assignments that respect capacity
            best_cluster_score = compute_score(best_res)
            best_cluster_assign = None
            
            # Generate candidate assignments for the cluster
            # For each customer in cluster, try a few random facilities
            candidates_per_cust = 3
            for trial in range(50):
                if time.time() > deadline:
                    break
                    
                new_assign = best_res.copy()
                new_load = best_load.copy()
                
                # For each customer in cluster, pick a random new facility
                valid = True
                temp_load = new_load.copy()
                temp_res = new_assign.copy()
                
                for idx, i in enumerate(cluster):
                    # Pick a random facility
                    f_new = random.randint(0, n_fac - 1)
                    temp_res[i] = f_new
                    temp_load[f_new] += demands[i]
                    temp_load[current_facs[idx]] -= demands[i]
                    
                    # Check capacity
                    if temp_load[f_new] > capacities[f_new] + 1e-9:
                        valid = False
                        break
                    if temp_load[current_facs[idx]] < -1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check all loads are non-negative
                if np.any(temp_load < -1e-9):
                    continue
                
                # Compute score
                score = compute_score(temp_res)
                if score < best_cluster_score - 1e-9:
                    best_cluster_score = score
                    best_cluster_assign = temp_res.copy()
                    best_cluster_load = temp_load.copy()
                    improved = True
            
            if improved and best_cluster_assign is not None:
                best_res = best_cluster_assign
                best_load = best_cluster_load
                best_score = best_cluster_score
        
        return best_res, best_load, best_score

    def simulated_annealing(res, load, max_iter=5000):
        """
        Simulated annealing to escape local optima.
        """
        current_res = res.copy()
        current_load = load.copy()
        current_score = compute_score(current_res)
        best_res = current_res.copy()
        best_load = current_load.copy()
        best_score = current_score
        
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.999
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if T < T_min:
                break
                
            # Pick a random move: relocate or swap
            move_type = random.random()
            
            if move_type < 0.7:
                # Relocate
                i = random.randint(0, n_cust - 1)
                f_new = random.randint(0, n_fac - 1)
                f_old = current_res[i]
                
                if f_new == f_old:
                    continue
                
                # Check capacity
                if current_load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                
                # Compute delta
                delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                if current_load[f_new] == 0:
                    delta += open_cost[f_new]
                if current_load[f_old] == demands[i]:
                    delta -= open_cost[f_old]
                
                # Acceptance criterion
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply move
                    current_load[f_old] -= demands[i]
                    current_load[f_new] += demands[i]
                    current_res[i] = f_new
                    current_score += delta
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_res = current_res.copy()
                        best_load = current_load.copy()
                        
            else:
                # Swap
                i, j = random.sample(range(n_cust), 2)
                f_i = current_res[i]
                f_j = current_res[j]
                
                if f_i == f_j:
                    continue
                
                # Check capacity
                if current_load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                    continue
                if current_load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                    continue
                
                # Compute delta
                delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Acceptance criterion
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply move
                    current_load[f_i] += demands[j] - demands[i]
                    current_load[f_j] += demands[i] - demands[j]
                    current_res[i], current_res[j] = f_j, f_i
                    current_score += delta
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_res = current_res.copy()
                        best_load = current_load.copy()
            
            T *= cooling_rate
        
        return best_res, best_load, best_score

    def solve_milp(res_init):
        """
        Solve the MILP formulation of the capacitated facility location problem.
        """
        # Variables: y_ij (assign i to j), x_j (open facility j)
        # Total variables: n_cust * n_fac + n_fac
        
        # Objective coefficients
        c = np.zeros(n_cust * n_fac + n_fac)
        c[:n_cust * n_fac] = assign_cost.flatten()
        c[n_cust * n_fac:] = open_cost
        
        # Inequality constraints: sum_i d_i * y_ij <= cap_j * x_j
        A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
        b_ub = np.zeros(n_fac)
        
        for j in range(n_fac):
            for i in range(n_cust):
                idx = i * n_fac + j
                A_ub[j, idx] = demands[i]
            A_ub[j, n_cust * n_fac + j] = -capacities[j]
        
        # Equality constraints: sum_j y_ij = 1
        A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
        b_eq = np.ones(n_cust)
        
        for i in range(n_cust):
            for j in range(n_fac):
                idx = i * n_fac + j
                A_eq[i, idx] = 1.0
        
        # Integer constraints
        integrality = np.ones(n_cust * n_fac + n_fac)
        
        # Bounds
        bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
        
        # Initial guess
        x0 = np.zeros(n_cust * n_fac + n_fac)
        for i in range(n_cust):
            j = res_init[i]
            x0[i * n_fac + j] = 1.0
            x0[n_cust * n_fac + j] = 1.0
        
        try:
            res_opt = so.milp(
                c=c,
                constraints=[
                    so.LinearConstraint(A_ub, ub=b_ub),
                    so.LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
                ],
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 30, "mip_rel_gap": 0.05}
            )
            
            if res_opt.status == 0:
                y = res_opt.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                x = res_opt.x[n_cust * n_fac:]
                
                # Round y to get assignment
                res_milp = np.argmax(y, axis=1)
                
                # Check feasibility
                load = np.zeros(n_fac)
                np.add.at(load, res_milp, demands)
                if np.any(load > capacities + 1e-9):
                    return None, np.inf
                
                score = compute_score(res_milp)
                return res_milp, score
            else:
                return None, np.inf
        except Exception:
            return None, np.inf

    best_assign = None
    best_score = np.inf
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        res = greedy_init(r)
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # Phase 1: Cluster local search
        res_cs, load_cs, score_cs = cluster_local_search(res, load, max_iter=500)
        if score_cs < best_score:
            best_score = score_cs
            best_assign = res_cs.copy()
        
        # Phase 2: Simulated annealing
        if time.time() < deadline - 10.0:
            res_sa, load_sa, score_sa = simulated_annealing(res_cs, load_cs, max_iter=3000)
            if score_sa < best_score:
                best_score = score_sa
                best_assign = res_sa.copy()
        
        # Phase 3: Try MILP
        if time.time() < deadline - 35.0:
            res_milp, score_milp = solve_milp(res_sa)
            if res_milp is not None:
                if score_milp < best_score:
                    best_score = score_milp
                    best_assign = res_milp.copy()
                
                # Refine MILP solution with cluster local search
                load_milp = np.zeros(n_fac)
                np.add.at(load_milp, res_milp, demands)
                res_refined, load_refined, score_refined = cluster_local_search(res_milp, load_milp, max_iter=300)
                if score_refined < best_score:
                    best_score = score_refined
                    best_assign = res_refined.copy()

    return best_assign.astype(int)
