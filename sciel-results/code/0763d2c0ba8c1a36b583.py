"""Capacitated facility location via min-cost flow with optional facility opening (branch on open set),
then local search for fixed-charge refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Returns: numpy int array of facility indices for each customer.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)
    total_demand = np.sum(demands)

    def compute_score(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def greedy_init():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res, load

    def local_search(res, load, max_passes=5):
        """Iterative relocate local search."""
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
            if not improved:
                break

    def min_cost_flow_for_open_set(open_set):
        """
        Solve min-cost flow: source -> customers (cap=demand[i], cost=0)
        customers -> facilities in open_set (cap=demand[i], cost=assign_cost[i,j])
        facilities -> sink (cap=capacity[j], cost=0)
        Uses scipy's min_weight_full_bipartite_matching on expanded graph? 
        Actually, scipy.sparse.csgraph.min_weight_full_bipartite_matching is for assignment, not flow.
        Let's use a different approach: formulate as a transportation problem and use scipy.optimize.linprog?
        That might be slow. Let's use a simple successive shortest path or just use scipy's min_cost_flow if available.
        Actually, scipy doesn't have a direct min_cost_flow in csgraph. 
        Let's implement a simple min-cost flow using successive shortest path with potentials, or use the fact that
        this is a transportation problem and use scipy.optimize.linprog with the transportation structure.
        For 100 customers and up to 20 facilities, the LP has 100*20=2000 variables and 120 constraints.
        linprog with method='highs' should be fast.
        """
        from scipy.optimize import linprog

        if len(open_set) == 0:
            return None, np.inf

        # Variables: x[i,j] for i in range(n_cust), j in open_set
        # Objective: sum assign_cost[i,j] * x[i,j]
        # Constraints:
        #   For each i: sum_{j in open_set} x[i,j] = demands[i]
        #   For each j in open_set: sum_i x[i,j] <= capacities[j]
        #   x[i,j] >= 0

        n_open = len(open_set)
        c = assign_cost[:, open_set].flatten()  # shape (n_cust * n_open,)

        # Equality constraints: one per customer
        # A_eq[i, j] = 1 if customer i and facility j (in open_set)
        A_eq = np.zeros((n_cust, n_cust * n_open))
        for i in range(n_cust):
            for k, j in enumerate(open_set):
                A_eq[i, i * n_open + k] = 1.0
        b_eq = demands

        # Inequality constraints: one per facility in open_set
        A_ub = np.zeros((n_open, n_cust * n_open))
        for k, j in enumerate(open_set):
            for i in range(n_cust):
                A_ub[k, i * n_open + k] = 1.0
        b_ub = capacities[open_set]

        bounds = [(0, None)] * (n_cust * n_open)

        try:
            res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res_lp.success:
                return None, np.inf
            x = res_lp.x.reshape(n_cust, n_open)
            # Extract assignment
            res = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                # Each customer is assigned to exactly one facility (sum x[i,:] = demands[i])
                # Since it's a flow, x[i,j] should be either 0 or demands[i] for some j
                # Find the j with max flow
                j_idx = np.argmax(x[i])
                res[i] = open_set[j_idx]
            # Verify
            load = np.zeros(n_fac)
            np.add.at(load, res, demands)
            if np.any(load > capacities + 1e-9):
                return None, np.inf
            return res, np.sum(c * x)
        except Exception:
            return None, np.inf

    # First, get a feasible solution via greedy
    res, load = greedy_init()
    best_score = compute_score(res)
    best_assign = res.copy()
    best_load = load.copy()

    # Try to improve using min-cost flow with different open sets
    # Heuristic: try open sets based on which facilities are used in greedy, and neighbors
    # Also try all single-facility, all pairs, etc. if n_fac is small enough
    # For n_fac=20, trying all subsets is 2^20 = 1M, too many.
    # Try: greedy's open set, then try removing one facility, adding one facility, etc.

    greedy_open = set(best_assign)
    
    # Try the greedy open set
    res_mcf, cost_mcf = min_cost_flow_for_open_set(sorted(greedy_open))
    if res_mcf is not None:
        total = cost_mcf + np.sum(open_cost[greedy_open])
        if total < best_score:
            best_score = total
            best_assign = res_mcf.copy()
            load = np.zeros(n_fac)
            np.add.at(load, best_assign, demands)
            best_load = load.copy()

    # Try to improve by local search
    local_search(best_assign, best_load, max_passes=10)
    new_score = compute_score(best_assign)
    if new_score < best_score:
        best_score = new_score

    # Try a few other open sets: 
    # 1. All facilities
    # 2. Top k cheapest facilities
    # 3. Facilities used in best_assign

    # All facilities
    all_open = list(range(n_fac))
    res_mcf, cost_mcf = min_cost_flow_for_open_set(all_open)
    if res_mcf is not None:
        load_check = np.zeros(n_fac)
        np.add.at(load_check, res_mcf, demands)
        used = load_check > 0
        total = cost_mcf + np.sum(open_cost[used])
        if total < best_score:
            best_score = total
            best_assign = res_mcf.copy()
            best_load = load_check.copy()
            local_search(best_assign, best_load, max_passes=5)
            new_score = compute_score(best_assign)
            if new_score < best_score:
                best_score = new_score

    # Top k cheapest facilities for k = 5, 10, 15
    for k in [5, 10, 15]:
        if k > n_fac:
            continue
        top_k = np.argsort(open_cost)[:k]
        res_mcf, cost_mcf = min_cost_flow_for_open_set(top_k.tolist())
        if res_mcf is not None:
            load_check = np.zeros(n_fac)
            np.add.at(load_check, res_mcf, demands)
            used = load_check > 0
            total = cost_mcf + np.sum(open_cost[used])
            if total < best_score:
                best_score = total
                best_assign = res_mcf.copy()
                best_load = load_check.copy()
                local_search(best_assign, best_load, max_passes=5)
                new_score = compute_score(best_assign)
                if new_score < best_score:
                    best_score = new_score

    # Final local search
    local_search(best_assign, best_load, max_passes=10)
    final_score = compute_score(best_assign)
    if final_score < best_score:
        best_score = final_score

    # Verify feasibility
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        # Fallback to greedy
        best_assign, _ = greedy_init()

    return best_assign.astype(int)

# EVOLVE-BLOCK-END
