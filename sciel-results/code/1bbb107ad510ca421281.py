"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    def grasp_construct(alpha=0.5):
        """
        Construct a solution using GRASP.
        Greedy choice: minimize marginal cost increase.
        Randomized: pick from top 10% (or alpha fraction) of candidates.
        """
        p = np.empty(n, dtype=np.int64)
        assigned_locs = np.zeros(n, dtype=bool)
        assigned_facilities = np.zeros(n, dtype=bool)
        
        # We need to track current cost incrementally? 
        # For n=150, n^2 * n = 3.3M ops per construction. 
        # If we do 100 constructions, 330M ops. Might be tight but numpy is fast.
        # However, calculating marginal cost for all unassigned pairs is O(n^2) per step -> O(n^3) total.
        # n=150 -> 150^3 = 3.375M per construction. 
        # 100 constructions -> 337M. In Python/NumPy, this might take ~1-2 seconds if vectorized well.
        
        # Vectorized GRASP Construction
        # State: 
        # unassigned_facs: list of indices
        # unassigned_locs: list of indices
        # current_cost: scalar
        # marginal_cost[i, j] for i in unassigned_facs, j in unassigned_locs
        
        unassigned_facs = list(range(n))
        unassigned_locs = list(range(n))
        
        # Precompute b for fast access? b is fixed.
        # Marginal cost of assigning facility i to location j given current assignment P_partial:
        # delta = sum_{k in assigned} a[i,k] b[j, p[k]] + a[k,i] b[p[k], j]
        #       + sum_{m in unassigned, m != i} a[i,m] * 0 (future terms not counted yet? No, QAP is sum over all pairs)
        # Actually, standard QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
        # If we build p incrementally, the cost contribution of placing i at j involves:
        # 1. Terms where i is first index: sum_{k} a[i,k] b[j, p[k]]
        # 2. Terms where i is second index: sum_{k} a[k,i] b[p[k], j]
        # Note: k includes already assigned facilities. It does NOT include unassigned ones yet 
        # (those will be accounted for when they are placed).
        # Wait, if we place i at j, the terms involving i and any previously placed k are fixed.
        # The terms involving i and future m will be added when m is placed.
        # So the marginal cost is:
        # C(i, j) = sum_{k in assigned} (a[i,k] * b[j, p[k]] + a[k,i] * b[p[k], j])
        
        # Let's maintain a vector `row_contrib` and `col_contrib`?
        # row_contrib[j] = sum_{k in assigned} a[?, k] ... no, depends on i.
        
        # Efficient way:
        # Let B_assigned be the submatrix of b for assigned locations.
        # Let A_i be row i of a, A^T_i be col i of a.
        # Cost = A_i @ B_assigned[:, p_assigned] + A^T_i @ B_assigned[p_assigned, :]
        
        # Since n=150 is small, we can just loop and use numpy vectors for the sum.
        
        assigned_p = {} # facility -> location
        cost = 0
        
        for _ in range(n):
            if not unassigned_facs: break
            
            # Calculate marginal costs for all (i, j) where i in unassigned_facs, j in unassigned_locs
            # We want the smallest marginal cost.
            
            # Optimize: 
            # For each i in unassigned_facs:
            #   vec_j = sum_{k in assigned} (a[i,k] * b[:, p[k]] + a[k,i] * b[p[k], :])
            #   This is a vector of length n. We only care about j in unassigned_locs.
            
            # Let's precompute for current assigned set:
            # M_row[j] = sum_{k in assigned} a[?, k] ... no.
            
            # Let's just do it directly.
            # n=150. Loop 150 times.
            # Inside loop, calculate costs for ~150 * 150 pairs.
            # Total ops ~ 150 * 150 * 150 = 3.3M.
            
            costs = np.full((len(unassigned_facs), len(unassigned_locs)), np.inf)
            
            # Get indices of assigned facilities and their locations
            assigned_facs = [f for f in range(n) if f not in unassigned_facs]
            # This list comprehension is slow. Let's maintain a boolean array.
            
            # Better: maintain `is_assigned` boolean array.
            
            # Let's restructure to maintain `is_assigned_f` and `is_assigned_l`
            
            pass
            
        # Given the complexity of perfectly vectorizing the incremental GRASP construction in pure Python/NumPy 
        # without significant overhead from list management, and the fact that n=150 is small enough 
        # that a simple O(n^3) construction in NumPy might be fast enough, let's try a simpler approach:
        # Generate several random permutations, sort by cost, and pick the best? 
        # No, GRASP is supposed to be smarter than random.
        
        # Alternative: Use the rank matching as a base, then perturb it randomly to create diverse starts?
        # Or, implement a simpler greedy: 
        # 1. Sort facilities by flow sum (descending).
        # 2. For the first facility, pick the location with min distance sum.
        # 3. For the next, pick the location that minimizes interaction with previously placed.
        
        # Let's implement a "Greedy with Randomization" where we pick from the top K candidates at each step.
        
        is_assigned_f = np.zeros(n, dtype=bool)
        is_assigned_l = np.zeros(n, dtype=bool)
        p = np.empty(n, dtype=np.int64)
        
        # Precompute row/col sums of a for initial sorting?
        # Let's stick to the marginal cost calculation.
        
        # To speed up marginal cost:
        # Maintain `S_row[j]` and `S_col[j]`? 
        # When we assign facility f to location l:
        # For any unassigned facility i, the new marginal cost contribution from f is:
        # a[i,f] * b[l, p[f]] (which is b[l,l]=0? No, p[f]=l) -> a[i,f] * b[l, l] ? No.
        # The term is a[i,f] * b[p[i], p[f]] = a[i,f] * b[p[i], l].
        # And a[f,i] * b[p[f], p[i]] = a[f,i] * b[l, p[i]].
        
        # So if we maintain for each unassigned location j:
        # `potential[j]` for a given facility i? No, it depends on i.
        
        # Let's just calculate the vector of marginal costs for all unassigned locations for a specific facility i.
        # cost_vec[j] = sum_{f in assigned} (a[i,f] * b[j, p[f]] + a[f,i] * b[p[f], j])
        
        # We can precompute two matrices:
        # B_assigned = b[:, p[assigned]] (n x k)
        # B_assigned_T = b[p[assigned], :] (k x n)
        
        # Then cost_vec = a[i, assigned_facs] @ B_assigned + a[assigned_facs, i] @ B_assigned_T
        # This is O(n*k) per facility i.
        # Total per step: O(n^2 * k).
        # Total for construction: O(n^3).
        
        # Let's implement this.
        
        assigned_facs = []
        assigned_locs = []
        
        # We need to pick facilities in some order? 
        # Standard GRASP picks the "most constrained" or just iterates?
        # Usually, we pick the facility that has the smallest range of costs or similar.
        # For simplicity, let's pick the facility with the highest flow sum first (heuristic), 
        # then randomize the choice of location from the RCL.
        
        # Actually, true GRASP constructs by picking one pair (i,j) at a time.
        # Let's do that.
        
        unassigned_facs_set = set(range(n))
        unassigned_locs_set = set(range(n))
        
        while unassigned_facs_set:
            # Calculate marginal costs for all (i, j)
            # i in unassigned_facs_set, j in unassigned_locs_set
            
            # To speed up, we can calculate for all i first.
            # costs[i, j] = sum_{f in assigned} (a[i,f] b[j, p[f]] + a[f,i] b[p[f], j])
            
            if not assigned_facs:
                # First assignment: cost is 0 for all? 
                # Wait, cost is sum a[i,j] b[p[i], p[j]].
                # If only i is assigned to j, the terms are a[i,i]b[j,j] + ... 
                # Usually diagonal is 0.
                # If we assume diagonal 0, first move cost is 0.
                # But we should still consider future interactions? 
                # No, marginal cost only considers already assigned.
                # So first move has 0 marginal cost.
                # We should pick based on a heuristic for the first move to set the stage.
                # Let's pick the facility with max flow sum and location with min dist sum.
                # Or randomize slightly.
                
                # Let's just pick the best rank-matching pair for the first step to anchor.
                # Or, calculate the "potential" cost if we assume uniform assignment?
                
                # For simplicity and speed, let's use the rank matching for the first 10% of assignments, 
                # then switch to randomized marginal cost?
                
                # Actually, let's just compute the full cost matrix for the first step?
                # It's 0. So we are free.
                # Let's pick i with max a.sum(axis=1) and j with min b.sum(axis=1).
                
                i = int(np.argmax(a.sum(axis=1)))
                j = int(np.argmin(b.sum(axis=1)))
                
                # Randomize?
                # Let's just do it deterministically for the first few to get a good base, 
                # then randomize.
                
            else:
                # Calculate costs
                # assigned_facs is a list of ints
                # p[assigned_facs] gives the locations
                
                # Vectorized calculation for all unassigned i
                # Let U_facs = list(unassigned_facs_set)
                # Let U_locs = list(unassigned_locs_set)
                
                # We want costs for i in U_facs, j in U_locs.
                
                # B_col = b[:, p[assigned_facs]] # shape (n, k)
                # B_row = b[p[assigned_facs], :] # shape (k, n)
                
                # A_sub = a[np.ix_(U_facs, assigned_facs)] # shape (n_u, k)
                # A_sub_T = a[np.ix_(assigned_facs, U_facs)] # shape (k, n_u)
                
                # cost_mat = A_sub @ B_col[np.ix_(np.arange(n), U_locs)] + A_sub_T @ B_row[np.ix_(np.arange(n), U_locs)]
                
                # This is heavy.
                
                # Let's approximate or simplify.
                # Given the time limit, maybe a simpler diverse initialization is better.
                
                pass
        
        # This GRASP construction is getting complex and might be slow in Python.
        # Let's pivot to a simpler structural change:
        # 1. Generate 5 diverse initial solutions using different heuristics (Rank, Random, Greedy).
        # 2. Run the fast local search on each.
        # 3. Return the best.
        
        # Heuristic 1: Rank Matching (current)
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        heavy_order = np.argsort(-a_sum, kind="stable")
        central_order = np.argsort(b_sum, kind="stable")
        p1 = np.empty(n, dtype=np.int64)
        p1[heavy_order] = central_order
        
        # Heuristic 2: Random
        p2 = np.random.permutation(n)
        
        # Heuristic 3: Greedy (place heavy facility at best location, remove, repeat)
        p3 = np.empty(n, dtype=np.int64)
        remaining_facs = list(range(n))
        remaining_locs = list(range(n))
        # Sort facs by flow sum descending
        facs_sorted = sorted(remaining_facs, key=lambda x: -a_sum[x])
        for i in facs_sorted:
            # Find location j in remaining_locs that minimizes interaction with already placed
            # Interaction = sum_{k placed} (a[i,k] b[j, p[k]] + a[k,i] b[p[k], j])
            # If none placed, pick min b_sum
            if not (remaining_locs): break
            if len(remaining_locs) == 1:
                j = remaining_locs[0]
            else:
                # Calculate scores for all remaining locs
                # placed_facs = [f for f in facs_sorted if f not in remaining_facs]
                # This is slow.
                
                # Let's just pick the one with min b_sum for simplicity in this heuristic variant
                j = min(remaining_locs, key=lambda l: b_sum[l])
            
            p3[i] = j
            remaining_facs.remove(i)
            remaining_locs.remove(j)
            
        # Heuristic 4: Randomized Rank (swap a few in rank matching)
        p4 = p1.copy()
        for _ in range(10):
            i, j = np.random.choice(n, 2, replace=False)
            p4[i], p4[j] = p4[j], p4[i]
            
        candidates = [p1, p2, p3, p4]
        
        best_p = None
        best_cost = np.inf
        
        for p in candidates:
            # Run local search
            # We need a fast local search.
            # The current one is O(n^2) per step.
            # With 115s, we can do many steps.
            
            # Let's implement a simple local search for each candidate
            # and keep the best result.
            
            # To save time, we'll run local search on the best 2 candidates based on initial cost?
            # Or just run on all if fast enough.
            
            cost = calc_cost(p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
            # Local search on p
            # Use the same logic as before but optimized?
            # The current code's local search is the bottleneck if not optimized.
            # Let's use the "incremental update" insight from the knowledge base.
            
            # But implementing incremental update for 2-swap in NumPy is tricky.
            # The current code uses `np.dot(ai - a[j], Bp[v] - Bp[u])` which is O(n) per pair (i,j).
            # Total O(n^3) per iteration? No, O(n^2) pairs * O(n) dot = O(n^3) per full sweep.
            # n=150 -> 3.3M ops per sweep.
            # 1000 sweeps -> 3.3B ops. Too slow for 115s?
            # NumPy dot is fast. 3.3M ops is ~1ms? 
            # 1000 sweeps -> 1s. So we can do ~1000 sweeps in 115s.
            
            # Let's run local search on the best candidate only to maximize depth?
            # Or run on all?
            
            # Strategy: Run local search on all 4 candidates, but with a time budget split.
            # Or, run local search on the best initial solution only.
            
            # Let's run local search on the best initial solution.
            # First, find the best initial solution among candidates.
            
            pass
            
        # Find best initial
        init_costs = [calc_cost(p) for p in candidates]
        best_init_idx = np.argmin(init_costs)
        current_p = candidates[best_init_idx].copy()
        current_cost = init_costs[best_init_idx]
        
        # Local Search
        tabu_size = 30
        tabu_list = {}
        current_step = 0
        no_improve_count = 0
        max_no_improve = 1000
        
        Bp = b[:, current_p]
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                # Perturb
                for _ in range(5):
                    i, j = np.random.choice(n, 2, replace=False)
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost = calc_cost(current_p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                no_improve_count = 0
                tabu_list.clear()
                Bp = b[:, current_p]
                continue
                
            best_delta = np.inf
            best_i = -1
            best_j = -1
            
            # Find best move
            # We can sample a subset of pairs to speed up?
            # Or do full sweep?
            # Full sweep is O(n^3).
            
            for i in range(n):
                u = current_p[i]
                ai = a[i]
                
                for j in range(i + 1, n):
                    key = (i, j)
                    if key in tabu_list and tabu_list[key] > current_step:
                        continue
                        
                    v = current_p[j]
                    delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        
            if best_i == -1:
                tabu_list.clear()
                continue
                
            # Perform swap
            u = current_p[best_i]
            v = current_p[best_j]
            current_p[best_i], current_p[best_j] = v, u
            current_cost += int(best_delta)
            
            # Update Bp
            # Bp[:, best_i] was b[:, u], now b[:, v]
            # Bp[:, best_j] was b[:, v], now b[:, u]
            # We can just rebuild Bp? O(n^2).
            # Or update columns.
            Bp[:, best_i] = b[:, v]
            Bp[:, best_j] = b[:, u]
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            tabu_list[(best_i, best_j)] = current_step + tabu_size
            tabu_list[(best_j, best_i)] = current_step + tabu_size
            current_step += 1
            
            if current_step % 100 == 0:
                keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
                for k in keys_to_remove:
                    del tabu_list[k]
                    
        return best_p.tolist()

# EVOLVE-BLOCK-END
