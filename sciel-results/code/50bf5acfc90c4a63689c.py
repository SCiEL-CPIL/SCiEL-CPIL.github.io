import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets
        # Two edges conflict if they share a vertex or are connected by an edge (induced matching condition)
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        # 1. Fast Greedy Initial Solution (DSATUR-like)
        # This ensures we always have a valid answer to return if ILP fails or times out.
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            while uncolored:
                best_i, best_score = None, (-1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg)
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        init_colors = greedy_dsatur()
        best_colors = init_colors[:]
        best_num_colors = max(init_colors) + 1 if init_colors else 0

        # 2. ILP Formulation
        # Variables: x_{i,k} = 1 if edge i has color k, for k in 0..K-1
        # We try to find a solution with K colors, starting from best_num_colors - 1 downwards.
        # If we find a solution with K, we can try K-1.
        
        # To make ILP efficient, we limit the number of colors K we test.
        # The Erdos-Nesetril conjecture suggests ~5d^2/4.
        # Greedy is usually close. We test K from best_num_colors-1 down to max(1, best_num_colors-3).
        
        # Precompute conflict matrix for ILP constraints
        # A[i, j] = 1 if edges i and j conflict
        # Constraint: x_{i,k} + x_{j,k} <= 1 for all conflicts (i,j) and all colors k
        
        def try_ilp(K):
            if K <= 0:
                return None
            
            # Number of variables: m * K
            n_vars = m * K
            
            # Objective: Minimize sum of all x (or just 0, since we fix K)
            # Actually, we just want to check feasibility.
            # c = zeros
            c = np.zeros(n_vars)
            
            # Bounds: 0 <= x <= 1
            lb = np.zeros(n_vars)
            ub = np.ones(n_vars)
            
            # Integrality: All variables are integer (0 or 1)
            integrality = np.ones(n_vars)
            
            # Constraints:
            # 1. Each edge i must have exactly one color: sum_k x_{i,k} = 1
            # 2. For each conflict (i, j) and color k: x_{i,k} + x_{j,k} <= 1
            
            # Build A matrix for constraints
            # We will use LinearConstraint(A, lb_cons, ub_cons)
            
            # List of rows for A
            A_rows = []
            lb_cons = []
            ub_cons = []
            
            # Constraint 1: Sum for each edge i is 1
            for i in range(m):
                row = np.zeros(n_vars)
                for k in range(K):
                    row[i * K + k] = 1.0
                A_rows.append(row)
                lb_cons.append(1.0)
                ub_cons.append(1.0)
            
            # Constraint 2: Conflicts
            # Iterate over unique pairs (i, j) where j > i and they conflict
            for i in range(m):
                for j in conflict_sets[i]:
                    if j <= i:
                        continue
                    for k in range(K):
                        row = np.zeros(n_vars)
                        row[i * K + k] = 1.0
                        row[j * K + k] = 1.0
                        A_rows.append(row)
                        lb_cons.append(-np.inf)
                        ub_cons.append(1.0)
            
            if not A_rows:
                return None
                
            A = np.array(A_rows)
            lb_cons = np.array(lb_cons)
            ub_cons = np.array(ub_cons)
            
            # SciPy milp requires integrality as a mask or array
            # integrality: 1 for integer, 0 for continuous
            # We want all variables to be integer (0 or 1)
            
            try:
                res = milp(c=c,
                           constraints=LinearConstraint(A, lb_cons, ub_cons),
                           integrality=integrality,
                           bounds=Bounds(lb, ub),
                           options={"time_limit": time_limit, "presolve": True})
                
                if res.status == 0: # Optimal found
                    # Extract colors
                    x = res.x
                    colors = [-1] * m
                    for i in range(m):
                        for k in range(K):
                            if x[i * K + k] > 0.5:
                                colors[i] = k
                                break
                        # Check if assigned
                        if colors[i] == -1:
                            return None
                    return colors
                else:
                    return None
            except Exception:
                return None

        # Try to improve using ILP
        # We have a time budget. Let's allocate most of it to the ILP attempts.
        # We try K = best_num_colors - 1, then -2, etc.
        # If ILP succeeds, update best.
        
        # To be safe, we limit the number of ILP calls.
        # Each ILP call might take some time.
        # Let's try at most 3 values of K.
        
        current_best = best_num_colors
        max_attempts = 3
        attempts = 0
        
        for k in range(current_best - 1, 0, -1):
            if attempts >= max_attempts:
                break
            if time.time() > deadline - 5.0: # Leave 5s for cleanup/next instance
                break
            
            # Allocate remaining time minus safety margin to this ILP call
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time <= 1.0:
                break
                
            # We might want to limit the time per ILP call to avoid getting stuck on one hard K
            # If we have lots of time, give it more.
            time_for_ilp = min(remaining_time, 20.0) # Cap at 20s per ILP to allow trying lower Ks if fast
            
            result_colors = try_ilp(k)
            attempts += 1
            
            if result_colors is not None:
                # Verify validity just in case
                valid = True
                for i in range(m):
                    ci = result_colors[i]
                    for j in conflict_sets[i]:
                        if result_colors[j] == ci:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current_best = k
                    best_colors = result_colors[:]
                    # Continue to try k-1
                else:
                    break # Invalid result, stop
            else:
                # If ILP failed or timed out without finding a solution, 
                # we don't know if k is possible or not.
                # If it timed out, we can't be sure.
                # If it returned infeasible (status != 0 but not timeout), then k is not possible.
                # SciPy milp status: 0 optimal, 1 iteration limit, 2 time limit, 3 infeasible, 4 unbounded
                # Since we don't easily access the status code from the helper, 
                # and we return None for both timeout and infeasible, we must be careful.
                # However, if we timeout, we should probably stop trying lower Ks because
                # if K is hard, K-1 is likely harder or equally hard.
                # But if it was infeasible, K-1 is definitely infeasible.
                # Let's assume if it returns None quickly, it might be infeasible.
                # If it takes long, it's a timeout.
                # To be safe, if we get None, we break. We already have a valid coloring with current_best.
                break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        
        # Allocate time proportionally or equally?
        # The graphs grow in size. Let's give more time to larger graphs.
        # n=40 (m~80), n=60 (m~120), n=80 (m~160).
        # Let's just use the remaining time for each, but ensure we don't overshoot.
        # Simple: use all remaining time for the last instance, split for others?
        # Better: Give each instance a fixed slice, e.g., 40s, 40s, 40s?
        # Total 120s. Deadline is 140s.
        # Let's give 45s to each.
        time_limit = 45.0
        if len(results) == 2: # Last instance
             time_limit = remaining - 2.0
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
        
        # Check time
        if time.time() > deadline - 2.0:
             # If we are running out of time, ensure we return valid results for remaining
             # by using the fallback greedy for any unsolved instances.
             # But the loop handles this by checking remaining at the top.
             pass

    return results
