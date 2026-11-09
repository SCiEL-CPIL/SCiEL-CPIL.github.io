import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Uses bitmasks for speed.
        """
        neigh = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            neigh[u] = mask
        
        cliques = []
        all_mask = (1 << n) - 1
        
        def expand(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            union = P | X
            if union:
                best_u = -1
                best_count = -1
                u = union
                while u:
                    lsb = u & (-u)
                    v = lsb.bit_length() - 1
                    u ^= lsb
                    count = bin(P & neigh[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_u = v
                u = best_u
            else:
                u = None
                
            if u is not None:
                candidates = P & ~neigh[u]
            else:
                candidates = P
                
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_R = R | lsb
                new_P = P & neigh[v]
                new_X = X & neigh[v]
                expand(new_R, new_P, new_X)
                
                P &= ~lsb
                X |= lsb
                
        expand(0, all_mask, 0)
        return cliques

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def greedy_hit_set(cliques, n):
        remaining_cliques = [set(c) for c in cliques]
        covered_cliques = [False] * len(remaining_cliques)
        selected = []
        
        while True:
            if all(covered_cliques):
                break
            
            hits = [0] * n
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i]:
                    for v in c:
                        hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break
                
            v = hits.index(max_hits)
            selected.append(v)
            
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i] and v in c:
                    covered_cliques[i] = True
                    
        return selected

    def solve_mwis_lp_bound(weights, adj_mask, n):
        """
        Solve MWIS via branch-and-bound with LP relaxation bounds.
        weights: list of n weights
        adj_mask: list of n bitmasks, adj_mask[i] has bit j set if i-j is an edge
        Returns: (best_weight, best_set_mask)
        """
        best_weight = [0.0]
        best_mask = [0]
        node_count = [0]
        
        def lp_relaxation(weights_sub, adj_sub, n_sub, time_limit=0.1):
            """
            Solve LP relaxation of MWIS on the subgraph.
            Maximize sum(w_i * x_i)
            s.t. x_i + x_j <= 1 for each edge (i,j)
                 0 <= x_i <= 1
            Returns upper bound (dual or primal objective).
            """
            if n_sub == 0:
                return 0.0
            
            # Simple fractional vertex cover dual:
            # Minimize sum(y_i) s.t. y_i + y_j >= 1 for each edge, y_i >= 0
            # This gives a lower bound on MWIS... wait, we need upper bound.
            # Actually, for MWIS, the LP relaxation is:
            # Max sum(w_i x_i) s.t. x_i + x_j <= 1 for edges, 0<=x<=1
            # The dual is: Min sum(z_e) s.t. sum_{e incident to i} z_e >= w_i, z_e >= 0
            # Let's just use a simple heuristic upper bound: sum of positive weights
            # Or better: use a greedy fractional solution
            
            # Simple upper bound: sum of all positive weights in subgraph
            total_w = sum(max(0, w) for w in weights_sub)
            
            # Better: use the fact that for any clique, sum of weights <= max weight in clique
            # But computing max clique is expensive.
            # Use a simple bound: for each vertex, it can coexist with at most one of its neighbors
            # Upper bound = sum over vertices of max(0, w_i) but that's loose.
            
            # Try a simple LP solve using scipy if available
            try:
                from scipy.optimize import linprog
                m = n_sub
                # Variables: x_0, ..., x_{n_sub-1}
                # Objective: maximize sum(w_i x_i) -> minimize -sum(w_i x_i)
                c_obj = [-w for w in weights_sub]
                
                # Constraints: x_i + x_j <= 1 for each edge
                A_ub = []
                b_ub = []
                for i in range(n_sub):
                    for j in range(i+1, n_sub):
                        if adj_sub[i] & (1 << j):
                            row = [0.0] * m
                            row[i] = 1.0
                            row[j] = 1.0
                            A_ub.append(row)
                            b_ub.append(1.0)
                
                if not A_ub:
                    # No edges, take all positive weight vertices
                    return sum(max(0, w) for w in weights_sub)
                
                res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                             bounds=[(0, 1)] * m, method='highs')
                if res.success:
                    return -res.fun
                else:
                    return total_w
            except Exception:
                return total_w
        
        def branch_and_bound(idx, current_mask, current_weight, P, X, time_start, time_limit):
            """
            idx: not used directly, we use P and X bitmasks
            current_mask: bitmask of selected vertices
            current_weight: weight of selected vertices
            P: candidates that can still be added
            X: vertices that were excluded (for maximality)
            """
            if time.time() - time_start > time_limit:
                return
            
            node_count[0] += 1
            
            if not P and not X:
                if current_weight > best_weight[0]:
                    best_weight[0] = current_weight
                    best_mask[0] = current_mask
                return
            
            # Upper bound: current_weight + LP bound on remaining
            # Extract subgraph induced by P
            p_vertices = []
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                p_vertices.append(v)
                temp ^= lsb
            
            if not p_vertices:
                if current_weight > best_weight[0]:
                    best_weight[0] = current_weight
                    best_mask[0] = current_mask
                return
            
            # Simple upper bound: current_weight + sum of positive weights in P
            # This is valid but loose
            simple_ub = current_weight + sum(max(0, weights[v]) for v in p_vertices)
            
            if simple_ub <= best_weight[0]:
                return
            
            # Try LP bound for tighter pruning (only for small subproblems)
            if len(p_vertices) <= 30:
                # Build subgraph
                w_sub = [weights[v] for v in p_vertices]
                adj_sub = [0] * len(p_vertices)
                for i, vi in enumerate(p_vertices):
                    for j, vj in enumerate(p_vertices):
                        if i != j and adj_mask[vi] & (1 << vj):
                            adj_sub[i] |= (1 << j)
                lp_ub = current_weight + lp_relaxation(w_sub, adj_sub, len(p_vertices))
                if lp_ub <= best_weight[0]:
                    return
            
            # Pivot selection
            union = P | X
            if union:
                best_u = -1
                best_count = -1
                u_iter = union
                while u_iter:
                    lsb = u_iter & (-u_iter)
                    v = lsb.bit_length() - 1
                    u_iter ^= lsb
                    count = bin(P & adj_mask[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_u = v
                pivot = best_u
            else:
                pivot = None
            
            if pivot is not None:
                candidates = P & ~adj_mask[pivot]
            else:
                candidates = P
            
            # Branch: try each candidate
            # Order by weight descending for better early solutions
            cand_list = []
            temp_c = candidates
            while temp_c:
                lsb = temp_c & (-temp_c)
                v = lsb.bit_length() - 1
                cand_list.append((weights[v], v))
                temp_c ^= lsb
            cand_list.sort(reverse=True)
            
            for w_val, v in cand_list:
                if time.time() - time_start > time_limit:
                    return
                
                new_mask = current_mask | (1 << v)
                new_weight = current_weight + weights[v]
                new_P = P & adj_mask[v]
                new_X = X & adj_mask[v]
                
                branch_and_bound(None, new_mask, new_weight, new_P, new_X, time_start, time_limit)
                
                P &= ~(1 << v)
                X |= (1 << v)
            
            return
        
        # Initial upper bound
        all_mask = (1 << n) - 1
        time_start = time.time()
        branch_and_bound(0, 0, 0.0, all_mask, 0, time_start, time_limit)
        
        return best_weight[0], best_mask[0]

    def solve_instance(n, p, seed, time_limit=30):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if bin(c).count('1') >= 2]
        
        if not relevant_cliques:
            return []
        
        cliques_as_sets = []
        for c in relevant_cliques:
            s = set()
            temp = c
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                s.add(v)
                temp ^= lsb
            cliques_as_sets.append(s)
        
        # Reduction to MWIS on line graph of complement
        # The complement graph has edges where original has no edge.
        # The line graph of the complement has vertices = edges of complement.
        # Two vertices in line graph are adjacent if corresponding edges share an endpoint.
        # Selecting vertex i in transversal means we exclude all complement edges incident to i.
        # So the independent set in line graph corresponds to a set of complement edges no two of which share an endpoint,
        # AND no selected transversal vertex is incident to any of these edges.
        # Wait, let me rethink.
        
        # Alternative: The problem is minimum vertex cover in a hypergraph where hyperedges are cliques.
        # This is NP-hard. Let's just use a good heuristic + local search.
        
        # Better approach: Use the fact that for sparse random graphs, the transversal is small.
        # Try multiple strategies and take the best.
        
        best_sol = greedy_hit_set(cliques_as_sets, n)
        best_size = len(best_sol)
        
        # Strategy 1: Greedy with different orderings
        for trial in range(5):
            sol = greedy_hit_set(cliques_as_sets, n)
            if len(sol) < best_size:
                best_sol = sol
                best_size = len(sol)
        
        # Strategy 2: Local search - try to remove vertices and re-add if needed
        for trial in range(3):
            current_sol = set(best_sol)
            improved = True
            while improved:
                improved = False
                # Try removing each vertex
                for v in list(current_sol):
                    candidate = current_sol - {v}
                    if is_valid_sol(candidate, cliques_as_sets):
                        current_sol = candidate
                        improved = True
                        break
                # Try swapping: remove one, add another
                if not improved:
                    for v_remove in list(current_sol):
                        for v_add in range(n):
                            if v_add in current_sol:
                                continue
                            candidate = (current_sol - {v_remove}) | {v_add}
                            if len(candidate) < len(current_sol) and is_valid_sol(candidate, cliques_as_sets):
                                current_sol = candidate
                                improved = True
                                break
                        if improved:
                            break
            if len(current_sol) < best_size:
                best_sol = list(current_sol)
                best_size = len(best_sol)
        
        # Strategy 3: Use scipy MILP if available and time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 10 and best_size > 0:
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                
                m = len(cliques_as_sets)
                A = np.zeros((m, n), dtype=np.int8)
                for i, clique in enumerate(cliques_as_sets):
                    for v in clique:
                        A[i, v] = 1
                
                c = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                
                instance_time_limit = min(time_limit, max(5, time_left - 5))
                
                res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                           bounds=bounds_obj, 
                           options={"time_limit": instance_time_limit, "presolve": True})
                
                if res.x is not None:
                    sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid_sol(sol, cliques_as_sets) and len(sol) < best_size:
                        best_sol = sol
                        best_size = len(sol)
            except Exception:
                pass
        
        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 5:
            break
        sol = solve_instance(n, p, seed, time_limit=min(30, max(5, time_left - 5)))
        results.append(sorted(sol))
        
        if time.time() > deadline:
             break
             
    while len(results) < 3:
         results.append([])
         
    return results
