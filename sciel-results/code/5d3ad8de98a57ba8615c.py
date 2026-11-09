import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    # Safety margin: The limit is 150s, let's aim for 140s to be safe.
    deadline = start_time + 140 

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.bool_)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u, v] = True
                    adj[v, u] = True
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivoting.
        Returns a list of frozensets of vertices.
        """
        cliques = []
        neighbors = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and adj[u, v]:
                    neighbors[u] |= (1 << v)

        def expand(R, P, X):
            if P == 0 and X == 0:
                if R:
                    # Convert bitmask to frozenset
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            
            # Pivot selection: choose u in P|X that maximizes |P & neighbors[u]|
            union = P | X
            if union:
                max_count = -1
                pivot = -1
                u = 0
                while u < n:
                    if (union >> u) & 1:
                        count = bin(P & neighbors[u]).count('1')
                        if count > max_count:
                            max_count = count
                            pivot = u
                    u += 1
                # Candidates are P \ neighbors[pivot]
                candidates = P & ~neighbors[pivot] if pivot >= 0 else P
            else:
                candidates = P

            v = 0
            while v < n:
                if (candidates >> v) & 1:
                    bit = 1 << v
                    new_R = R | bit
                    new_P = P & neighbors[v]
                    new_X = X & neighbors[v]
                    expand(new_R, new_P, new_X)
                    P &= ~bit
                    X |= bit
                v += 1

        expand(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        # Filter cliques of size >= 2
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []

        m = len(relevant_cliques)
        
        # --- Step 1: Preprocessing and Dominance Reduction ---
        # Map each clique to an index 0..m-1
        # For each vertex v, compute a bitmask of cliques it hits.
        # Since m can be large, we might need to handle bitmasks carefully.
        # For n=40, max cliques is usually manageable (< 1000s).
        
        # If m > 60, Python integers handle big ints fine, but operations might slow down.
        # Let's assume m is small enough for big-int bitmasks.
        
        vertex_clique_mask = [0] * n
        for i, c in enumerate(relevant_cliques):
            mask_bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= mask_bit
        
        all_cliques_mask = (1 << m) - 1

        # Dominance Reduction:
        # If vertex u's clique mask is a subset of vertex v's clique mask (mask_u & ~mask_v == 0),
        # then u is dominated by v. We can remove u from the candidate pool.
        # Note: We must be careful not to remove all vertices for a specific clique.
        # But since we are looking for a hitting set, if v covers everything u covers, 
        # picking v is always at least as good as picking u.
        
        # We will keep a list of "active" vertices.
        active_vertices = list(range(n))
        
        # Simple dominance check: O(n^2 * bitmask_op)
        # This might be slow if n is large, but n=40 is small.
        for u in range(n):
            if u not in active_vertices:
                continue
            for v in range(n):
                if u == v or v not in active_vertices:
                    continue
                # If mask_u is subset of mask_v, u is dominated by v
                if (vertex_clique_mask[u] & ~vertex_clique_mask[v]) == 0:
                    # Remove u
                    if u in active_vertices:
                        active_vertices.remove(u)
                    break # No need to check other v for this u

        if not active_vertices:
            # Fallback if something went wrong (should not happen if cliques exist)
            return list(range(n))

        # --- Step 2: Greedy Initialization ---
        # Greedy Hitting Set: Pick vertex that covers most uncovered cliques.
        uncovered_mask = all_cliques_mask
        selected = []
        current_mask = 0
        
        # To speed up greedy, we can pre-sort or just iterate.
        # Since n is small, simple iteration is fine.
        while uncovered_mask != 0:
            best_v = -1
            best_gain = 0
            for v in active_vertices:
                if v in selected:
                    continue
                gain = bin(uncovered_mask & vertex_clique_mask[v]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1 or best_gain == 0:
                # Should not happen if problem is valid
                break
            selected.append(best_v)
            current_mask |= vertex_clique_mask[best_v]
            uncovered_mask &= ~vertex_clique_mask[best_v]
            
        # Post-process: Remove redundant vertices from greedy solution
        for v in list(selected):
            # Check if selected without v still covers all
            temp_mask = 0
            for u in selected:
                if u != v:
                    temp_mask |= vertex_clique_mask[u]
            if temp_mask == all_cliques_mask:
                selected.remove(v)
        
        best_sol = selected
        best_size = len(best_sol)
        
        # --- Step 3: Local Search ---
        # 1-swap and 2-swap local search
        # Using bitmasks for fast coverage checks
        
        rng = random.Random(42)
        ls_start = time.time()
        # Allocate ~20% of remaining time to local search
        time_left_total = deadline - ls_start
        ls_deadline = ls_start + max(1.0, time_left_total * 0.2)
        
        improved = True
        while improved and time.time() < ls_deadline:
            improved = False
            sol_set = set(best_sol)
            
            # 1-Remove
            for v in best_sol:
                temp_mask = 0
                for u in best_sol:
                    if u != v:
                        temp_mask |= vertex_clique_mask[u]
                if temp_mask == all_cliques_mask:
                    best_sol.remove(v)
                    best_size -= 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # 1-Swap
            found_swap = False
            for v in best_sol:
                # Find cliques only covered by v
                others_mask = 0
                for u in best_sol:
                    if u != v:
                        others_mask |= vertex_clique_mask[u]
                unique_to_v = vertex_clique_mask[v] & ~others_mask
                
                if unique_to_v == 0:
                    continue # v is redundant, should have been caught by 1-Remove
                
                # Find a vertex w not in sol that covers unique_to_v
                for w in active_vertices:
                    if w in sol_set:
                        continue
                    if (vertex_clique_mask[w] & unique_to_v) == unique_to_v:
                        # Check if swapping v with w maintains coverage
                        # New mask = (others_mask) | mask[w]
                        if (others_mask | vertex_clique_mask[w]) == all_cliques_mask:
                            best_sol.remove(v)
                            best_sol.append(w)
                            sol_set.add(w)
                            sol_set.remove(v)
                            found_swap = True
                            break
                if found_swap:
                    improved = True
                    break
            
            if found_swap:
                # Try to remove again after swap
                continue
                
            # 2-Swap (optional, might be slow)
            # Try to remove 2 vertices and add 1 or 2
            # For now, stick to 1-swap for speed, maybe add 2-swap if time permits
            
            # Random Perturbation if stuck
            if rng.random() < 0.05:
                # Add a random vertex
                candidates = [v for v in active_vertices if v not in best_sol]
                if candidates:
                    v = rng.choice(candidates)
                    new_sol = best_sol + [v]
                    # Check validity
                    temp_mask = 0
                    for u in new_sol:
                        temp_mask |= vertex_clique_mask[u]
                    if temp_mask == all_cliques_mask:
                        best_sol = new_sol
                        best_size = len(best_sol)
                        # Cleanup
                        for rv in list(best_sol):
                            temp_mask = 0
                            for u in best_sol:
                                if u != rv:
                                    temp_mask |= vertex_clique_mask[u]
                            if temp_mask == all_cliques_mask:
                                best_sol.remove(rv)
                                best_size -= 1
                                break

        # --- Step 4: MILP ---
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        
        if time_left > 2:
            try:
                # Build sparse matrix for MILP
                # Variables: x_v for v in active_vertices
                # Constraints: sum_{v in C_i} x_v >= 1 for each clique i
                # Objective: min sum x_v
                
                # Map active vertices to indices 0..k-1
                k = len(active_vertices)
                v_to_idx = {v: i for i, v in enumerate(active_vertices)}
                
                rows = []
                cols = []
                data = []
                
                for i, clique in enumerate(relevant_cliques):
                    for v in clique:
                        if v in v_to_idx:
                            rows.append(i)
                            cols.append(v_to_idx[v])
                            data.append(1)
                
                A = sp.csr_matrix((data, (rows, cols)), shape=(m, k))
                
                c_obj = np.ones(k, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                integrality = np.ones(k, dtype=np.int8)
                
                # Warm start
                incumbent = np.zeros(k)
                for v in best_sol:
                    if v in v_to_idx:
                        incumbent[v_to_idx[v]] = 1
                
                # Allocate remaining time to MILP, minus a small buffer
                instance_time = max(1.0, time_left - 5)
                
                res = milp(c=c_obj, constraints=constraints, integrality=integrality,
                           bounds=bounds_obj,
                           options={"time_limit": instance_time, "presolve": True,
                                    "MIP_start": incumbent})
                
                if res.x is not None and res.status in [0, 1, 2]: # 0: optimal, 1: time limit, 2: ?
                    milp_sol = [active_vertices[i] for i in range(k) if res.x[i] > 0.5]
                    # Verify validity
                    if len(milp_sol) > 0:
                        sol_set = set(milp_sol)
                        valid = True
                        for c in relevant_cliques:
                            if not sol_set & c:
                                valid = False
                                break
                        if valid:
                            if len(milp_sol) < best_size:
                                best_sol = milp_sol
                                best_size = len(best_sol)
            except Exception:
                pass

        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, use greedy for remaining
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            if relevant_cliques:
                # Quick greedy
                vertex_clique_mask = [0] * n
                for i, c in enumerate(relevant_cliques):
                    mask_bit = 1 << i
                    for v in c:
                        vertex_clique_mask[v] |= mask_bit
                all_cliques_mask = (1 << len(relevant_cliques)) - 1
                uncovered_mask = all_cliques_mask
                selected = []
                while uncovered_mask != 0:
                    best_v = -1
                    best_gain = 0
                    for v in range(n):
                        if v in selected: continue
                        gain = bin(uncovered_mask & vertex_clique_mask[v]).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    if best_v == -1: break
                    selected.append(best_v)
                    uncovered_mask &= ~vertex_clique_mask[best_v]
                # Cleanup
                for v in list(selected):
                    temp_mask = 0
                    for u in selected:
                        if u != v: temp_mask |= vertex_clique_mask[u]
                    if temp_mask == all_cliques_mask:
                        selected.remove(v)
                results.append(sorted(selected))
            else:
                results.append([])
            continue

        sol = solve_instance(n, p, seed)
        results.append(sol)

    while len(results) < 3:
        results.append([])

    return results[:3]
