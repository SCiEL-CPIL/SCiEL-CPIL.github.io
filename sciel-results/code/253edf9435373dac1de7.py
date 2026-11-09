import time
import random
import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

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
        """Returns list of maximal cliques as frozensets using bitmasks"""
        cliques = []
        # Precompute neighbor bitmasks
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
            
            # Pivot selection
            union = P | X
            if union:
                # Choose pivot with maximum |P & neighbors[u]|
                max_count = -1
                pivot = -1
                # Iterate through bits in union
                u = 0
                while u < n:
                    if (union >> u) & 1:
                        count = bin(P & neighbors[u]).count('1')
                        if count > max_count:
                            max_count = count
                            pivot = u
                    u += 1
                candidates = P & ~neighbors[pivot] if pivot >= 0 else P
            else:
                candidates = P

            # Iterate through candidates
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

    def is_valid(sol, cliques):
        sol_set = set(sol)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_hitting_set(cliques, n):
        """Greedy hitting set: repeatedly pick vertex hitting most uncovered cliques"""
        # Count how many cliques each vertex hits
        vertex_counts = [0] * n
        for c in cliques:
            for v in c:
                vertex_counts[v] += 1
        
        # Sort vertices by count descending
        vertices = sorted(range(n), key=lambda v: (-vertex_counts[v], v))
        
        uncovered = [True] * len(cliques)
        num_uncovered = len(cliques)
        selected = []
        
        while num_uncovered > 0:
            # Find vertex that hits most uncovered cliques
            best_v = -1
            best_hits = 0
            for v in vertices:
                if v in selected:
                    continue
                hits = 0
                for i, c in enumerate(cliques):
                    if uncovered[i] and v in c:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_v = v
            
            if best_v == -1 or best_hits == 0:
                break
            
            selected.append(best_v)
            for i, c in enumerate(cliques):
                if uncovered[i] and best_v in c:
                    uncovered[i] = False
                    num_uncovered -= 1
        
        # Post-processing: remove unnecessary vertices
        sol_set = set(selected)
        for v in list(selected):
            test_set = sol_set - {v}
            ok = True
            for c in cliques:
                if not test_set & c:
                    ok = False
                    break
            if ok:
                selected.remove(v)
                sol_set = test_set
        
        return selected

    def greedy_init(cliques, n):
        """Greedy hitting set initialization"""
        return greedy_hitting_set(cliques, n)

    def local_search(sol, cliques, n, max_iters=5000):
        """Local search to minimize solution size. Returns improved solution."""
        best_sol = list(sol)
        best_size = len(best_sol)
        cliques_list = list(cliques)
        
        # Precompute: for each vertex, bitmask of cliques it covers
        num_cliques = len(cliques_list)
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)
        
        # Mask of all cliques
        all_cliques_mask = (1 << num_cliques) - 1
        
        def covers_all(sol_set):
            mask = 0
            for v in sol_set:
                mask |= vertex_clique_mask[v]
            return mask == all_cliques_mask
        
        if not covers_all(best_sol):
            sol_set = set(best_sol)
            for c in cliques_list:
                if not sol_set & c:
                    best_sol.append(min(c))
            best_size = len(best_sol)
        
        rng = random.Random(42)
        ls_start = time.time()
        ls_deadline = ls_start + max(1.0, (deadline - (time.time() - start_time)) * 0.8)
        
        for it in range(max_iters):
            if time.time() > ls_deadline:
                break
            improved = False
            sol_set = set(best_sol)
            
            # Try removing a vertex
            for v in best_sol:
                new_mask = 0
                for u in best_sol:
                    if u != v:
                        new_mask |= vertex_clique_mask[u]
                if new_mask == all_cliques_mask:
                    best_sol.remove(v)
                    best_size -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap
            found_swap = False
            for v in best_sol:
                # Find cliques that only v covers
                only_v_mask = 0
                for u in best_sol:
                    if u != v:
                        only_v_mask |= vertex_clique_mask[u]
                v_unique = vertex_clique_mask[v] & ~only_v_mask
                if v_unique == 0:
                    continue  # v is redundant, should have been caught
                
                # Find w that covers all of v_unique
                candidates = [w for w in range(n) if w not in sol_set and (vertex_clique_mask[w] & v_unique) == v_unique]
                for w in candidates:
                    # Verify all cliques covered after swap
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    new_mask |= vertex_clique_mask[w]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_sol.append(w)
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                # Try to remove vertices after swap
                sol_set = set(best_sol)
                for v in list(best_sol):
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_size -= 1
                        break
                continue
            
            # If no improvement, try random perturbation
            if rng.random() < 0.05:
                candidates = [v for v in range(n) if v not in best_sol]
                if candidates:
                    v = rng.choice(candidates)
                    new_sol = best_sol + [v]
                    if covers_all(new_sol):
                        best_sol = new_sol
                        best_size = len(best_sol)
                        # Try to clean up
                        for rv in list(best_sol):
                            new_mask = 0
                            for u in best_sol:
                                if u != rv:
                                    new_mask |= vertex_clique_mask[u]
                            if new_mask == all_cliques_mask:
                                best_sol.remove(rv)
                                best_size -= 1
                                break
        
        return best_sol

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2:
            return greedy_init(relevant_cliques, n)

        # Get greedy solution
        sol = greedy_init(relevant_cliques, n)
        best_sol = sol
        best_size = len(sol)

        # Local search to improve
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        ls_iters = max(500, min(3000, int(time_left * 50)))
        sol2 = local_search(sol, relevant_cliques, n, max_iters=ls_iters)
        if len(sol2) < best_size:
            best_sol = sol2
            best_size = len(sol2)

        # Run local search again from the improved solution
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5 and len(best_sol) > 0:
            ls_iters2 = max(200, min(1500, int(time_left * 30)))
            sol3 = local_search(best_sol, relevant_cliques, n, max_iters=ls_iters2)
            if len(sol3) < best_size:
                best_sol = sol3
                best_size = len(sol3)

        # Try MILP if time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 10:
            try:
                m = len(relevant_cliques)
                # Build sparse matrix for MILP
                rows = []
                cols = []
                data = []
                for i, clique in enumerate(relevant_cliques):
                    for v in clique:
                        rows.append(i)
                        cols.append(v)
                        data.append(1)
                
                A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
                
                c_obj = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                integrality = np.ones(n, dtype=np.int8)

                # Use incumbent to help
                incumbent = np.zeros(n)
                for v in best_sol:
                    incumbent[v] = 1

                instance_time = min(time_left - 5, 20)
                res = milp(c=c_obj, constraints=constraints, integrality=integrality,
                           bounds=bounds_obj,
                           options={"time_limit": instance_time, "presolve": True,
                                    "MIP_start": incumbent})

                if res.x is not None:
                    milp_sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid(milp_sol, relevant_cliques):
                        if len(milp_sol) < best_size:
                            best_sol = milp_sol
                            best_size = len(milp_sol)
            except Exception:
                pass

        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, use greedy for remaining
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            sol = greedy_init(relevant_cliques, n) if relevant_cliques else []
            results.append(sorted(sol))
            continue

        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
