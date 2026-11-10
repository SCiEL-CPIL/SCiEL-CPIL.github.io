import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    N = 256
    max_sum = 2 * N

    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # 2. Filter vertices: 2*v must not be a square (diagonal constraint)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]

    # Map vertex value to index in valid_vertices
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # 3. Build adjacency bitsets for the conflict graph
    # Edge (i, j) exists if valid_vertices[i] + valid_vertices[j] is a square
    adj_bits = [0] * n_vars
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    # 4. Dominance Reduction
    # A vertex u is dominated by v (v != u) if closed_neigh[v] is a subset of closed_neigh[u].
    # If we include v, we cannot include any neighbor of v. Since N[v] subset N[u],
    # any set containing u and no neighbors of u can be swapped to contain v and no neighbors of v
    # without reducing size (actually, it's a standard domination rule for independent sets:
    # if N[v] subset N[u], u is never better than v).
    # Specifically, if we have an independent set S containing u, we can replace u with v
    # to get S' = (S \ {u}) U {v}. Since N[v] subset N[u], and S is independent (no neighbors of u in S),
    # S' is also independent. Thus, there exists an optimal solution not containing u.
    # We can safely remove u from consideration.

    closed_neigh = [adj_bits[i] | (1 << i) for i in range(n_vars)]

    # Sort by popcount of closed neighborhood (ascending)
    # Smaller neighborhoods are more likely to be subsets of larger ones.
    order = sorted(range(n_vars), key=lambda i: bin(closed_neigh[i]).count('1'))

    keep = [True] * n_vars
    # We iterate through vertices. If a vertex is dominated by a previously *kept* vertex, we discard it.
    # Note: We must be careful. If v dominates u, we discard u.
    # If we process in increasing neighborhood size, and v has smaller or equal neighborhood,
    # we check if N[v] subset N[u].
    
    kept_indices = []
    for u_idx in order:
        is_dominated = False
        for v_idx in kept_indices:
            # Check if N[v] is subset of N[u]
            if (closed_neigh[v_idx] & closed_neigh[u_idx]) == closed_neigh[v_idx]:
                is_dominated = True
                break
        if not is_dominated:
            kept_indices.append(u_idx)

    n_reduced = len(kept_indices)
    if n_reduced == 0:
        return [valid_vertices[0]]

    # Map reduced index to original valid_vertices index
    reduced_to_orig = [0] * n_reduced
    for i, idx in enumerate(kept_indices):
        reduced_to_orig[i] = idx

    # Build constraints for the reduced graph
    # We need to find edges between kept vertices.
    constraint_pairs = []
    for i in range(n_reduced):
        orig_i = kept_indices[i]
        for j in range(i + 1, n_reduced):
            orig_j = kept_indices[j]
            if valid_vertices[orig_i] + valid_vertices[orig_j] in squares:
                constraint_pairs.append((i, j))

    m_constraints = len(constraint_pairs)

    best_solution = []

    # Helper to convert reduced solution to original values
    def get_final_set(reduced_sol):
        return [valid_vertices[reduced_to_orig[i]] for i in reduced_sol]

    # Helper to verify and return size
    def verify_and_size(sol_values):
        # Check diagonal
        for v in sol_values:
            if 2 * v in squares:
                return 0
        # Check pairs
        n = len(sol_values)
        for i in range(n):
            for j in range(i + 1, n):
                if sol_values[i] + sol_values[j] in squares:
                    return 0
        return n

    # Fallback: Greedy on reduced graph
    def greedy_solve():
        # Build reduced adjacency
        red_adj = [0] * n_reduced
        for i, j in constraint_pairs:
            red_adj[i] |= (1 << j)
            red_adj[j] |= (1 << i)
        
        avail = (1 << n_reduced) - 1
        sol = []
        while avail:
            # Pick vertex with min degree in remaining graph
            best_v = -1
            best_deg = float('inf')
            for v in range(n_reduced):
                if avail & (1 << v):
                    deg = bin(red_adj[v] & avail).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            sol.append(best_v)
            avail &= ~(red_adj[best_v] | (1 << best_v))
        return sol

    if m_constraints == 0:
        best_solution = list(range(n_reduced))
    else:
        A_ub = np.zeros((m_constraints, n_reduced), dtype=np.float64)
        b_ub = np.ones(m_constraints, dtype=np.float64)
        
        for k, (i, j) in enumerate(constraint_pairs):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            
        c = -np.ones(n_reduced, dtype=np.float64)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_reduced, dtype=np.int8)
        
        # Try MIP
        try:
            res = milp(
                c=c,
                constraints=[LinearConstraint(A_ub, ub=b_ub)],
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 35}
            )
            
            if res.status == 0 or res.status == 1:
                mip_sol = []
                for i in range(n_reduced):
                    if res.x[i] > 0.5:
                        mip_sol.append(i)
                if len(mip_sol) > 0:
                    best_solution = mip_sol
            else:
                best_solution = greedy_solve()
        except Exception:
            best_solution = greedy_solve()

    # Polish: Local Search (1-swap)
    # Try to swap one vertex out and two in, or just add one if possible?
    # Since it's a Max Independent Set, we can try:
    # 1. Try to add any remaining vertex if it doesn't conflict.
    # 2. Try to swap: remove u, add v, w if they don't conflict with each other or the rest.
    
    current_set = set(best_solution)
    current_mask = 0
    for i in current_set:
        current_mask |= (1 << i)
        
    # Build reduced adjacency for fast checks
    red_adj = [0] * n_reduced
    for i, j in constraint_pairs:
        red_adj[i] |= (1 << j)
        red_adj[j] |= (1 << i)

    improved = True
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        
        # 1. Try to add a single vertex
        for v in range(n_reduced):
            if v not in current_set:
                if (red_adj[v] & current_mask) == 0:
                    current_set.add(v)
                    current_mask |= (1 << v)
                    improved = True
                    break # Restart loop to check for more additions
                    
        if improved:
            continue

        # 2. Try 1-out 2-in swap
        # For each u in current_set, for each pair (v, w) not in current_set:
        # If v, w don't conflict with each other, and don't conflict with (current_set - {u}),
        # then swap.
        
        # Optimization: Precompute for each u, the set of vertices that conflict with u?
        # Or just iterate. n_reduced is likely small enough after dominance reduction.
        # Let's check size.
        if n_reduced > 200:
            # If too large, skip expensive swap search to avoid timeout
            break
            
        for u in list(current_set):
            # Temporarily remove u
            new_mask = current_mask & ~(1 << u)
            
            # Find candidates v, w that do not conflict with new_mask
            candidates = []
            for v in range(n_reduced):
                if v not in current_set and v != u:
                    if (red_adj[v] & new_mask) == 0:
                        candidates.append(v)
                        
            # Try all pairs in candidates
            for i in range(len(candidates)):
                v = candidates[i]
                for j in range(i + 1, len(candidates)):
                    w = candidates[j]
                    # Check if v and w conflict
                    if (red_adj[v] & (1 << w)) == 0:
                        # Valid swap
                        current_set.remove(u)
                        current_set.add(v)
                        current_set.add(w)
                        current_mask = 0
                        for x in current_set:
                            current_mask |= (1 << x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Convert to final values
    final_set = get_final_set(sorted(current_set))
    
    # Final Verification
    size = verify_and_size(final_set)
    if size == 0:
        # Fallback to a known valid single element
        return [valid_vertices[0]]
        
    return final_set
