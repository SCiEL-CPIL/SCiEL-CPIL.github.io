"""QAP tai256c solver: iterated local search with Hungarian re-optimization and spectral bisection init."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def spectral_bisection_init(a, b):
        """
        Spectral bisection: recursively partition facilities by flow graph connectivity
        and locations by distance structure, then assign hierarchically.
        
        Build graph G_f where edge weight = flow a[i,j], find Fiedler vector to bisect.
        Build graph G_d where edge weight = -distance b[i,j] (or use distance directly),
        find Fiedler vector to bisect locations.
        Recursively partition and assign.
        """
        def fiedler_partition(matrix, size):
            """Find Fiedler vector of a graph Laplacian to bisect nodes."""
            # Build Laplacian from matrix (treat as adjacency)
            # For flow matrix: higher flow = stronger connection = should be together
            # For distance matrix: lower distance = closer = should be together
            D = np.diag(matrix.sum(axis=1))
            L = D - matrix
            # Find smallest non-zero eigenvalue (Fiedler vector)
            try:
                # Use eigsh for large sparse-like problems, but matrix is dense
                # For n=256, dense eigendecomposition is fine
                vals, vecs = np.linalg.eigh(L)
                # Sort by eigenvalue
                idx = np.argsort(vals)
                # Fiedler vector is the second smallest (index 1)
                # Skip the trivial constant vector
                fiedler = vecs[:, idx[1]]
            except Exception:
                fiedler = np.random.randn(size)
            
            # Partition: nodes with positive fiedler value go to one side
            positive = fiedler > 0
            return positive

        def recursive_assign(facilities, locations, depth=0):
            """Recursively bisect and assign."""
            if len(facilities) <= 1:
                if len(facilities) == 1 and len(locations) == 1:
                    return {facilities[0]: locations[0]}
                return {}
            
            if len(facilities) <= 2:
                # Direct assignment
                result = {}
                for i, f in enumerate(facilities):
                    result[f] = locations[i]
                return result
            
            # Bisect facilities by flow graph
            fac_idx = list(range(len(facilities)))
            sub_a = a[np.ix_(facilities, facilities)]
            fac_partition = fiedler_partition(sub_a, len(facilities))
            
            # Bisect locations by distance graph (use negative distance as "similarity")
            loc_idx = list(range(len(locations)))
            sub_b = b[np.ix_(locations, locations)]
            # For distance, we want to group close locations together
            # Use -b as similarity (higher = closer = should be together)
            loc_partition = fiedler_partition(-sub_b, len(locations))
            
            # Split into two groups
            fac_group1 = [facilities[i] for i in range(len(facilities)) if fac_partition[i]]
            fac_group2 = [facilities[i] for i in range(len(facilities)) if not fac_partition[i]]
            loc_group1 = [locations[i] for i in range(len(locations)) if loc_partition[i]]
            loc_group2 = [locations[i] for i in range(len(locations)) if not loc_partition[i]]
            
            # Balance the partitions
            if len(fac_group1) > len(loc_group1) and len(fac_group2) > 0:
                # Move some facilities from group1 to group2
                diff = len(fac_group1) - len(loc_group1)
                for _ in range(min(diff, len(fac_group1))):
                    fac_group2.append(fac_group1.pop())
            elif len(fac_group1) < len(loc_group1) and len(fac_group2) > 0:
                diff = len(loc_group1) - len(fac_group1)
                for _ in range(min(diff, len(fac_group2))):
                    fac_group1.append(fac_group2.pop())
            
            # Ensure groups are non-empty
            if len(fac_group1) == 0:
                fac_group1 = [facilities[0]]
                fac_group2 = facilities[1:]
            if len(fac_group2) == 0:
                fac_group2 = [facilities[-1]]
                fac_group1 = facilities[:-1]
            if len(loc_group1) == 0:
                loc_group1 = [locations[0]]
                loc_group2 = locations[1:]
            if len(loc_group2) == 0:
                loc_group2 = [locations[-1]]
                loc_group1 = locations[:-1]
            
            # Recursively assign
            result = {}
            result.update(recursive_assign(fac_group1, loc_group1, depth + 1))
            result.update(recursive_assign(fac_group2, loc_group2, depth + 1))
            
            return result

        # Get all facilities and locations
        all_facilities = list(range(n))
        all_locations = list(range(n))
        
        # Recursively assign
        assignment = recursive_assign(all_facilities, all_locations)
        
        # Build permutation
        p = np.empty(n, dtype=np.int64)
        for fac, loc in assignment.items():
            p[fac] = loc
        
        # Handle any unassigned (shouldn't happen but safety)
        assigned_locs = set(assignment.values())
        unassigned_facs = [f for f in range(n) if f not in assignment]
        unassigned_locs = [l for l in range(n) if l not in assigned_locs]
        for f, l in zip(unassigned_facs, unassigned_locs):
            p[f] = l
        
        return p

    # Initial solution 1: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_rank = p_rank.astype(np.int64)

    # Initial solution 2: spectral bisection
    p_spec = spectral_bisection_init(a, b)

    # Choose the better initial solution
    cost_rank = compute_cost(p_rank)
    cost_spec = compute_cost(p_spec)
    
    if cost_rank <= cost_spec:
        current_p = p_rank.copy()
        current_cost = cost_rank
    else:
        current_p = p_spec.copy()
        current_cost = cost_spec

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    best_p = current_p.copy()
    best_cost = current_cost

    # Also try local search from the other initial solution if it's competitive
    other_p = p_spec.copy() if cost_rank <= cost_spec else p_rank.copy()
    other_cost = cost_spec if cost_rank <= cost_spec else cost_rank
    other_p, other_cost = local_search(other_p, other_cost)
    if other_cost < best_cost:
        best_p = other_p.copy()
        best_cost = other_cost

    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = np.random.randint(2, 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 20:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
