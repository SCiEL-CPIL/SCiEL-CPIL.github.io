"""QAP tai256c solver: iterated local search with spectral bisection init, Hungarian re-optimization, and tabu diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s limit

    # Precompute symmetric A for faster delta calculations
    A_sym = a + a.T

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def build_D(p):
        """Build D[i,j] = b[p[i], p[j]] for fast swap deltas."""
        return b[np.ix_(p, p)]

    def swap_delta(D, i, j):
        """
        Compute cost delta for swapping positions i and j in the permutation.
        D[i,k] = b[p[i], p[k]]
        
        C = sum_{i,j} a[i,j] * D[i,j]
        
        After swapping positions i and j:
        - D'[i,k] = D[j,k] for k != i,j; D'[i,i] = D[i,i]; D'[i,j] = D[j,i]
        - D'[j,k] = D[i,k] for k != i,j; D'[j,j] = D[j,j]; D'[j,i] = D[i,j]
        - D'[k,i] = D[k,j] for k != i,j; D'[i,i] = D[i,i]; D'[j,i] = D[i,j]
        - D'[k,j] = D[k,i] for k != i,j; D'[k,i] = D[k,i]; D'[k,j] = D[k,j]
        
        delta = sum over all changed terms
        """
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # For k not in {i,j}:
        # a[i,k] changes D[i,k] -> D[j,k]: diff = D[j,k] - D[i,k]
        # a[j,k] changes D[j,k] -> D[i,k]: diff = D[i,k] - D[j,k]
        # a[k,i] changes D[k,i] -> D[k,j]: diff = D[k,j] - D[k,i]
        # a[k,j] changes D[k,j] -> D[k,i]: diff = D[k,i] - D[k,j]
        
        row_diff = D[j, mask] - D[i, mask]  # shape (n-2,)
        col_diff = D[mask, j] - D[mask, i]  # shape (n-2,)
        
        delta = np.dot(a[i, mask], row_diff)
        delta += np.dot(a[j, mask], -row_diff)
        delta += np.dot(a[mask, i], col_diff)
        delta += np.dot(a[mask, j], -col_diff)
        
        # For the i-j interaction:
        # a[i,j]: D[i,j] -> D[j,i], diff = D[j,i] - D[i,j]
        # a[j,i]: D[j,i] -> D[i,j], diff = D[i,j] - D[j,i]
        delta += a[i, j] * (D[j, i] - D[i, j])
        delta += a[j, i] * (D[i, j] - D[j, i])
        
        return int(delta)

    def fast_local_search(p, cost, max_passes=100):
        """
        Optimized local search using precomputed D matrix.
        D[i, j] = b[p[i], p[j]]
        """
        D = build_D(p)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n):
                if time.time() > deadline:
                    return p, cost, D
                best_j = -1
                best_delta = 0
                for j in range(i + 1, n):
                    delta = swap_delta(D, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j >= 0:
                    # Perform swap
                    i, j = i, best_j
                    p[i], p[j] = p[j], p[i]
                    cost += best_delta
                    # Update D
                    D[i, :], D[j, :] = D[j, :], D[i, :]
                    D[:, i], D[:, j] = D[:, j], D[:, i]
                    improved = True
        
        return p, cost, D

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def fiedler_partition(matrix, size):
        """Find Fiedler vector of a graph Laplacian to bisect nodes."""
        D = np.diag(matrix.sum(axis=1))
        L = D - matrix
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(size)
        
        return fiedler > 0

    def recursive_assign(facilities, locations, depth=0):
        """Recursively bisect and assign."""
        if len(facilities) <= 1:
            if len(facilities) == 1 and len(locations) == 1:
                return {facilities[0]: locations[0]}
            return {}
        
        if len(facilities) <= 2:
            result = {}
            for i, f in enumerate(facilities):
                result[f] = locations[i]
            return result
        
        sub_a = a[np.ix_(facilities, facilities)]
        fac_partition = fiedler_partition(sub_a, len(facilities))
        
        sub_b = b[np.ix_(locations, locations)]
        loc_partition = fiedler_partition(-sub_b, len(locations))
        
        fac_group1 = [facilities[i] for i in range(len(facilities)) if fac_partition[i]]
        fac_group2 = [facilities[i] for i in range(len(facilities)) if not fac_partition[i]]
        loc_group1 = [locations[i] for i in range(len(locations)) if loc_partition[i]]
        loc_group2 = [locations[i] for i in range(len(locations)) if not loc_partition[i]]
        
        if len(fac_group1) > len(loc_group1) and len(fac_group2) > 0:
            diff = len(fac_group1) - len(loc_group1)
            for _ in range(min(diff, len(fac_group1))):
                fac_group2.append(fac_group1.pop())
        elif len(fac_group1) < len(loc_group1) and len(fac_group2) > 0:
            diff = len(loc_group1) - len(fac_group1)
            for _ in range(min(diff, len(fac_group2))):
                fac_group1.append(fac_group2.pop())
        
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
        
        result = {}
        result.update(recursive_assign(fac_group1, loc_group1, depth + 1))
        result.update(recursive_assign(fac_group2, loc_group2, depth + 1))
        
        return result

    def spectral_bisection_init():
        """Generate initial permutation using spectral bisection approach."""
        all_facilities = list(range(n))
        all_locations = list(range(n))
        
        assignment = recursive_assign(all_facilities, all_locations)
        
        p = np.empty(n, dtype=np.int64)
        for fac, loc in assignment.items():
            p[fac] = loc
        
        assigned_locs = set(assignment.values())
        unassigned_facs = [f for f in range(n) if f not in assignment]
        unassigned_locs = [l for l in range(n) if l not in assigned_locs]
        for f, l in zip(unassigned_facs, unassigned_locs):
            p[f] = l
        
        return p

    def rank_matching_init():
        """Fallback rank-matching initialization."""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def hungarian_init():
        """Hungarian-based initialization."""
        # Linearize: for each facility i, compute cost of assigning to each location j
        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        # But we don't have p yet. Use a greedy construction.
        
        # Start with random p, then use Hungarian to improve
        p = np.random.permutation(n).astype(np.int64)
        p, cost, _ = fast_local_search(p, compute_cost(p), max_passes=5)
        
        # Now use Hungarian re-optimization
        h_p = hungarian_reopt(p)
        h_cost = compute_cost(h_p)
        h_p, h_cost, _ = fast_local_search(h_p, h_cost, max_passes=10)
        
        return h_p, h_cost

    def tabu_kick(p, cost, n_kicks=3):
        """Apply block-reversal kicks with tabu for stronger diversification."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 6
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            
            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            key = (i, j)
            
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            p[i:j+1] = p[i:j+1][::-1]
            new_cost = compute_cost(p)
            
            if not is_tabu or (new_cost < best_cost):
                cost = new_cost
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                p[i:j+1] = p[i:j+1][::-1]
            
            iteration += 1
        
        return best_p, best_cost

    # Phase 1: Generate multiple diverse initial solutions
    inits = []
    
    # Spectral bisection
    p_spec = spectral_bisection_init()
    inits.append(p_spec)
    
    # Rank matching
    p_rank = rank_matching_init()
    inits.append(p_rank)
    
    # Hungarian-based
    if time.time() < deadline - 30:
        p_hung, _ = hungarian_init()
        inits.append(p_hung)
    
    # Random starts
    for _ in range(3):
        if time.time() > deadline - 25:
            break
        inits.append(np.random.permutation(n).astype(np.int64))

    # Phase 2: Local search each initial solution
    best_p = None
    best_cost = float('inf')
    
    for p_init in inits:
        if time.time() > deadline - 20:
            break
        p_ls, cost_ls, _ = fast_local_search(p_init.copy(), compute_cost(p_init), max_passes=30)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # Phase 3: Iterated Local Search
    iteration = 0
    while time.time() < deadline - 15:
        iteration += 1

        p = best_p.copy()
        
        # Diversify with block reversal
        block_len = np.random.randint(4, 12)
        start = np.random.randint(0, n - block_len)
        p[start:start+block_len] = p[start:start+block_len][::-1]
        
        # Tabu kick
        p, cost = tabu_kick(p, compute_cost(p), n_kicks=2)

        # Local search
        p, cost, _ = fast_local_search(p, cost, max_passes=40)

        # Periodic Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost, _ = fast_local_search(h_p, h_cost, max_passes=20)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        final_p = best_p.copy()
        final_cost = best_cost
        
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            final_p, final_cost, _ = fast_local_search(final_p, final_cost, max_passes=50)
            if final_cost < best_cost:
                best_p = final_p.copy()
                best_cost = final_cost

    return best_p.tolist()
