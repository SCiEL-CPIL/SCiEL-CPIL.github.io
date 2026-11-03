import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (5s buffer)

    # Convert to int64 arrays for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    best_p = None
    best_cost = np.iinfo(np.int64).max

    def compute_cost(p):
        """Compute the QAP cost for permutation p using int64 arithmetic."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return int(np.sum(a * b[np.ix_(p, p)]))

    def delta_cost_swap(p, i, j):
        """
        Calculate the change in cost when swapping locations of facilities i and j.
        p[i] = u, p[j] = v. After swap, p'[i] = v, p'[j] = u.
        
        Uses vectorized dot products for O(n) computation.
        """
        u = p[i]
        v = p[j]
        
        if u == v:
            return 0
        
        before_i = np.dot(a[i, :], b[u, p])
        before_j = np.dot(a[j, :], b[v, p])
        before_ci = np.dot(a[:, i], b[p, u])
        before_cj = np.dot(a[:, j], b[p, v])
        before_dd = a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
        
        after_i = np.dot(a[i, :], b[v, p])
        after_j = np.dot(a[j, :], b[u, p])
        after_ci = np.dot(a[:, i], b[p, v])
        after_cj = np.dot(a[:, j], b[p, u])
        after_dd = a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
        
        delta = (after_i + after_j + after_ci + after_cj - after_dd) - \
                (before_i + before_j + before_ci + before_cj - before_dd)
        
        return int(delta)

    def local_search_2opt(p, max_time=None):
        """Perform 2-opt local search with time limit."""
        p = p.copy()
        current_cost = compute_cost(p)
        improved = True
        passes = 0
        
        while improved:
            improved = False
            passes += 1
            for i in range(n):
                if max_time and time.time() > max_time:
                    return p, current_cost
                for j in range(i+1, n):
                    delta = delta_cost_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    def spectral_bisection_init(a, b):
        """
        Spectral bisection initialization using graph partitioning.
        Construct weighted graph from flow matrix, use Fiedler vector for bisection,
        then recursively partition and map to locations sorted by distance.
        """
        # Build adjacency from flow matrix (symmetrize)
        A = (a + a.T) / 2.0
        A = A.astype(np.float64)
        
        # Create Laplacian L = D - A
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        
        # Find Fiedler vector (2nd smallest eigenvalue)
        try:
            # Use sparse eigensolver for efficiency
            L_sparse = csr_matrix(L)
            eigenvalues, eigenvectors = eigsh(L_sparse, k=2, sigma=0, which='LM')
            # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
            fiedler_idx = np.argmin(np.abs(eigenvalues - eigenvalues[1]))
            fiedler = eigenvectors[:, fiedler_idx]
        except Exception:
            # Fallback: use degree as proxy
            fiedler = degrees.copy()
        
        # Partition based on Fiedler vector sign
        median_val = np.median(fiedler)
        left_nodes = np.where(fiedler <= median_val)[0]
        right_nodes = np.where(fiedler > median_val)[0]
        
        # Ensure balanced partition
        if len(left_nodes) < n // 4:
            # Rebalance by moving some nodes
            sorted_by_fiedler = np.argsort(fiedler)
            left_nodes = sorted_by_fiedler[:n//2]
            right_nodes = sorted_by_fiedler[n//2:]
        elif len(right_nodes) < n // 4:
            sorted_by_fiedler = np.argsort(fiedler)
            left_nodes = sorted_by_fiedler[:n//2]
            right_nodes = sorted_by_fiedler[n//2:]
        
        # Now assign locations: sort locations by their "distance centrality"
        # Locations with lower total distance should get high-flow facilities
        loc_centralities = b.sum(axis=1).astype(np.float64)
        loc_order = np.argsort(loc_centralities)  # Most central first
        
        # Assign left partition (high flow cluster) to central locations
        # Assign right partition to peripheral locations
        p = np.empty(n, dtype=np.int64)
        
        # Within each partition, rank by flow centrality
        left_flow = a.sum(axis=1)[left_nodes]
        left_rank = np.argsort(-left_flow)  # Highest flow first
        right_flow = a.sum(axis=1)[right_nodes]
        right_rank = np.argsort(-right_flow)
        
        # Map to locations
        left_locs = loc_order[:len(left_nodes)]
        right_locs = loc_order[len(left_nodes):]
        
        for idx, node in enumerate(left_nodes[left_rank]):
            p[node] = left_locs[idx]
        for idx, node in enumerate(right_nodes[right_rank]):
            p[node] = right_locs[idx]
        
        return p

    def linex_init(a, b):
        """
        Linex initialization: Use linear assignment to create initial permutation.
        """
        fac_weights = a.sum(axis=1).astype(np.float64)
        loc_weights = b.sum(axis=1).astype(np.float64)
        C = np.outer(fac_weights, loc_weights)
        _, p = linear_sum_assignment(C)
        return p.astype(np.int64)

    def random_restart_init(n, rng):
        """Generate a random permutation."""
        return rng.permutation(n).astype(np.int64)

    # Generate initial solutions
    # 1. Spectral bisection initialization (primary structural change)
    p_spectral = spectral_bisection_init(a, b)
    p_ls, cost_ls = local_search_2opt(p_spectral, max_time=deadline - 10)
    best_p = p_ls.copy()
    best_cost = cost_ls

    # 2. Linex initialization
    p_linex = linex_init(a, b)
    p_ls, cost_ls = local_search_2opt(p_linex, max_time=deadline - 10)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # 3. Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_ls, cost_ls = local_search_2opt(p_rank, max_time=deadline - 10)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # 4. Try multiple random restarts with remaining time
    random.seed(42)
    np_random = np.random.RandomState(42)
    
    num_restarts = 0
    while time.time() < deadline - 5:
        p_rand = random_restart_init(n, np_random)
        p_ls, cost_ls = local_search_2opt(p_rand, max_time=deadline - 2)
        num_restarts += 1
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # Final check
    if best_p is None:
        best_p = p_rank
    
    return best_p.tolist()
