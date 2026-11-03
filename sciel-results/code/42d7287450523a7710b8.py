"""QAP solver for tho150 using vectorized local search with MDS-based initialization and SA polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Ensure matrices are float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])

    def compute_swap_delta(i, j, p):
        """
        Compute cost delta when swapping facilities at indices i and j in permutation p.
        Uses vectorized operations for speed.
        """
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0.0
        
        # Diagonal terms
        delta = a_f[i, i] * (b_f[lj, lj] - b_f[li, li])
        delta += a_f[j, j] * (b_f[li, li] - b_f[lj, lj])
        
        # Cross terms between i and j
        delta += a_f[i, j] * (b_f[lj, li] - b_f[li, lj])
        delta += a_f[j, i] * (b_f[li, lj] - b_f[lj, li])
        
        # Terms with other facilities k
        # Get all indices except i and j
        if i < j:
            ks = np.concatenate([np.arange(i), np.arange(i+1, j), np.arange(j+1, n)])
        else:
            ks = np.concatenate([np.arange(j), np.arange(j+1, i), np.arange(i+1, n)])
        
        if len(ks) > 0:
            pk = p[ks]
            
            # Vectorized sums
            # a[i, ks] * (b[lj, pk] - b[li, pk])
            delta += np.dot(a_f[i, ks], b_f[lj, pk] - b_f[li, pk])
            # a[ks, i] * (b[pk, lj] - b[pk, li])
            delta += np.dot(a_f[ks, i], b_f[pk, lj] - b_f[pk, li])
            # a[j, ks] * (b[li, pk] - b[lj, pk])
            delta += np.dot(a_f[j, ks], b_f[li, pk] - b_f[lj, pk])
            # a[ks, j] * (b[pk, li] - b[pk, lj])
            delta += np.dot(a_f[ks, j], b_f[pk, li] - b_f[pk, lj])
            
        return delta

    def local_search_first_improvement(p, current_cost, end_time, seed=12345):
        """
        Perform First Improvement 2-swap Local Search.
        Randomizes the order of swaps to avoid bias.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        while time.time() < end_time:
            improved = False
            # Create list of all pairs (i, j) with i < j
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                delta = compute_swap_delta(i, j, curr_p)
                
                if delta < 0:
                    # Accept move
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                break
                
        return best_p, best_cost

    def simulated_annealing(p, current_cost, end_time, seed=42):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta < 1e-6:
            avg_delta = 1.0
            
        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost

    def mds_embedding(D, dim=2):
        """
        Classical Multidimensional Scaling to embed a distance matrix into dim dimensions.
        Returns coordinates of shape (n, dim).
        """
        n = D.shape[0]
        # Centering matrix
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(B)
        # Sort eigenvalues in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        # Take top dim eigenvalues/vectors
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        # Ensure non-negative eigenvalues
        vals = np.maximum(vals, 0)
        # Coordinates
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y, dim=2):
        """
        Solve Procrustes problem: find orthogonal Q minimizing ||X Q - Y||_F.
        Returns the permutation that best matches X to Y based on nearest neighbors in the embedded space?
        Actually, we want to align the facility embedding X to location embedding Y.
        The assignment is p[i] = j such that X[i] is closest to Y[j]?
        Or we can use the alignment to derive a cost matrix.
        
        Strategy:
        1. Embed A (flow) into X (facilities)
        2. Embed B (distance) into Y (locations)
        3. Align X to Y using Procrustes (find Q such that X@Q approx Y)
        4. The aligned coordinates X_aligned = X @ Q
        5. Assign facility i to location j that minimizes ||X_aligned[i] - Y[j]||^2
        This is an assignment problem. We can solve it greedily or using Hungarian.
        Since n=150, Hungarian is fast.
        """
        from scipy.optimize import linear_sum_assignment
        
        # Align X to Y
        # We want to find Q such that X @ Q approx Y.
        # SVD of Y^T X
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        
        X_aligned = X @ Q
        
        # Compute cost matrix C where C[i, j] = ||X_aligned[i] - Y[j]||^2
        # C[i, j] = ||X_aligned[i]||^2 + ||Y[j]||^2 - 2 * X_aligned[i] @ Y[j]
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # 1. MDS Initialization
    # Embed Flow Matrix A
    # A is not a distance matrix, but we can treat it as one for embedding purposes.
    # Or better, use row sums for a simpler heuristic first, then MDS.
    
    # Let's try MDS on B (distance) directly.
    # And for A, we can use a similar embedding.
    # Note: MDS assumes triangle inequality, which A might not satisfy.
    # But it often works as a heuristic.
    
    X_facilities = mds_embedding(a_f, dim=2)
    Y_locations = mds_embedding(b_f, dim=2)
    
    p_mds = procrustes_alignment(X_facilities, Y_locations)
    cost_mds = compute_cost(p_mds)
    
    # 2. Rank Matching Initialization (Backup/Comparison)
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    
    # Choose best initialization
    if cost_mds < cost_rank:
        best_p = p_mds
        best_cost = cost_mds
    else:
        best_p = p_rank
        best_cost = cost_rank

    # 3. Polish with Local Search
    end_time_lk1 = deadline - 30.0
    p_lk1, cost_lk1 = local_search_first_improvement(best_p, best_cost, end_time_lk1, seed=12345)
    if cost_lk1 < best_cost:
        best_p = p_lk1
        best_cost = cost_lk1
    
    # 4. Try Random Initialization + Local Search
    if time.time() < deadline - 20.0:
        rng_init = np.random.RandomState(123)
        p_rand = rng_init.permutation(n)
        cost_rand = compute_cost(p_rand)
        
        end_time_lk2 = deadline - 15.0
        p_lk2, cost_lk2 = local_search_first_improvement(p_rand, cost_rand, end_time_lk2, seed=456)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2
            
    # 5. Perturb Best and Local Search
    if time.time() < deadline - 10.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        # Swap 5 random pairs to perturb
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        end_time_lk3 = deadline - 5.0
        p_lk3, cost_lk3 = local_search_first_improvement(p_pert, cost_pert, end_time_lk3, seed=999)
        if cost_lk3 < best_cost:
            best_p = p_lk3
            best_cost = cost_lk3
    
    # 6. Final SA Polish if time remains
    if time.time() < deadline - 2.0:
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 0.5, seed=111)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
