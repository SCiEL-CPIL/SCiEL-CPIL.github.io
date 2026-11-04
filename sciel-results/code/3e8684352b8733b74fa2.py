"""QAP solver for tho150 using Spectral Relaxation and Sinkhorn-Knopp refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    # Convert to float64 for fast arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])
    
    def spectral_init(a_mat, b_mat, dim=10):
        """
        Spectral initialization:
        1. Compute top eigenvectors of A (flow) and B (distance).
        2. Align them via Procrustes (rotation/reflection).
        3. Solve assignment problem on the aligned coordinates.
        
        Note: For QAP, we want to minimize sum a[i,j] b[p[i], p[j]].
        This is equivalent to aligning the "structure" of A with the "inverse structure" of B?
        Actually, standard spectral heuristic for QAP:
        Let A = U_A Lambda_A U_A^T, B = U_B Lambda_B U_B^T.
        We want P such that P^T B P approx A (in some sense).
        A common heuristic is to match the eigenvectors.
        However, since we want to minimize, and B is distance (large values = bad),
        we might want to align large flow with small distance.
        
        Let's try:
        X = U_A * sqrt(Lambda_A)  (Facility coordinates)
        Y = U_B * sqrt(Lambda_B)  (Location coordinates)
        
        We want to assign facility i to location p[i] such that the "geometry" matches.
        If A and B have similar structure (e.g., both block-like), matching eigenvectors works.
        If we want to minimize, and assuming positive definite-ish matrices,
        we align X and Y.
        
        Let's use the top `dim` eigenvectors.
        """
        # Eigen decomposition
        # np.linalg.eigh returns eigenvalues in ascending order
        eigvals_a, eigvecs_a = np.linalg.eigh(a_mat)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_mat)
        
        # Take top dim eigenvectors
        idx_a = np.argsort(eigvals_a)[::-1][:dim]
        idx_b = np.argsort(eigvals_b)[::-1][:dim]
        
        X = eigvecs_a[:, idx_a] * np.sqrt(np.maximum(eigvals_a[idx_a], 0))
        Y = eigvecs_b[:, idx_b] * np.sqrt(np.maximum(eigvals_b[idx_b], 0))
        
        # Procrustes alignment: find R such that X @ R approx Y
        # Minimize ||X R - Y||_F^2
        # Solution: R = (X^T X)^{-1} X^T Y ... wait, X and Y are n x dim.
        # We want R (dim x dim) such that X R approx Y.
        # This is least squares: (X^T X) R = X^T Y.
        # Since X^T X is dim x dim, we can solve it.
        
        XtX = X.T @ X
        XtY = X.T @ Y
        try:
            R = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            R = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
            
        # Align X to Y
        X_aligned = X @ R
        
        # Solve assignment: minimize sum ||x_i - y_{p[i]}||^2
        # Equivalent to maximizing sum x_i . y_{p[i]}
        D = X_aligned @ Y.T
        row_ind, col_ind = linear_sum_assignment(-D)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    def sinkhorn_refinement(p_init, end_time):
        """
        Refine the initial permutation using a continuous relaxation approach.
        We represent the assignment as a doubly stochastic matrix P.
        Objective: minimize tr(P^T A P B) ? No, the QAP objective is sum a[i,j] b[p[i], p[j]].
        In matrix form: sum_{i,j,k,l} A[i,j] P[i,k] B[k,l] P[j,l] = tr(P^T A P B).
        
        We can use gradient descent on the Birkhoff polytope.
        Gradient wrt P: G = A P B + A^T P^T B^T (assuming symmetric A, B).
        For non-symmetric: G = A P B + (A^T P^T B^T)^T = A P B + B P A^T.
        
        We update P via:
        1. Compute gradient G.
        2. Update P_soft = P * exp(-lr * G) (multiplicative weights) or P - lr * G.
        3. Project P_soft back to Birkhoff polytope using Sinkhorn-Knopp.
        
        We start from the hard assignment P_init (one-hot matrix).
        """
        P = np.zeros((n, n))
        P[np.arange(n), p_init] = 1.0
        
        # Precompute for gradient
        # G = A @ P @ B + B @ P @ A.T
        # Since A and B are symmetric in QAPLIB usually, but let's be general.
        # Actually, QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
        # Let's check symmetry of A and B.
        # If A and B are symmetric, G = A P B + B P A.
        
        lr = 0.05
        sinkhorn_iters = 20
        
        curr_cost = compute_cost(p_init)
        best_p = p_init.copy()
        best_cost = curr_cost
        
        # We need to track the discrete permutation corresponding to the current P.
        # We can extract the best discrete assignment from P using linear_sum_assignment on -P?
        # Or just track the row-max? Linear sum assignment on the "cost" matrix derived from P is better.
        # Actually, P is a probability matrix. The best discrete assignment is argmin sum C[i, p[i]] where C is negative log P?
        # Simpler: Use linear_sum_assignment on -P to get the most likely permutation.
        
        t_start = time.time()
        step = 0
        
        while time.time() < end_time:
            # Compute Gradient
            AP = a_f @ P
            APB = AP @ b_f
            BP = b_f @ P
            BPA = BP @ a_f.T
            
            G = APB + BPA
            
            # Multiplicative Update (similar to EM)
            # P_new = P * exp(-lr * G)
            # To avoid underflow/overflow, we can normalize.
            # Alternatively, use additive update and project.
            # Let's use multiplicative update as it keeps P positive.
            
            # Clip G to prevent extreme exponentials
            G_clipped = np.clip(G, -10.0, 10.0)
            
            P = P * np.exp(-lr * G_clipped)
            
            # Sinkhorn-Knopp projection
            # Normalize rows and columns to sum to 1
            for _ in range(sinkhorn_iters):
                # Row normalization
                row_sums = P.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                P = P / row_sums
                # Column normalization
                col_sums = P.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                P = P / col_sums
                
            # Extract discrete permutation
            # The most likely assignment is the one with highest P values.
            # We can solve the assignment problem minimizing -log(P) or simply maximizing P.
            # linear_sum_assignment minimizes cost. So we use -P as cost.
            row_ind, col_ind = linear_sum_assignment(-P)
            p_curr = np.empty(n, dtype=np.int64)
            p_curr[row_ind] = col_ind
            
            # Check if permutation changed
            if not np.array_equal(p_curr, best_p):
                cost_curr = compute_cost(p_curr)
                if cost_curr < best_cost:
                    best_cost = cost_curr
                    best_p = p_curr.copy()
                    # Reset P to the new best hard assignment to avoid getting stuck in a bad basin?
                    # Or keep P soft? Keeping P soft allows exploration.
                    # Let's keep P soft but maybe sharpen it?
                    # If we reset P, we lose the "momentum" of the relaxation.
                    # Let's try to sharpen P towards the new best solution.
                    # P = 0.9 * P + 0.1 * (one-hot of best_p)
                    P_new = np.zeros((n, n))
                    P_new[np.arange(n), best_p] = 1.0
                    P = 0.8 * P + 0.2 * P_new
                    # Re-project
                    for _ in range(sinkhorn_iters):
                        row_sums = P.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1.0
                        P = P / row_sums
                        col_sums = P.sum(axis=0, keepdims=True)
                        col_sums[col_sums == 0] = 1.0
                        P = P / col_sums
            
            step += 1
            # Adaptive learning rate?
            if step % 100 == 0:
                lr *= 0.95
                
        return best_p, best_cost

    def fast_local_search(p, current_cost, end_time, seed=12345):
        """
        Fast 2-swap local search.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        while time.time() < end_time:
            improved = False
            for _ in range(1000):
                if time.time() > end_time:
                    break
                
                i, j = rng.choice(n, 2, replace=False)
                
                li = curr_p[i]
                lj = curr_p[j]
                
                if li == lj:
                    continue
                
                # Compute delta
                # delta = sum_{k != i, j} a[i, k] (b[lj, p[k]] - b[li, p[k]]) + ...
                # Using vectorized sums
                pk = curr_p
                
                b_lj_pk = b_f[lj, pk]
                b_li_pk = b_f[li, pk]
                b_pk_lj = b_f[pk, lj]
                b_pk_li = b_f[pk, li]
                
                S1 = np.dot(a_f[i, :], b_lj_pk)
                S2 = np.dot(a_f[i, :], b_li_pk)
                S3 = np.dot(a_f[:, i], b_pk_lj)
                S4 = np.dot(a_f[:, i], b_pk_li)
                S5 = np.dot(a_f[j, :], b_li_pk)
                S6 = np.dot(a_f[j, :], b_lj_pk)
                S7 = np.dot(a_f[:, j], b_pk_li)
                S8 = np.dot(a_f[:, j], b_pk_lj)
                
                # Subtract k=i, j terms
                S1 -= a_f[i, i] * b_f[lj, li] + a_f[i, j] * b_f[lj, lj]
                S2 -= a_f[i, i] * b_f[li, li] + a_f[i, j] * b_f[li, lj]
                S3 -= a_f[i, i] * b_f[li, lj] + a_f[j, i] * b_f[lj, lj]
                S4 -= a_f[i, i] * b_f[li, li] + a_f[j, i] * b_f[lj, li]
                S5 -= a_f[j, i] * b_f[li, li] + a_f[j, j] * b_f[li, lj]
                S6 -= a_f[j, i] * b_f[lj, li] + a_f[j, j] * b_f[lj, lj]
                S7 -= a_f[i, j] * b_f[li, li] + a_f[j, j] * b_f[lj, li]
                S8 -= a_f[i, j] * b_f[li, lj] + a_f[j, j] * b_f[lj, lj]
                
                delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
                
                # Extra terms for k=i, j
                extra = 0.0
                extra += a_f[i, i] * (b_f[lj, li] - b_f[li, li])
                extra += a_f[i, j] * (b_f[lj, lj] - b_f[li, lj])
                extra += a_f[i, i] * (b_f[li, lj] - b_f[li, li])
                extra += a_f[j, i] * (b_f[lj, lj] - b_f[lj, li])
                extra += a_f[j, i] * (b_f[li, li] - b_f[lj, li])
                extra += a_f[j, j] * (b_f[li, lj] - b_f[lj, lj])
                extra += a_f[i, j] * (b_f[li, li] - b_f[li, lj])
                extra += a_f[j, j] * (b_f[lj, li] - b_f[lj, lj])
                
                delta += extra
                
                if delta < 0:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    break
            
            if not improved:
                break
                
        return best_p, best_cost

    # 1. Spectral Initialization
    p_init = spectral_init(a_f, b_f, dim=10)
    cost_init = compute_cost(p_init)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. Sinkhorn Refinement
    # Allocate 50% of time to Sinkhorn
    end_sinkhorn = start_time + (deadline - start_time) * 0.5
    p_sink, cost_sink = sinkhorn_refinement(best_p, end_sinkhorn)
    if cost_sink < best_cost:
        best_p = p_sink
        best_cost = cost_sink
        
    # 3. Discrete Local Search
    # Allocate remaining time to local search
    end_ls = deadline - 2.0
    p_ls, cost_ls = fast_local_search(best_p, best_cost, end_ls, seed=12345)
    if cost_ls < best_cost:
        best_p = p_ls
        best_cost = cost_ls

    return best_p.tolist()
# EVOLVE-BLOCK-END
