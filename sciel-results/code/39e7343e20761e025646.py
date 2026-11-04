"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p, b):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Efficient delta calculation using precomputed difference vectors
        diff_b_row = b[lj, :] - b[li, :]
        diff_b_col = b[:, lj] - b[:, li]
        
        # Vectorized delta calculation
        term1 = a[i, :] @ diff_b_row[p]
        term2 = a[:, i] @ diff_b_col[p]
        term3 = -a[j, :] @ diff_b_row[p]
        term4 = -a[:, j] @ diff_b_col[p]
        
        delta = term1 + term2 + term3 + term4
        
        # Exclude i and j from the sum (they were included in the dot products)
        # We need to subtract the contributions involving i and j
        # a[i,i]*diff_b_row[p[i]] + a[i,j]*diff_b_row[p[j]]
        # a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]]
        # a[i,i]*diff_b_col[p[i]] + a[i,j]*diff_b_col[p[j]]
        # a[j,i]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]]
        
        # Actually, the formula above is correct for the sum over all k != i, j
        # but the dot products include k=i and k=j. Let me re-derive.
        
        # Delta = sum_{k != i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[j,k]*(b[li,p[k]] - b[lj,p[k]])
        #       + sum_{k != i,j} a[k,i]*(b[p[k],lj] - b[p[k],li]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
        #       + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
        #       + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
        
        # The first sum can be written as a[i,:]*diff_b_row[p] - a[i,i]*diff_b_row[p[i]] - a[i,j]*diff_b_row[p[j]]
        # The second sum can be written as a[j,:]*(-diff_b_row[p]) + a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]]
        # The third sum can be written as a[:,i]*diff_b_col[p] - a[i,i]*diff_b_col[p[i]] - a[j,i]*diff_b_col[p[j]]
        # The fourth sum can be written as a[:,j]*(-diff_b_col[p]) + a[i,j]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]]
        
        # So delta = (a[i,:] @ diff_b_row[p] - a[i,i]*diff_b_row[p[i]] - a[i,j]*diff_b_row[p[j]])
        #         + (-a[j,:] @ diff_b_row[p] + a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]])
        #         + (a[:,i] @ diff_b_col[p] - a[i,i]*diff_b_col[p[i]] - a[j,i]*diff_b_col[p[j]])
        #         + (-a[:,j] @ diff_b_col[p] + a[i,j]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]])
        #         + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
        #         + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
        
        # This is getting complicated. Let me use the simpler approach from the original code
        # which was correct, just slower.
        
        # Actually, let me just use the original correct formula
        delta_k = (a[i, :] - a[j, :]) * diff_b_row[p] + (a[:, i] - a[:, j]) * diff_b_col[p]
        delta_k[i] = 0
        delta_k[j] = 0
        delta = np.sum(delta_k)
        
        # Diagonal and i-j terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        return delta

    def local_search(p, best_cost, b):
        """Perform 2-opt local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    delta = compute_swap_delta(i, j, p, b)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        improved = True
        return p, best_cost

    def mds_initialize(a, b):
        """MDS-based initialization."""
        def double_center(M):
            n = len(M)
            J = np.eye(n) - np.ones((n, n)) / n
            return J @ M @ J

        C_A = double_center(a.astype(float))
        C_B = double_center(b.astype(float))
        
        k = 10
        
        eigvals_A, eigvecs_A = np.linalg.eigh(C_A)
        eigvals_B, eigvecs_B = np.linalg.eigh(C_B)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        
        eigvals_B = eigvals_B[idx_B]
        eigvecs_B = eigvecs_B[:, idx_B]
        
        X_A = eigvecs_A[:, :k] * np.sqrt(np.maximum(eigvals_A[:k], 0))
        X_B = eigvecs_B[:, :k] * np.sqrt(np.maximum(eigvals_B[:k], 0))
        
        M = X_A.T @ X_B
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        X_A_rot = X_A @ R
        
        norm_A = np.sum(X_A_rot**2, axis=1)
        norm_B = np.sum(X_B**2, axis=1)
        dists = norm_A[:, None] + norm_B[None, :] - 2 * (X_A_rot @ X_B.T)
        
        row_ind, col_ind = linear_sum_assignment(dists)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    # 1. MDS Initialization
    p_mds = mds_initialize(a, b)
    best_p = p_mds.copy()
    best_cost = compute_cost_fast(p_mds)
    
    # 2. Local Search
    best_p, best_cost = local_search(best_p, best_cost, b)
    
    # 3. Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        p_rand = rng.permutation(n)
        p_rand, cost_rand = local_search(p_rand, compute_cost_fast(p_rand), b)
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
