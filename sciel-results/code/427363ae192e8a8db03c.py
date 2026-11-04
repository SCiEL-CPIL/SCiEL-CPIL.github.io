"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. MDS-based Initialization
    def mds_initialize(a, b):
        """MDS-based initialization using eigenvectors of centered matrices."""
        def double_center(M):
            n = len(M)
            J = np.eye(n) - np.ones((n, n)) / n
            return J @ M @ J

        C_A = double_center(a.astype(float))
        C_B = double_center(b.astype(float))
        
        k = 12  # Number of dimensions for embedding
        
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
        
        # Use Hungarian algorithm for optimal assignment based on distances
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dists)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    p = mds_initialize(a, b)
    
    # Helper to compute cost
    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    best_cost = compute_cost(p)
    best_p = p.copy()

    # 2. Fast 2-opt Local Search
    def local_search(p, current_cost):
        """Perform 2-opt local search until no improvements found or time limit."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, current_cost
                    
                    li = p[i]
                    lj = p[j]
                    
                    # Vectorized delta calculation
                    # delta = sum_{k != i,j} (a[i,k] + a[j,k]) * (b[lj, p[k]] - b[li, p[k]]) 
                    #        + terms for i,j interaction
                    # More efficient formulation:
                    # Let d_b_k = b[lj, p[k]] - b[li, p[k]]
                    # Let d_a_k = a[i, k] + a[j, k]
                    # Sum over k != i,j of d_a_k * d_b_k
                    # Plus corrections for k=i and k=j
                    
                    diff_b = b[lj, p] - b[li, p]
                    sum_a = a[i, :] + a[j, :]
                    
                    # This sum includes k=i and k=j terms which need adjustment
                    total_delta = np.dot(sum_a, diff_b)
                    
                    # Remove k=i term: (a[i,i]+a[j,i]) * (b[lj, li] - b[li, li])
                    total_delta -= (a[i, i] + a[j, i]) * (b[lj, li] - b[li, li])
                    # Remove k=j term: (a[i,j]+a[j,j]) * (b[lj, lj] - b[li, lj])
                    total_delta -= (a[i, j] + a[j, j]) * (b[lj, lj] - b[li, lj])
                    
                    # Add correct terms for the swapped positions i and j
                    # New cost terms involving i and j:
                    # a[i,i]*b[lj,lj] + a[j,j]*b[li,li] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
                    # Old cost terms involving i and j:
                    # a[i,i]*b[li,li] + a[j,j]*b[lj,lj] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
                    
                    new_ii = a[i, i] * b[lj, lj]
                    new_jj = a[j, j] * b[li, li]
                    new_ij = a[i, j] * b[lj, li]
                    new_ji = a[j, i] * b[li, lj]
                    
                    old_ii = a[i, i] * b[li, li]
                    old_jj = a[j, j] * b[lj, lj]
                    old_ij = a[i, j] * b[li, lj]
                    old_ji = a[j, i] * b[lj, li]
                    
                    total_delta += (new_ii + new_jj + new_ij + new_ji) - (old_ii + old_jj + old_ij + old_ji)
                    
                    if total_delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += total_delta
                        improved = True
        
        return p, current_cost

    p, best_cost = local_search(p, best_cost)
    best_p = p.copy()

    # 3. Simulated Annealing Phase
    # Use fast delta calculation to allow many iterations
    T = 50000.0
    T_min = 1e-5
    alpha = 0.9999
    
    # Pre-calculate some constants if possible, but delta depends on p
    # We will use the same delta logic but inside the SA loop
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    # To speed up SA, we can pre-compute row sums or use the vectorized delta directly
    # The vectorized delta is O(n). With n=150, this is very fast.
    
    while time.time() < deadline - 5.0: # Leave 5s for final polish
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        li = current_p[i]
        lj = current_p[j]
        
        # Calculate delta
        diff_b = b[lj, current_p] - b[li, current_p]
        sum_a = a[i, :] + a[j, :]
        
        total_delta = np.dot(sum_a, diff_b)
        
        # Adjust for i and j terms (same as in local search)
        total_delta -= (a[i, i] + a[j, i]) * (b[lj, li] - b[li, li])
        total_delta -= (a[i, j] + a[j, j]) * (b[lj, lj] - b[li, lj])
        
        new_ii = a[i, i] * b[lj, lj]
        new_jj = a[j, j] * b[li, li]
        new_ij = a[i, j] * b[lj, li]
        new_ji = a[j, i] * b[li, lj]
        
        old_ii = a[i, i] * b[li, li]
        old_jj = a[j, j] * b[lj, lj]
        old_ij = a[i, j] * b[li, lj]
        old_ji = a[j, i] * b[lj, li]
        
        total_delta += (new_ii + new_jj + new_ij + new_ji) - (old_ii + old_jj + old_ij + old_ji)
        
        if total_delta < 0 or (T > T_min and random.random() < np.exp(-total_delta / T)):
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += total_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # 4. Final Polish
    if time.time() < deadline - 1.0:
        best_p, best_cost = local_search(best_p, best_cost)

    return best_p.tolist()
# EVOLVE-BLOCK-END
