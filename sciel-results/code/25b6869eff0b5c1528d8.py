"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 105  # safety margin before 120s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k', k*n+i'] = 1 iff j==j' and i==i' and k==k'
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    # Verify schoolbook is correct
    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch
    best_V = V_sch
    best_W = W_sch

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    # Try ranks in descending order from 68 down to 54
    # Known lower bound for 3x4x6 is around 54-56
    target_ranks = list(range(68, 53, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)

            # Refine using alternating least squares with more iterations
            for iteration in range(30):
                if time.time() > deadline:
                    break
                
                # Update U
                MW = np.einsum('br,cr->bcr', V_init, W_init)
                MW_flat = MW.transpose(2, 0, 1).reshape(target_rank, m * p * p * n)
                try:
                    G = MW_flat @ MW_flat.T
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                    U_new = T_A @ MW_flat.T @ G_inv
                except np.linalg.LinAlgError:
                    break

                # Update V
                UW = np.einsum('ar,cr->acr', U_new, W_init)
                UW_flat = UW.transpose(2, 0, 1).reshape(target_rank, n * m * p * n)
                try:
                    G = UW_flat @ UW_flat.T
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                    V_new = T_B @ UW_flat.T @ G_inv
                except np.linalg.LinAlgError:
                    break

                # Update W
                UV = np.einsum('ar,br->abr', U_new, V_new)
                UV_flat = UV.transpose(2, 0, 1).reshape(target_rank, n * m * m * p)
                try:
                    G = UV_flat @ UV_flat.T
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                    W_new = T_C @ UV_flat.T @ G_inv
                except np.linalg.LinAlgError:
                    break

                U_init, V_init, W_init = U_new, V_new, W_new

            # Check if we found an exact solution
            if verify(U_init, V_init, W_init, atol=1e-8):
                rounded = round_and_verify(U_init, V_init, W_init)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break  # Found a valid lower-rank decomposition, stop searching

    # If no better found, use schoolbook
    if best_rank == rank_sch:
        U_out = U_sch
        V_out = V_sch
        W_out = W_sch
    else:
        U_out = best_U
        V_out = best_V
        W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
