"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build rank-50 Strassen decomposition, then use efficient SVD-based
    rank reduction to find and eliminate linearly dependent terms.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Efficient rank reduction using SVD on the stacked tensor
    def try_reduce_rank_svd(U_arr, V_arr, W_arr, n, m, p):
        """Use SVD to find and eliminate linearly dependent columns."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        
        if current_rank <= 1:
            return U_arr, V_arr, W_arr, current_rank
        
        # Stack all columns into a single matrix for SVD
        # Shape: (n*m + m*p + p*n, current_rank)
        stacked = np.vstack([U_arr.T, V_arr.T, W_arr.T])  # (dim, rank)
        
        # SVD to find numerical rank
        try:
            U_svd, s_vals, Vt = np.linalg.svd(stacked, full_matrices=False)
        except:
            return U_arr, V_arr, W_arr, current_rank
        
        # Find effective rank (singular values > tolerance)
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank >= current_rank:
            return U_arr, V_arr, W_arr, current_rank
        
        # Keep only the top effective_rank right singular vectors
        # Reconstruct the stacked matrix
        stacked_reduced = (U_svd[:, :effective_rank] * s_vals[:effective_rank]) @ Vt[:effective_rank, :]
        
        # Extract reduced U, V, W
        dim_um = n * m
        dim_v = m * p
        dim_w = p * n
        
        U_reduced = stacked_reduced[:dim_um, :].T
        V_reduced = stacked_reduced[dim_um:dim_um+dim_v, :].T
        W_reduced = stacked_reduced[dim_um+dim_v:, :].T
        
        # Round to nearest integers if close
        U_reduced = np.round(U_reduced, 10)
        V_reduced = np.round(V_reduced, 10)
        W_reduced = np.round(W_reduced, 10)
        
        # Verify
        if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            return U_reduced, V_reduced, W_reduced, effective_rank
        
        return U_arr, V_arr, W_arr, current_rank
    
    # Attempt SVD-based reduction
    if time.time() < deadline - 20:
        U_red, V_red, W_red, new_rank = try_reduce_rank_svd(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Try additional reduction passes
    for _ in range(3):
        if time.time() > deadline - 15:
            break
        U_red, V_red, W_red, new_rank = try_reduce_rank_svd(best_U, best_V, best_W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
        else:
            break
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
