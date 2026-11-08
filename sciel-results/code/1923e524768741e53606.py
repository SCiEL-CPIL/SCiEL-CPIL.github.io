"""
Matrix multiplication 2x4 x 4x4 - rank-minimal exact tensor decomposition.
Uses a structured search over rank r, trying to find an exact decomposition.
For n=2, m=4, p=4, the known rank is 16 (schoolbook). We search for lower rank.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS

def construct():
    """
    Returns factor matrices U, V, W for the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Score = rank (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 130  # safety margin for 150s limit
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} in flattened form
    # T has shape (n, m, p), where T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # Actually, the tensor for matrix multiplication C = A * B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k) since every product contributes.
    
    # The tensor T of shape (n, m, p) where T[i,j,k] = 1 if the (i,j) of A and (j,k) of B multiply to give (i,k) of C
    # In fact T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the "multiplication tensor" M[i,j,k] satisfies C[i,k] = sum_{j} A[i,j] * M[i,j,k] * B[j,k]
    # where M[i,j,k] = 1 if and only if... no.
    
    # The tensor T of matrix multiplication is defined as:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where (i,j) in A, (j,k) in B, and the product goes to (i,k) in C
    # More precisely: T[i, j, k] = 1 if the (i,j)-entry of A and (j,k)-entry of B both participate in computing C[i,k]
    # This is always 1 for valid indices since C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor is a 3-tensor of shape (n, m, p) with T[i,j,k] = 1 for all i,j,k.
    # Wait no. Let me reconsider.
    
    # For matrix multiplication C = AB where A is n×m and B is m×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # The multiplication tensor T ∈ R^{n × m × p} is defined as:
    # T[i, j, k] = 1 if (i,j) is a valid index for A, (j,k) is valid for B, and the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since every such product contributes, T[i,j,k] = 1 for all valid i,j,k.
    
    # Actually, I need to be more careful. The standard tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i ⊗ f_j ⊗ g_k  where the term appears if (i,j,k) is a valid triple
    # In our case, every triple (i,j,k) with i∈[0,n), j∈[0,m), k∈[0,p) is valid, so T[i,j,k] = 1.
    
    # Hmm, but that would mean the tensor is all-ones, which has rank 1, which is clearly wrong.
    # Let me reconsider.
    
    # The correct definition: The matrix multiplication tensor T_{n,m,p} ∈ R^{n×m × m×p × p×n}
    # is defined by: T_{(i,j),(j',k),(k',i')} = 1 if j=j' and k=k' and i=i'... no.
    
    # Standard definition: T is a 3-tensor with T[a,b,c] where a indexes rows of A × cols of A, 
    # b indexes rows of B × cols of B, c indexes rows of C × cols of C.
    # T[(i,j),(j,k),(i,k)] = 1, and 0 otherwise.
    
    # In flat indexing:
    # U index: i*m + j for i in [0,n), j in [0,m)
    # V index: j*p + k for j in [0,m), k in [0,p)  
    # W index: k*n + i for k in [0,p), i in [0,n)
    
    # T[U_idx, V_idx, W_idx] = 1 if U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    # i.e., the j in U matches the j in V, and the k in V matches the k in W, and the i in U matches the i in W.
    
    n_rows_U = n * m  # 8
    n_rows_V = m * p  # 16
    n_rows_W = p * n  # 8
    
    T = np.zeros((n_rows_U, n_rows_V, n_rows_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try to find a low-rank decomposition
    # We'll search over rank r from 1 upward
    # For each r, we try to find U (n_rows_U × r), V (n_rows_V × r), W (n_rows_W × r)
    # such that T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    
    # Start with schoolbook rank = n*m*p = 32
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U_school = np.zeros((n_rows_U, best_rank))
    V_school = np.zeros((n_rows_V, best_rank))
    W_school = np.zeros((n_rows_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try lower ranks using randomized approach
    # For a given rank r, we can try to solve for the factors
    # This is a hard problem, so we'll use a heuristic
    
    def try_rank(r, seed=42):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize factors randomly
        U = rng.randn(n_rows_U, r)
        V = rng.randn(n_rows_V, r)
        W = rng.randn(n_rows_W, r)
        
        # Use ALS (Alternating Least Squares) to refine
        for it in range(50):
            # Update U
            for u_idx in range(n_rows_U):
                # T[u_idx, :, :] = sum_r U[u_idx, r] * V[:, r] * W[:, r]^T
                # Fix: T[u_idx, v, w] = sum_r U[u_idx, r] * V[v, r] * W[w, r]
                # For fixed u_idx, T[u_idx, v, w] = sum_r U[u_idx, r] * (V[v, r] * W[w, r])
                # Let B_r = V[:, r] ⊗ W[:, r], a vector of length n_rows_V * n_rows_W
                # T[u_idx] (flattened) = sum_r U[u_idx, r] * B_r
                # Solve: U[u_idx, :] = argmin sum_r U[u_idx, r] * B_r ≈ T_flat[u_idx]
                
                T_flat = T[u_idx, :, :].flatten()
                B = np.zeros((r, n_rows_V * n_rows_W))
                for rr in range(r):
                    B[rr, :] = (V[:, rr] * W[:, rr]).flatten()
                
                # Solve B^T U[u_idx, :]^T = T_flat
                # i.e., U[u_idx, :] = (B^T B)^{-1} B^T T_flat
                try:
                    U[u_idx, :] = np.linalg.lstsq(B.T, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for v_idx in range(n_rows_V):
                T_flat = T[:, v_idx, :].flatten()
                B = np.zeros((r, n_rows_U * n_rows_W))
                for rr in range(r):
                    B[rr, :] = (U[:, rr] * W[:, rr]).flatten()
                try:
                    V[v_idx, :] = np.linalg.lstsq(B.T, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for w_idx in range(n_rows_W):
                T_flat = T[:, :, w_idx].flatten()
                B = np.zeros((r, n_rows_U * n_rows_V))
                for rr in range(r):
                    B[rr, :] = (U[:, rr] * V[:, rr]).flatten()
                try:
                    W[w_idx, :] = np.linalg.lstsq(B.T, T_flat, rcond=None)[0]
                except:
                    pass
        
        # Check if we found an exact solution
        # Compute reconstruction error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m+j
                    v_idx = j*p+k
                    w_idx = k*n+i
                    recon = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    error += abs(recon - 1.0)
        
        if error < 1e-8:
            return U, V, W
        return None
    
    # Try various ranks from low to high
    for r in range(1, best_rank):
        if time.time() > deadline - 20:
            break
        # Try multiple random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            result = try_rank(r, seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        lst = []
        for row in mat:
            lst_row = []
            for val in row:
                # Round to nearest half-integer if close
                val_rounded = round(val * 2) / 2
                if abs(val - val_rounded) < 1e-6:
                    val = val_rounded
                lst_row.append(val)
            lst.append(lst_row)
        return lst
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
