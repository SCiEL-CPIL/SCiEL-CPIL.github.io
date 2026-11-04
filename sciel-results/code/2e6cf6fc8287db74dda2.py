"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For correct decomposition, T[i,j,k] should be 1 if the indices form a valid
    # matrix multiplication term, i.e., for all i,j,k it should be 1 (since every
    # (i,j,k) triple contributes to exactly one output element C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor for matmul is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor has 1s at all (i,j,k).
    
    # Compute sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-tensor. We can check it equals 1 for all (i,j,k).
    # Efficient way: for each c, the contribution is outer product of U[:,c], V[:,c], W[:,c]
    # But that's expensive. Instead, we can use the fact that:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
    
    # Let's do it more carefully. For fixed i,k, the sum over j of the tensor values
    # should give us the correct matmul structure. Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let me just compute it directly for small sizes.
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u_c = U[:, c]  # shape (n*m,)
        v_c = V[:, c]  # shape (m*p,)
        w_c = W[:, c]  # shape (p*n,)
        # Reshape
        u_c = u_c.reshape(n, m)
        v_c = v_c.reshape(m, p)
        w_c = w_c.reshape(p, n)
        # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    # T should be all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _verify_fast(U, V, W, tol=1e-10):
    """Faster verification using tensor operations."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T == 1 for all i,j,k
    
    # Alternative: check that for random A, B: C = A @ B equals the decomposition result
    # C[i,k] = sum_j sum_c U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c]
    # Wait, that's not quite right. Let me think again.
    # 
    # The decomposition says: C[i,k] = sum_j A[i,j] B[j,k]
    # And also: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k' ... )
    # 
    # Actually the standard formulation: 
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Hmm, that doesn't look right either.
    #
    # Let me use the trace convention from the problem: W[k*n+i, r]
    # The tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_{j,c} A[i,j] B[j,k] * T[i,j,k,c]... no.
    #
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_k V[j*p+k,c] B[j,k]) W[k*n+i,c]
    # This is getting confusing. Let me just use the direct tensor check.
    
    # Direct approach: build the full tensor and check
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal 1 for all i,j,k
    
    # Use einsum or manual
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:n*m, c].reshape(n, m)
        v = V[:m*p, c].reshape(m, p)
        w = W[:p*n, c].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is a 3D sum
        for i in range(n):
            for k in range(p):
                T[i, :, k] += u[i, :] * (v * w[k, :].reshape(1, p))
    
    return np.allclose(T, 1.0, atol=tol)

def _verify_einsum(U, V, W, tol=1e-10):
    """Use einsum for verification."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try to remove columns one by one
    # Strategy: try removing each column, check if still valid
    # This is O(rank * verification_cost)
    
    # Use the einsum verification which is fast
    def is_valid(U_mat, V_mat, W_mat):
        return _verify_einsum(U_mat, V_mat, W_mat, tol=1e-9)
    
    # Greedy column removal
    current_rank = schoolbook_rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try removing columns in various orders
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Try in natural order
            order = list(range(current_rank))
        elif attempt == 1:
            order = list(range(current_rank - 1, -1, -1))
        elif attempt == 2:
            order = list(range(current_rank))
            random.seed(42 + attempt)
            random.shuffle(order)
        elif attempt == 3:
            order = list(range(current_rank))
            random.seed(123 + attempt)
            random.shuffle(order)
        else:
            order = list(range(current_rank))
            random.seed(999 + attempt)
            random.shuffle(order)
        
        U_try = U_cur.copy()
        V_try = V_cur.copy()
        W_try = W_cur.copy()
        rank_try = current_rank
        
        removed = 0
        for idx in order:
            if time.time() > deadline:
                break
            if idx >= rank_try:
                continue
            # Try removing column idx
            U_new = np.delete(U_try, idx, axis=1)
            V_new = np.delete(V_try, idx, axis=1)
            W_new = np.delete(W_try, idx, axis=1)
            
            if is_valid(U_new, V_new, W_new):
                U_try = U_new
                V_try = V_new
                W_try = W_new
                rank_try -= 1
                removed += 1
        
        if rank_try < best_rank:
            best_rank = rank_try
            best_U = U_try.copy()
            best_V = V_try.copy()
            best_W = W_try.copy()
            U_cur = U_try.copy()
            V_cur = V_try.copy()
            W_cur = W_try.copy()
            current_rank = rank_try
    
    # If we found improvement, return it
    if best_rank < schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Otherwise return schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
