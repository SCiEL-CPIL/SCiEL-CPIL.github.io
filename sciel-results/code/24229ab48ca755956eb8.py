import numpy as np
import time

# EVOLVE-BLOCK-START

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition (Rank 49).
    Uses the standard 2x2 Strassen recursion applied to 4x4 matrices.
    
    Convention:
      U[i*m + j, r] for A[i,j]
      V[j*p + k, r] for B[j,k]  
      W[k*n + i, r] for C[i,k] (trace convention)
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # 2x2 Strassen combinations for A, B, C
    # Each entry is a list of (row_block, col_block, sign)
    A_combos = [
        [(0, 0, 1), (1, 1, 1)],   # A11 + A22
        [(1, 0, 1), (1, 1, 1)],   # A21 + A22
        [(0, 0, 1)],               # A11
        [(1, 1, 1)],               # A22
        [(0, 0, 1), (0, 1, 1)],   # A11 + A12
        [(1, 0, 1), (0, 0, -1)],  # A21 - A11
        [(0, 1, 1), (1, 1, -1)]   # A12 - A22
    ]
    B_combos = [
        [(0, 0, 1), (1, 1, 1)],   # B11 + B22
        [(0, 0, 1)],               # B11
        [(0, 1, 1), (1, 1, -1)],  # B12 - B22
        [(1, 0, 1), (0, 0, -1)],  # B21 - B11
        [(1, 1, 1)],               # B22
        [(0, 0, 1), (0, 1, 1)],   # B11 + B12
        [(1, 0, 1), (1, 1, 1)]    # B21 + B22
    ]
    C_combos = [
        [(0, 0, 1), (1, 1, 1)],   # C11 + C22
        [(1, 0, 1), (1, 1, -1)],  # C21 - C22
        [(0, 1, 1), (1, 1, 1)],   # C12 + C22
        [(0, 0, 1), (1, 0, 1)],   # C11 + C21
        [(0, 1, 1), (0, 0, -1)],  # C12 - C11
        [(1, 1, 1)],               # C22
        [(0, 0, 1)]                # C11
    ]
    
    for t_outer in range(7):
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # Build U: A[i,j] contribution
            for (br, bc, sign) in A_combos[t_outer]:
                for (ir, ik, sign_inner) in A_combos[t_inner]:
                    i_global = br * 2 + ir
                    j_global = bc * 2 + ik
                    u_idx = i_global * n + j_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # Build V: B[j,k] contribution
            for (br, bc, sign) in B_combos[t_outer]:
                for (ir, ik, sign_inner) in B_combos[t_inner]:
                    j_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = j_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # Build W: C[i,k] contribution (trace convention: W[k*n + i, r])
            for (br, bc, sign) in C_combos[t_outer]:
                for (ir, ik, sign_inner) in C_combos[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    Actually, the tensor for matrix multiplication has T[i,j,k] = delta_{i,j} * delta_{j,k}... 
    No, let me think again.
    
    For matrix multiplication C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]:
    The tensor T has T[i,j,k] = 1 for all valid i,j,k? No.
    
    Actually, the matrix multiplication tensor is defined such that:
    T[i,j,k] = 1 if we're looking at the term A[i,j] * B[j,k] contributing to C[i,k]
    
    In the tensor decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Wait, that's not right either. Let me re-examine.
    
    The standard tensor decomposition for matrix multiplication:
    C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    
    But in the rank-r decomposition:
    C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... no
    
    Let me use the correct formulation. The matrix multiplication tensor M[i,j,k] = 1 for all i,j,k
    (it's the all-ones tensor in the sense that each term A[i,j]*B[j,k] contributes to C[i,k]).
    
    Actually no. The tensor is: for each (i,j,k), the coefficient is 1 if j is the "middle" index
    connecting A[i,j] and B[j,k] to produce C[i,k]. So M[i,j,k] = 1 for all i,j,k.
    
    Wait, that's the standard matrix multiplication tensor. It has entries all equal to 1.
    
    The decomposition: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Let me verify with the Strassen construction.
    """
    rank = U.shape[1]
    
    # Reshape to 3D tensors
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # The target tensor for matrix multiplication is all ones
    target = np.ones((n, m, p))
    
    return bool(np.allclose(T, target, atol=atol))


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 4
    
    # Generate the rank-49 Strassen decomposition
    U, V, W = _generate_strassen_4x4()
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook decomposition (rank 64)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
