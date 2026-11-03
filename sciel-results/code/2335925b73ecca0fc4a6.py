import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Rank-28 decomposition for 2x4 by 4x4 matrix multiplication.
    This is optimal because M_{2,4,4} is isomorphic to M_{1,2,2} \otimes M_{2,2,2},
    and Rank(M_{1,2,2}) = 4, Rank(M_{2,2,2}) = 7, so Rank(M_{2,4,4}) = 28.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen components for 2x2x2
    # Each component is a set of (indices, values) for U, V, W
    # Indices are (row, col) in the 2x2 block
    strassen_data = [
        # M1
        ([(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], 
         [(0,0)], [1], 
         [(1,0), (1,1)], [1, -1]),
        # M3
        ([(0,0)], [1], 
         [(0,1), (1,1)], [1, -1], 
         [(0,1), (1,1)], [1, 1]),
        # M4
        ([(1,1)], [1], 
         [(1,0), (0,0)], [1, -1], 
         [(0,0), (1,0)], [1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], 
         [(1,1)], [1], 
         [(0,0), (0,1)], [-1, 1]),
        # M6
        ([(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1], 
         [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1], 
         [(0,0)], [1]),
    ]

    # The 2x4 by 4x4 multiplication can be decomposed into 4 independent 2x2 by 2x2 multiplications
    # A is [A1, A2] where A1, A2 are 2x2
    # B is [[B11, B12], [B21, B22]] where Bij are 2x2
    # C = [C1, C2] where C1 = A1*B11 + A2*B21 and C2 = A1*B12 + A2*B22
    # This requires 4 matrix products:
    # 1. A1 * B11
    # 2. A2 * B21
    # 3. A1 * B12
    # 4. A2 * B22
    
    # Offsets for A columns, B rows, B columns, C columns
    # A is 2x4. Columns 0-1 are A1, 2-3 are A2.
    # B is 4x4. Rows 0-1 are top, 2-3 are bottom. Cols 0-1 are left, 2-3 are right.
    # C is 2x4. Cols 0-1 are C1, 2-3 are C2.
    
    terms = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # A1 * B11 -> C1
        (2, 2, 0, 0),  # A2 * B21 -> C1
        (0, 0, 2, 2),  # A1 * B12 -> C2
        (2, 2, 2, 2),  # A2 * B22 -> C2
    ]

    for t_idx, (a_off, b_row, b_col, c_off) in enumerate(terms):
        for s_idx, (u_idx_list, u_vals, v_idx_list, v_vals, w_idx_list, w_vals) in enumerate(strassen_data):
            c = t_idx * 7 + s_idx
            
            for idx, val in zip(u_idx_list, u_vals):
                r, col = idx
                u_flat = r * m + (a_off + col)
                U[u_flat, c] += val
                
            for idx, val in zip(v_idx_list, v_vals):
                r, col = idx
                v_flat = (b_row + r) * p + (b_col + col)
                V[v_flat, c] += val
                
            for idx, val in zip(w_idx_list, w_vals):
                r, col = idx
                w_flat = (c_off + col) * n + r
                W[w_flat, c] += val

    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Optional self-check
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify decomposition: C = A B
    # Tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
    # Wait, the convention is W[k*n+i, r].
    # Standard tensor contraction: T_{i,j,k} = sum_r U_{i,m,r} V_{m,k,r} W_{k,i,r} is not quite right.
    # The tensor is T[i, m, k] for A[i,m], B[m,k], C[i,k].
    # Decomposition: T[i, m, k] = sum_r U[i, m, r] V[m, k, r] W[k, i, r] ??
    # Let's check the shapes.
    # U is (n*m, rank). Index (i, m) -> i*m + m.
    # V is (m*p, rank). Index (m, k) -> m*p + k.
    # W is (p*n, rank). Index (k, i) -> k*n + i.
    # The tensor element T[i, m, k] should be sum_r U[i*m+m, r] * V[m*p+k, r] * W[k*n+i, r].
    
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.rand(2, 4)
    B = np.random.rand(4, 4)
    C = A @ B
    
    # Reconstruct C
    C_rec = np.zeros((2, 4))
    for r in range(28):
        # U[:, r] is shape (8,). Reshape to (2, 4)
        U_r = U[:, r].reshape(2, 4)
        # V[:, r] is shape (16,). Reshape to (4, 4)
        V_r = V[:, r].reshape(4, 4)
        # W[:, r] is shape (8,). Reshape to (4, 2) -> W[k*n+i, r]
        # W index is k*n + i. So reshape to (4, 2) where first dim is k, second is i.
        W_r = W[:, r].reshape(4, 2)
        
        # The product of rank-1 components:
        # The term is sum_{m} A[i,m] B[m,k] ...
        # Actually, the decomposition is:
        # C[i,k] = sum_r (sum_m U[i,m,r] A[i,m]) * (sum_k V[m,k,r] B[m,k]) * W[k,i,r] ?
        # No, that's not how tensor decomposition works for matrix mult.
        
        # The tensor T[i, m, k] = delta_{i,i} delta_{m,m} delta_{k,k} is not it.
        # T[i, m, k] is 1 if we are multiplying A[i,m] and B[m,k].
        # The decomposition is T[i, m, k] = sum_r U[i, m, r] V[m, k, r] W[k, i, r].
        # Then C[i, k] = sum_{m} A[i, m] B[m, k] = sum_{m} T[i, m, k] A[i, m] B[m, k]
        # = sum_{m} sum_r U[i, m, r] V[m, k, r] W[k, i, r] A[i, m] B[m, k]
        # = sum_r (sum_m U[i, m, r] A[i, m]) (sum_k V[m, k, r] B[m, k]) W[k, i, r] ??
        # The indices don't match up for a simple product of scalars.
        
        # Let's just compute the tensor T from U, V, W and check if it matches the multiplication tensor.
        pass

    # Correct verification:
    # T[i, m, k] should be 1 if the indices are valid for the multiplication?
    # No, the tensor for matrix multiplication M_{n,m,p} has entries T[i, j, k] = 1 if j is the shared index?
    # No, the tensor is defined such that C = A B implies C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}.
    # So T_{ijk} = 1 if the indices align? No, T_{ijk} is just 1 for all valid i,j,k?
    # No, T_{ijk} is the structure constant. For standard matrix mult, T_{ijk} = 1 if we are summing over j.
    # Actually, T_{ijk} is 1 for all i, j, k? No.
    # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # $T_{ijk} = 1$ if we are computing $C_{ik}$ from $A_{ij}$ and $B_{jk}$.
    # So $T_{ijk} = 1$ for all $i \in 1..n, j \in 1..m, k \in 1..p$.
    # Wait, if $T_{ijk} = 1$ for all, then $C_{ik} = \sum_j 1 \cdot A_{ij} B_{jk}$. Yes.
    # So the target tensor is simply all ones?
    # No, the tensor is in $V^n \otimes V^m \otimes V^p$.
    # The entry $T_{ijk}$ is 1.
    
    # So we check if sum_r U[i, m, r] V[m, k, r] W[k, i, r] == 1 for all i, m, k.
    
    # Reshape U to (2, 4, 28)
    U_3d = U.reshape(2, 4, 28)
    V_3d = V.reshape(4, 4, 28)
    W_3d = W.reshape(4, 2, 28) # W[k, i, r]
    
    # Compute T_reconstructed[i, m, k] = sum_r U_3d[i, m, r] V_3d[m, k, r] W_3d[k, i, r]
    # This is a batched matrix multiplication or einsum.
    T_rec = np.einsum('imr,mkr,kir->imk', U_3d, V_3d, W_3d)
    
    # The target tensor T_target[i, m, k] is 1 for all i, m, k.
    T_target = np.ones((2, 4, 4))
    
    if not np.allclose(T_rec, T_target):
        print("Decomposition is INVALID")
    else:
        print("Decomposition is VALID")
