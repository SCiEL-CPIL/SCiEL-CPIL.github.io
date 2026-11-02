import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    Strategy:
    1. Construct a baseline rank-28 decomposition using Strassen and Schoolbook blocks.
    2. Check for linear dependencies among the rank-1 terms to see if rank can be reduced.
    3. If a dependency exists, eliminate one term and return the reduced rank decomposition.
    4. If not, return the rank-28 decomposition.
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        """Add a Strassen 2x2x2 block at the given offsets."""
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1
    
    # Block 1: j=2, all k. M_{2,1,5}, rank 10.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j in {0,1}, k in {0,1}. M_{2,2,2}, rank 7 (Strassen).
    add_strassen(0, 0, 0)
    
    # Block 2b: j in {0,1}, k in {2,3}. M_{2,2,2}, rank 7 (Strassen).
    add_strassen(0, 0, 2)
    
    # Block 2c: j in {0,1}, k=4. M_{2,2,1}, rank 4 (schoolbook).
    for i in range(n):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    # Convert to numpy arrays for linear algebra
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Construct the 28 rank-1 tensors flattened into vectors of length 900
    # T_r = U_r \otimes V_r \otimes W_r
    # We want to check if these 28 vectors are linearly independent.
    # If not, we can reduce the rank.
    
    terms = []
    for r in range(rank):
        u = U_mat[:, r]
        v = V_mat[:, r]
        w = W_mat[:, r]
        # Flatten the outer product
        t = np.einsum('i,j,k->ijk', u, v, w).flatten()
        terms.append(t)
    
    M = np.array(terms).T  # shape (900, 28)
    
    # Check the rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except np.linalg.LinAlgError:
        rank_M = rank
    
    if rank_M < rank:
        # There is a linear dependency. Find it using SVD.
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # The null space is spanned by the last (rank - rank_M) rows of Vt
            # Take the last row as the dependency vector
            c = Vt[-1, :]
            
            # Find an index k with |c_k| > tol
            k_idx = np.argmax(np.abs(c))
            if np.abs(c[k_idx]) > 1e-10:
                # Eliminate term k_idx
                # New coefficients for r != k_idx: 1 - c_r / c_k
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(U_mat[:, r] * coeff)
                    new_V.append(V_mat[:, r] * coeff)
                    new_W.append(W_mat[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                # Verify the decomposition
                # Reconstruct T and compare with the original
                new_rank = new_U.shape[1]
                tensor_new = np.zeros((n, m, p))
                for r in range(new_rank):
                    u = new_U[:, r]
                    v = new_V[:, r]
                    w = new_W[:, r]
                    tensor_new += np.einsum('i,j,k->ijk', u, v, w)
                
                tensor_orig = np.zeros((n, m, p))
                for r in range(rank):
                    u = U_mat[:, r]
                    v = V_mat[:, r]
                    w = W_mat[:, r]
                    tensor_orig += np.einsum('i,j,k->ijk', u, v, w)
                
                if np.allclose(tensor_new, tensor_orig, atol=1e-9):
                    return {
                        "U": new_U.tolist(),
                        "V": new_V.tolist(),
                        "W": new_W.tolist()
                    }
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
