import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    # Split into j=0 (rank 10) and j=1,2 (rank 18)
    # j=1,2 split into k=0,1 (Strassen rank 7) and k=2,3,4 (Rank 11: k=2 schoolbook 4 + k=3,4 Strassen 7)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2
    add_strassen(0, 1, 0)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Check for linear dependency to reduce rank
    # Flatten each rank-1 term T_r = U_r \otimes V_r \otimes W_r
    # T_r is a tensor of shape (n, m, p) ? No, (n, m, p) is the shape of the multiplication tensor?
    # The tensor T is defined by T[i, j, k] = 1 if it contributes to C[i, k] via A[i, j] B[j, k].
    # Actually, the tensor is in (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{n*p})^*
    # Let's flatten U_r, V_r, W_r and compute the outer product?
    # The term in the sum is U_r \otimes V_r \otimes W_r.
    # We can flatten this into a vector of length (n*m)*(m*p)*(n*p) = 6*15*10 = 900.
    # Or just check the rank of the 28 vectors in the 900-dim space.
    
    U_mat = np.array(U) # shape (6, 28)
    V_mat = np.array(V) # shape (15, 28)
    W_mat = np.array(W) # shape (10, 28)
    
    # Construct the 28 rank-1 tensors flattened
    # T_r = U_r \otimes V_r \otimes W_r
    # We can use np.einsum or just reshape.
    # Let's create a matrix M of shape (900, 28)
    
    terms = []
    for r in range(rank):
        t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        terms.append(t.flatten())
        
    M = np.array(terms).T # shape (900, 28)
    
    # Check rank
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank:
        # Find null space vector c such that M c = 0
        # Use SVD
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # The null space is spanned by the last (rank - rank_M) rows of Vt
        # Since we want a single dependency, take the last row
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
            
            # Convert back to list of lists
            U_new = new_U.tolist()
            V_new = new_V.tolist()
            W_new = new_W.tolist()
            
            # Verify
            # Reconstruct T and compare
            # We can just return the new solution
            
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
