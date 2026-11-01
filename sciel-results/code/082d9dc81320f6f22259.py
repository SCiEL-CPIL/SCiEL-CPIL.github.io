import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 2, 4]

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Note: The task defines W[k*n+i, r]. 
    # Let's map indices carefully.
    # U is (n*m, rank). Index i*m + j? No, U index is usually i*m+j for A[i,j].
    # V is (m*p, rank). Index j*p + k for B[j,k].
    # W is (p*n, rank). Index k*n + i for C[i,k] or C[k,i]?
    # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the output C is indexed by k (row of B/col of C?) and i (row of A/row of C?).
    # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # If W index is k*n + i, then W[k*n+i] corresponds to C[i, k].
    # Let's verify the tensor definition.
    # The tensor T for matrix multiplication has non-zero entries T[i, j, k] = 1 if we consider the basis.
    # Actually, the tensor is defined such that <T, A \otimes B \otimes C> = 0 for all A, B, C^T?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # The "target" tensor is the matrix multiplication tensor.
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # So T[i, j, k] is 1 for all valid i, j, k.
    # Wait, the tensor is trilinear. 
    # T(A, B, C) = trace(A^T B C^T) ? No.
    # The standard matrix multiplication tensor T_{ijk} is 1 if the indices form a valid chain i->j->k.
    # In our case: A is 2x2, B is 2x4. C is 2x4.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has dimensions (n*m) x (m*p) x (p*n).
    # Entry (i*m+j, j*p+k, k*n+i) should be 1?
    # Let's check the rank 14 construction.
    # U[0][0]=1 (A00), V[0][0]=1 (B00), W[0][0]=1 (C00).
    # i=0, j=0, k=0.
    # U index: 0*2+0 = 0.
    # V index: 0*4+0 = 0.
    # W index: 0*2+0 = 0.
    # This matches T[0, 0, 0] = 1.
    # So the target tensor T has T[u_idx, v_idx, w_idx] = 1 if u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i.
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_target[u_idx, v_idx, w_idx] = 1.0

    # Compute T_decomp
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u_col = np.array(U[:, r])
        v_col = np.array(V[:, r])
        w_col = np.array(W[:, r])
        # Outer product
        T_decomp += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
        
    return np.allclose(T_decomp, T_target, atol=1e-9)

def _search_rank12(start_time, max_time=10.0):
    """Search for a rank 12 decomposition using sparse patterns."""
    n, m, p = 2, 2, 4
    rank = 12
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_target[u_idx, v_idx, w_idx] = 1.0

    # We will try to find U, V, W with entries in {-1, 0, 1}
    # This is a 3-SAT / ILP problem.
    # Heuristic: Random sparse initialization + Local Search / Gaussian Elimination
    
    # Number of non-zeros in U, V, W
    # Total non-zeros in T is n*m*p = 16.
    # Each rank-1 term contributes at most (nnz_u * nnz_v * nnz_w) non-zeros.
    # To cover 16 non-zeros with 12 terms, average nnz is low.
    
    # Let's try a structured search.
    # We know U has 4 rows. V has 8 rows. W has 8 rows.
    
    # Strategy: 
    # 1. Randomly generate sparse U, V, W with density ~ 0.5
    # 2. Solve for the remaining degrees of freedom?
    # Actually, if we fix the support (positions of non-zeros), the values are determined by the linear system.
    # But the support itself is unknown.
    
    # Let's use a simpler heuristic:
    # Try to construct U, V, W such that they satisfy the tensor equation.
    # We can fix U and V, and solve for W?
    # T = sum_r U_r V_r W_r^T
    # This is not linear in W if U, V are fixed? No, it is linear in W.
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # For a fixed i, j, T[i, j, :] = sum_r (U[i, r] V[j, r]) W[:, r]
    # This is a linear system for W[:, r].
    
    # Let's try:
    # 1. Randomly initialize U and V with entries in {-1, 0, 1}.
    # 2. Construct the linear system for W.
    # 3. Solve for W.
    # 4. Check if W has small integer/rational entries.
    
    num_trials = 5000
    random.seed(42)
    
    for t in range(num_trials):
        if time.time() - start_time > max_time:
            return None
            
        # Generate sparse U and V
        # U: 4x12, V: 8x12
        # Density 0.5
        U = np.random.choice([-1, 0, 1], size=(n*m, rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, rank))
        
        # Build the linear system for W
        # For each (i, j) in A, B indices:
        # We have a vector equation for W[:, :]
        # Let M_ij be a (p*n) x (p*n * rank) ? No.
        # T[i*m+j, :, :] is a matrix of size (m*p) x (p*n).
        # T[i*m+j, v_idx, w_idx] = sum_r U[i*m+j, r] * V[v_idx, r] * W[w_idx, r]
        
        # Let's flatten the unknowns.
        # Unknowns: W[w_idx, r] for all w_idx, r. Total 8*12 = 96 unknowns.
        # Equations: For each u_idx (4) and v_idx (8), we have p*n (8) equations?
        # No, the equation is for each specific w_idx.
        # T[u_idx, v_idx, w_idx] = sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
        
        # This is a system of 4*8*8 = 256 equations.
        # Unknowns: 8*12 = 96.
        # Overdetermined.
        
        # We can set up a least squares problem.
        # A_matrix * w_vec = b_vec
        # w_vec is the flattened W (96 entries).
        # b_vec is the flattened T_target (256 entries).
        
        # For each u_idx, v_idx, w_idx:
        # b[u_idx, v_idx, w_idx] = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        
        # Let's build the matrix A for the least squares.
        # Rows: 256. Cols: 96.
        A_ls = np.zeros((n*m * m*p * p*n, p*n * rank))
        b_ls = np.zeros(n*m * m*p * p*n)
        
        row = 0
        for u_idx in range(n*m):
            for v_idx in range(m*p):
                for w_idx in range(p*n):
                    b_ls[row] = T_target[u_idx, v_idx, w_idx]
                    # Coefficients for W[w_idx, r]
                    for r in range(rank):
                        coeff = U[u_idx, r] * V[v_idx, r]
                        if coeff != 0:
                            # Column index in A_ls
                            col = w_idx * rank + r
                            A_ls[row, col] = coeff
                    row += 1
        
        # Solve least squares
        try:
            W_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_ls, b_ls, rcond=None)
        except Exception:
            continue
            
        # Check residuals
        if residuals[0] > 1e-9:
            continue
            
        # Check if W_sol has small integer values
        W_sol = W_sol.reshape(p*n, rank)
        
        # Round to nearest integer
        W_int = np.round(W_sol)
        
        # Verify if W_int works
        # Reconstruct T
        T_check = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_check += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W_int[:, r])
            
        if np.allclose(T_check, T_target, atol=1e-9):
            # Found a solution!
            # Convert to lists of ints
            U_list = U.astype(int).tolist()
            V_list = V.astype(int).tolist()
            W_list = W_int.astype(int).tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 140.0 # Safety margin from 150s
    
    # First, try to find a rank 12 solution
    # We give it 10 seconds to search
    result_12 = _search_rank12(start_time, max_time=10.0)
    
    if result_12 is not None:
        # Verify once more to be absolutely sure
        if _verify_decomposition(result_12['U'], result_12['V'], result_12['W'], n, m, p):
            return result_12
        else:
            # Should not happen, but fallback
            pass
    
    # If rank 12 not found, return the verified rank 14 solution
    return _build_rank14()

