import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor is T[i,j,k] = 1 for all valid i, j, k.
    Decomposition: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m + j, :], V_np[j*p + k, :] * W_np[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x3():
    """
    Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indices: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=3
    """
    # Strassen's 7 bilinear forms for 3x3x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A22+A33)(B22+B33)
    # M3 = (A11+A33)(B11+B33)
    # M4 = A12(B21-B11)
    # M5 = A21(B12-B22)
    # M6 = A31(B13-B33)
    # M7 = A13(B32-B22)
    
    # U factors: coefficients for A elements
    # V factors: coefficients for B elements
    # W factors: coefficients for C elements
    
    # Let's define the 7 terms explicitly
    # Term 1: (A11+A22)(B11+B22)
    # Term 2: (A22+A33)(B22+B33)
    # Term 3: (A11+A33)(B11+B33)
    # Term 4: A12(B21-B11)
    # Term 5: A21(B12-B22)
    # Term 6: A31(B13-B33)
    # Term 7: A13(B32-B22)
    
    # U: 9x7 matrix, row index = i*3+j (i,j in 0..2)
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U[0, 0] = 1  # A[0][0]
    U[3, 0] = 1  # A[1][1] (i=1,j=1 -> index 3)
    V[0, 0] = 1  # B[0][0]
    V[3, 0] = 1  # B[1][1] (j=1,k=1 -> index 3)
    # C elements: C[0][0] += M1, C[1][1] += M1, C[0][1] -= M1, C[1][0] -= M1
    W[0, 0] = 1   # C[0][0] (k=0,i=0)
    W[3, 0] = 1   # C[1][1] (k=1,i=1)
    W[1, 0] = -1  # C[0][1] (k=1,i=0)
    W[2, 0] = -1  # C[1][0] (k=0,i=1)
    
    # Term 2: (A11+A22)(B11+B22)
    U[3, 1] = 1  # A[1][1]
    U[6, 1] = 1  # A[2][2] (i=2,j=2 -> index 6)
    V[3, 1] = 1  # B[1][1]
    V[6, 1] = 1  # B[2][2] (j=2,k=2 -> index 6)
    # C elements: C[1][1] += M2, C[2][2] += M2, C[1][2] -= M2, C[2][1] -= M2
    W[3, 1] = 1   # C[1][1]
    W[8, 1] = 1   # C[2][2] (k=2,i=2)
    W[5, 1] = -1  # C[1][2] (k=2,i=1)
    W[6, 1] = -1  # C[2][1] (k=1,i=2)
    
    # Term 3: (A00+A22)(B00+B22)
    U[0, 2] = 1  # A[0][0]
    U[6, 2] = 1  # A[2][2]
    V[0, 2] = 1  # B[0][0]
    V[6, 2] = 1  # B[2][2]
    # C elements: C[0][0] += M3, C[2][2] += M3, C[0][2] -= M3, C[2][0] -= M3
    W[0, 2] = 1   # C[0][0]
    W[8, 2] = 1   # C[2][2]
    W[2, 2] = -1  # C[0][2] (k=2,i=0)
    W[7, 2] = -1  # C[2][0] (k=0,i=2)
    
    # Term 4: A01(B10-B00)
    U[1, 3] = 1  # A[0][1] (i=0,j=1 -> index 1)
    V[3, 3] = 1  # B[1][0] (j=1,k=0 -> index 3)
    V[0, 3] = -1 # B[0][0]
    # C elements: C[0][0] += M4, C[0][1] -= M4
    W[0, 3] = 1   # C[0][0]
    W[1, 3] = -1  # C[0][1]
    
    # Term 5: A10(B01-B11)
    U[3, 4] = 1  # A[1][0] (i=1,j=0 -> index 3)
    V[1, 4] = 1  # B[0][1] (j=0,k=1 -> index 1)
    V[3, 4] = -1 # B[1][1]
    # C elements: C[1][1] += M5, C[1][0] -= M5
    W[3, 4] = 1   # C[1][1]
    W[2, 4] = -1  # C[1][0]
    
    # Term 6: A20(B02-B22)
    U[6, 5] = 1  # A[2][0] (i=2,j=0 -> index 6)
    V[2, 5] = 1  # B[0][2] (j=0,k=2 -> index 2)
    V[6, 5] = -1 # B[2][2]
    # C elements: C[2][2] += M6, C[2][0] -= M6
    W[8, 5] = 1   # C[2][2]
    W[7, 5] = -1  # C[2][0]
    
    # Term 7: A02(B21-B11)
    U[2, 6] = 1  # A[0][2] (i=0,j=2 -> index 2)
    V[5, 6] = 1  # B[2][1] (j=2,k=1 -> index 5)
    V[3, 6] = -1 # B[1][1]
    # C elements: C[0][2] += M7, C[0][1] -= M7
    W[2, 6] = 1   # C[0][2]
    W[1, 6] = -1  # C[0][1]
    
    return U, V, W

def _construct_block_decomposition():
    """
    Construct a decomposition for 3x3x5 by embedding Strassen (rank 7) for first 3 cols
    and solving for additional terms for columns 4 and 5.
    """
    n, m, p = 3, 3, 5
    
    # Get Strassen decomposition for 3x3x3
    U_s, V_s, W_s = _strassen_3x3x3()
    
    # For the full 3x3x5 problem:
    # U: (9, r), V: (15, r), W: (15, r)
    # First 7 columns come from Strassen (padded to p=5)
    # We need to handle k=3,4 (columns 4,5)
    
    # Start with Strassen terms extended to p=5
    # For k=3,4, V entries are 0 in the first 7 terms
    U_base = np.zeros((9, 7))
    V_base = np.zeros((15, 7))
    W_base = np.zeros((15, 7))
    
    U_base[:, :] = U_s
    V_base[:9, :] = V_s  # First 9 rows (k=0,1,2)
    # V_base[9:15, :] remains 0 for k=3,4
    
    W_base[:9, :] = W_s  # First 9 entries (k*n+i for k=0,1,2)
    # W_base[9:15, :] remains 0 for k=3,4
    
    # Now we need to add terms to satisfy:
    # For all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # 
    # For k=0,1,2: Already satisfied by Strassen (need to verify)
    # For k=3,4: Need U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j
    
    # Let's try adding r_add terms. We'll use a simple approach:
    # Add terms that are separable in (i,j) for each k in {3,4}
    
    # Strategy: For each additional column k in {3,4}, we need:
    # sum_{r=r_base}^{r_base+r_add-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Simple approach: Use outer products
    # For k=3: Add terms of form U[i,j,r] = a[i,j], V[j,3,r] = b[j], W[3,i,r] = c[i]
    # such that sum_r a[i,j]*b[j]*c[i] = 1
    
    # We can use 3 terms for each extra column (since 3x3 needs rank 3 for identity-like)
    # Actually, we need: for fixed k, the tensor T[i,j] = 1 must be decomposed as
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # Let's use a direct approach: try different numbers of additional terms
    for r_add in range(3, 20):
        # Total rank = 7 + r_add
        r_total = 7 + r_add
        
        # Build the linear system
        # Unknowns: U[9, r_add], V[15, r_add], W[15, r_add]
        # But we can simplify: for k=3,4, only V[:, 3:5, :] and W[3:5, :, :] matter
        # And U is shared
        
        # Let's parameterize:
        # U_extra: (9, r_add)
        # V_extra: (15, r_add) but only rows 9-14 (k=3,4) are non-zero
        # W_extra: (15, r_add) but only rows 9-14 (k=3,4) are non-zero
        
        # Actually, let's be more careful. The full decomposition is:
        # T[i,j,k] = sum_{r=0}^{6} U_s[i,j]*V_s[j,k]*W_s[k,i] + sum_{r=7}^{r_total-1} U_e[i,j,r-7]*V_e[j,k,r-7]*W_e[k,i,r-7]
        
        # For k=0,1,2: V_e[j,k,r] = 0, so the second sum is 0. Good.
        # For k=3,4: V_s[j,k,r] = 0 (since Strassen only has k=0,1,2), so:
        # T[i,j,k] = sum_{r=7}^{r_total-1} U_e[i,j,r-7]*V_e[j,k,r-7]*W_e[k,i,r-7] = 1
        
        # So we need to find U_e (9 x r_add), V_e (15 x r_add), W_e (15 x r_add)
        # such that for k in {3,4}: sum_{s=0}^{r_add-1} U_e[i,j,s]*V_e[j,k,s]*W_e[k,i,s] = 1
        
        # This is still a nonlinear problem. Let's try a specific structure.
        
        # For each k in {3,4}, we need a rank-r_add decomposition of the all-ones 3x3 matrix
        # where the factors are coupled: V_e[j,k,s] and W_e[k,i,s] share the same s index.
        
        # Simple approach: use 3 terms per extra column, with:
        # U_e[i,j,s] = 1 for s=0,1,2 (same for all i,j)
        # V_e[j,k,s] = 1/3 for s=0,1,2
        # W_e[k,i,s] = 1 for s=0,1,2
        # Then sum_s (1)*(1/3)*(1) = 3*(1/3) = 1. 
        
        # But this uses 3 terms per column, total 6 additional terms, rank = 13.
        # Let's try this first.
        
        if r_add == 6:
            # Use 3 terms for k=3, 3 terms for k=4
            U_e = np.zeros((9, 6))
            V_e = np.zeros((15, 6))
            W_e = np.zeros((15, 6))
            
            # Terms 0-2 for k=3
            for s in range(3):
                U_e[:, s] = 1
                V_e[9, s] = 1/3  # j=0, k=3
                V_e[10, s] = 1/3 # j=1, k=3
                V_e[11, s] = 1/3 # j=2, k=3
                W_e[9, s] = 1    # k=3, i=0
                W_e[10, s] = 1   # k=3, i=1
                W_e[11, s] = 1   # k=3, i=2
            
            # Terms 3-5 for k=4
            for s in range(3):
                U_e[:, s+3] = 1
                V_e[12, s+3] = 1/3 # j=0, k=4
                V_e[13, s+3] = 1/3 # j=1, k=4
                V_e[14, s+3] = 1/3 # j=2, k=4
                W_e[12, s+3] = 1   # k=4, i=0
                W_e[13, s+3] = 1   # k=4, i=1
                W_e[14, s+3] = 1   # k=4, i=2
            
            # Combine
            U_full = np.hstack([U_base, U_e])
            V_full = np.hstack([V_base, V_e])
            W_full = np.hstack([W_base, W_e])
            
            # Verify
            if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        # Try other structures for different r_add values
        # For now, just try the simple 6-term approach
    
    # If we didn't find a valid decomposition, return None
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Provide the guaranteed schoolbook solution
    best_result = _schoolbook(n, m, p)
    
    # Try the block decomposition approach
    result = _construct_block_decomposition()
    if result is not None:
        best_result = result
    
    return best_result
