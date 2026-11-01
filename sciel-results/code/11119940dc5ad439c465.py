import time
import random
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,k,j
    
    deadline = time.time() + 145.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)
    
    def mat_to_list(mat):
        """Convert matrix to list of lists of Python floats/ints"""
        return mat.tolist()
    
    def try_targeted_rank_25():
        """
        Attempt to find a rank 25 decomposition using a structured heuristic.
        The 2x2x2 Strassen decomposition has rank 7.
        The 2x2x7 decomposition can be viewed as extending this.
        We will try to construct a rank 25 solution by leveraging the symmetry.
        """
        r = 25
        # Initialize with a Strassen-like pattern for the first few terms
        # Then use random perturbation or solving for the rest
        
        # Known Strassen 7 terms for 2x2x2 (indices i,k in {0,1}, j in {0,1})
        # We need to extend to j in {0..6}
        
        # Strategy: 
        # 1. Use 7 terms for j=0,1 (Strassen)
        # 2. Use 7 terms for j=2,3 (Strassen on B[:,2:4])
        # 3. Use 7 terms for j=4,5 (Strassen on B[:,4:6])
        # 4. Use 4 terms for j=6? 
        # Total 7+7+7+4 = 25.
        # This assumes we can reuse U factors? No, U depends on i,k.
        # If we reuse U factors, we save rank.
        
        # Let's try a random search specifically for rank 25 with a better initialization.
        # We will fix the first 7 terms to be Strassen for j=0,1.
        
        U_mat = np.zeros((n*m, r))
        V_mat = np.zeros((m*p, r))
        W_mat = np.zeros((p*n, r))
        
        # Strassen coefficients for 2x2x2
        # Terms c=0..6
        # U[i*m+k, c], V[k*p+j, c], W[j*n+i, c]
        
        # Strassen:
        # M0 = (A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11; W: C00,C11
        # M1 = (A10+A11)B00 -> U: A10,A11; V: B00; W: C10
        # M2 = A00(B10-B11) -> U: A00; V: B10,B11; W: C01
        # M3 = A11(B01-B00) -> U: A11; V: B01,B00; W: C11
        # M4 = (A00+A10)B11 -> U: A00,A10; V: B11; W: C01
        # M5 = (A10-A00)(B00+B01) -> U: A10,A00; V: B00,B01; W: C10,C11
        # M6 = (A01-A11)(B10+B11) -> U: A01,A11; V: B10,B11; W: C01,C11
        
        # Note: Strassen indices are usually (i,k) for A, (k,j) for B.
        # My indices: U[i*m+k], V[k*p+j], W[j*n+i].
        
        # Let's just use a random search for rank 25, but with a time limit.
        # If it fails, return None.
        
        half_ints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        
        for trial in range(200):
            if time.time() > deadline - 10:
                return None
            
            # Random U, V
            U_mat = half_ints[np.random.randint(0, len(half_ints), (n*m, r))]
            V_mat = half_ints[np.random.randint(0, len(half_ints), (m*p, r))]
            
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline - 5:
                        consistent = False
                        break
                    
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                    
                    b_row = np.ones(m)
                    
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                        if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                            consistent = False
                            break
                        W_mat[j*n + i, :] = sol
                    except Exception:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            if verify(U_mat, V_mat, W_mat):
                if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                    return {
                        "U": mat_to_list(U_mat),
                        "V": mat_to_list(V_mat),
                        "W": mat_to_list(W_mat)
                    }
        
        return None
    
    # Try rank 25 first (targeted)
    sol = try_targeted_rank_25()
    if sol is not None:
        return sol
    
    # Fallback to schoolbook
    return best
# EVOLVE-BLOCK-END
