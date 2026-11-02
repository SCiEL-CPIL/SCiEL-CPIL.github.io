import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (14, rank)
    # W shape: (p*n, rank) = (14, rank)
    # W[k*n+i, c] convention: W[7*i+k, c] for C[i,k]
    
    deadline = time.time() + 145.0
    
    def build_rank25():
        """Guaranteed rank-25: Strassen for (0,1),(2,3),(4,5) + schoolbook for col 6."""
        rank = 25
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_strassen(col_start, k_start):
            # M1 = (A00+A11)*(B0,ks+B1,ks+1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1
            V[7*1+(k_start+1), col_start+0] += 1
            W[2*k_start+0, col_start+0] += 1
            W[2*(k_start+1)+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[2*k_start+1, col_start+1] += 1
            W[2*(k_start+1)+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1-B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+(k_start+1), col_start+2] += 1
            V[7*1+(k_start+1), col_start+2] -= 1
            W[2*(k_start+1)+0, col_start+2] += 1
            W[2*(k_start+1)+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks-B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1
            V[7*0+k_start, col_start+3] -= 1
            W[2*k_start+0, col_start+3] += 1
            W[2*k_start+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1+(k_start+1), col_start+4] += 1
            W[2*k_start+0, col_start+4] -= 1
            W[2*(k_start+1)+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1
            V[7*0+(k_start+1), col_start+5] += 1
            W[2*(k_start+1)+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1
            V[7*1+(k_start+1), col_start+6] += 1
            W[2*k_start+0, col_start+6] += 1
        
        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
        U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
        U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
        U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U_mat, V_mat, W_mat):
        """Verify: sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    def try_rank24():
        """
        Attempt rank-24 by starting from rank-25 and trying to merge one schoolbook
        term into the Strassen structure.
        
        Idea: For column 6, instead of 4 separate terms, can we express one of the
        A[i,k]*B[k,6] products as a linear combination of the existing Strassen
        multiplication results?
        
        The Strassen M values for columns (4,5) give us certain linear combinations.
        We need: C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6]
        
        If we can find a rank-1 update to U,V,W that handles column 6 with only 3
        additional terms instead of 4, we get rank 24.
        
        Alternative: Try to find a rank-24 decomposition by solving the linear system
        directly with random U,V and checking if W exists with half-integer entries.
        """
        r = 24
        max_trials = 200
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Random sparse U and V with entries in {-1, 0, 1}
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            # Ensure each row of U has at least 2 non-zeros for coverage
            for c in range(r):
                # Randomly choose which U positions are non-zero
                num_u = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                u_pos = np.random.choice(n*m, size=num_u, replace=False)
                U_mat[u_pos, c] = np.random.choice([-1, 1], size=num_u)
                
                num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                v_pos = np.random.choice(m*p, size=num_v, replace=False)
                V_mat[v_pos, c] = np.random.choice([-1, 1], size=num_v)
            
            # Solve for W: for each (j,i) pair:
            # sum_c U[i*m+k,c] * V[k*p+j,c] * W[j*n+i,c] = 1 for all k
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
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
                # Check half-integer
                if (np.allclose(U_mat * 2, np.round(U_mat * 2), atol=1e-8) and
                    np.allclose(V_mat * 2, np.round(V_mat * 2), atol=1e-8) and
                    np.allclose(W_mat * 2, np.round(W_mat * 2), atol=1e-8)):
                    return {
                        "U": np.round(U_mat * 2).astype(int).tolist(),
                        "V": np.round(V_mat * 2).astype(int).tolist(),
                        "W": np.round(W_mat * 2).astype(int).tolist()
                    }
        
        return None
    
    def try_rank24_structured():
        """
        Structured approach: take the rank-25 Strassen solution and try to
        eliminate one degree of freedom by finding a linear dependency.
        
        For the last column (col 6), we have 4 schoolbook terms. If we can
        express one of {A00*B06, A01*B16, A10*B06, A11*B16} as a combination
        of the Strassen M-terms from adjacent blocks, we save 1 rank.
        
        Actually, let's try a different structural approach: use a 2x4 Strassen-like
        block for columns 2-5 (rank 14) + Strassen for 0-1 (rank 7) + 3 terms for col 6.
        That gives 7+14+3 = 24 if we can find a 3-term decomposition for one column.
        
        For one column, C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6].
        This is a rank-4 tensor (2x2x1). Can it be done in rank 3?
        The tensor is T[i,k,j] = delta_{i,k} for j=0 (just one j).
        Actually for a single column, we need 4 terms minimum (it's a 2x2 identity-like).
        Wait - for a single column, C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j].
        This requires at least 4 rank-1 terms in general (the 2x2 matrix product for one column).
        Actually no - it's just 2 scalar multiplications per output element, 4 total.
        The tensor for one column has rank 4 (it's the 2x2 matrix multiplication tensor restricted).
        
        So rank 24 via this path seems impossible. Let's try another approach.
        
        What about using a different grouping: Strassen for (0,1), then a custom
        rank-14 for (2,3,4,5), then 3 for col 6? Total = 7+14+3 = 24.
        But rank-14 for 2x2*2x4 is the same as two Strassen blocks (7+7=14), so no gain.
        
        Let's try: can we do 2x2*2x7 in rank 24 by a non-block approach?
        """
        r = 24
        
        # Try with a bias toward the Strassen structure
        for trial in range(100):
            if time.time() > deadline:
                return None
            
            # Start with a perturbation of the known rank-25 structure
            # Remove one column and see if we can redistribute
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            # Use Strassen-like patterns for most columns
            for c in range(r):
                if c < 21:
                    # Strassen-like: 2 non-zeros in U, 2 in V
                    if c % 7 == 0:
                        U_mat[np.random.choice([0,3], size=2, replace=False), c] = 1
                    elif c % 7 == 1:
                        U_mat[np.random.choice([2,3], size=2, replace=False), c] = 1
                    elif c % 7 == 4:
                        U_mat[np.random.choice([0,1], size=2, replace=False), c] = 1
                    else:
                        U_mat[np.random.choice(n*m, size=np.random.choice([1,2]), replace=False), c] = np.random.choice([-1,1], size=1)
                    
                    V_mat[np.random.choice(m*p, size=np.random.choice([1,2,3]), replace=False), c] = np.random.choice([-1,1], size=1)
                else:
                    # Last 3 columns: for column 6
                    U_mat[np.random.choice(n*m, size=2, replace=False), c] = np.random.choice([-1,1], size=2)
                    V_mat[np.random.choice([7*0+6, 7*1+6], size=2, replace=False), c] = np.random.choice([-1,1], size=2)
            
            # Solve for W
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
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
                if (np.allclose(U_mat * 2, np.round(U_mat * 2), atol=1e-8) and
                    np.allclose(V_mat * 2, np.round(V_mat * 2), atol=1e-8) and
                    np.allclose(W_mat * 2, np.round(W_mat * 2), atol=1e-8)):
                    return {
                        "U": np.round(U_mat * 2).astype(int).tolist(),
                        "V": np.round(V_mat * 2).astype(int).tolist(),
                        "W": np.round(W_mat * 2).astype(int).tolist()
                    }
        
        return None
    
    def try_rank23():
        """Attempt rank-23 with more aggressive search."""
        r = 23
        max_trials = 50
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            for c in range(r):
                num_u = np.random.choice([1, 2], p=[0.4, 0.6])
                u_pos = np.random.choice(n*m, size=num_u, replace=False)
                U_mat[u_pos, c] = np.random.choice([-1, 1], size=num_u)
                
                num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                v_pos = np.random.choice(m*p, size=num_v, replace=False)
                V_mat[v_pos, c] = np.random.choice([-1, 1], size=num_v)
            
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
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
                if (np.allclose(U_mat * 2, np.round(U_mat * 2), atol=1e-8) and
                    np.allclose(V_mat * 2, np.round(V_mat * 2), atol=1e-8) and
                    np.allclose(W_mat * 2, np.round(W_mat * 2), atol=1e-8)):
                    return {
                        "U": np.round(U_mat * 2).astype(int).tolist(),
                        "V": np.round(V_mat * 2).astype(int).tolist(),
                        "W": np.round(W_mat * 2).astype(int).tolist()
                    }
        
        return None
    
    # First, get the guaranteed rank-25 solution
    best = build_rank25()
    
    # Try to improve: rank 24
    sol = try_rank24()
    if sol is not None:
        return sol
    
    sol = try_rank24_structured()
    if sol is not None:
        return sol
    
    # Try rank 23
    sol = try_rank23()
    if sol is not None:
        return sol
    
    return best
# EVOLVE-BLOCK-END
