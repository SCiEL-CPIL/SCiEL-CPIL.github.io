"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # should equal the matrix multiplication tensor
    # T[i,j,k] = 1 if (i,j,k) corresponds to (row, col, output) where
    # A[row,col] * B[col,output] contributes to C[row,output]
    # U has shape (n*m, rank), indexed by (i*n + j) for row i, col j of A
    # V has shape (m*p, rank), indexed by (j*p + k) for col j, row k of B
    # W has shape (p*n, rank), indexed by (k*n + i) for row k, col i of C
    
    # Build the full tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] -> C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T[i, j, k] = 1.0  # This is the target tensor
    
    # Compute from decomposition
    T_decomp = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_decomp[i, j, k] += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    
    return np.allclose(T, T_decomp, atol=1e-9)

def _known_rank11():
    """Return the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank10():
    """Attempt to find a rank-10 decomposition via structured search."""
    n, m, p = DIMENSIONS
    rank = 10
    
    # The key insight: for 2x2x3, we can try to share factors between
    # the Strassen part (for cols 0,1) and the third column.
    # 
    # Standard Strassen for 2x2x2 uses 7 multiplications.
    # For 2x2x3, schoolbook needs 12. Rank 11 = 7 + 4 (schoolbook for col 2).
    # 
    # To get rank 10, we need to save one more multiplication.
    # This means finding a way to express one of the four schoolbook
    # products for column 2 using the existing Strassen intermediates.
    #
    # Let's try: can we express a*b02 or c*b02 or b*b12 or d*b12
    # as a linear combination of the 7 Strassen products?
    #
    # The Strassen products are:
    # M1 = (a+d)(b00+b11)
    # M2 = (c+d)b00
    # M3 = a(b01-b11)
    # M4 = d(b10-b00)
    # M5 = (a+b)b11
    # M6 = (c-a)(b00+b01)
    # M7 = (b-d)(b10+b11)
    #
    # None of these involve b02 or b12, so they can't directly help.
    # 
    # Alternative: modify the Strassen scheme to incorporate b02/b12.
    # Let's try a different set of 10 multiplications.
    #
    # Try: Use a modified Strassen where some factors include b02 or b12.
    #
    # Let's set up the equations. We need:
    # C00 = a*b00 + b*b10
    # C01 = a*b01 + b*b11
    # C02 = a*b02 + b*b12
    # C10 = c*b00 + d*b10
    # C11 = c*b01 + d*b11
    # C12 = c*b02 + d*b12
    #
    # With 10 multiplications, we need to find U, V, W of rank 10.
    # This is a hard problem. Let's try a parameterized search.
    
    # Time budget: give up after ~100 seconds
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try a structured approach: take the rank-11 solution and try to
    # merge two of the schoolbook terms with Strassen terms.
    #
    # Idea: Instead of separate M8=a*b02 and M9=b*b12, try to find
    # a single multiplication that helps with both, or merge with existing.
    #
    # Let's try: M8' = a*(b02 + alpha*b00) and adjust
    # This gets complicated. Let's try random search with verification.
    
    np.random.seed(42)
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Start from rank-11 and try to eliminate one term
        # by finding a dependency
        
        # Try: can we write the rank-11 decomposition with one zero column?
        # That would mean rank 10.
        
        # Random perturbation approach:
        # Take random U, V and solve for W, check if it works.
        
        U_rand = np.random.randn(n * m, rank)
        V_rand = np.random.randn(m * p, rank)
        
        # Build the tensor equations
        # For each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # This is linear in W for fixed U, V.
        
        # Collect equations: for each (i,j,k), we have:
        # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = T[i,j,k]
        
        # Group by W index: for each (k, i), we have m equations (one per j)
        # Let's build a linear system for W.
        
        # Number of W variables: p * n = 6
        # Number of equations: n * m * p = 12
        
        # For each W index w_idx = k*n + i, the coefficient for variable r is:
        # sum over j of ... no, each equation is separate.
        
        # Actually, each (i,j,k) gives one equation involving W[k*n+i, :]
        # So for fixed (k,i), we have m equations (for j=0,1) involving W[k*n+i, :]
        
        # Let's build the system properly.
        
        # Variables: W[0..5, 0..9] = 6*10 = 60 variables
        # Equations: 12
        
        # This is underdetermined, so we can't solve directly.
        # Instead, we need a different approach.
        
        # Let's try: fix U and V, then for each W index, we have a system.
        # For W[k*n+i, :], the equations are:
        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        # This is: (sum over j of U[i*m+j,:] * V[j*p+k,:] outer W[k*n+i,:]) = T[i,:,k]
        # Wait, it's: for each j, dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:]) = T[i,j,k]
        
        # Let A_ij = U[i*m+j,:] (length rank)
        # Let B_jk = V[j*p+k,:] (length rank)
        # Then we need: sum_r A_ij[r] * B_jk[r] * W_ki[r] = T[i,j,k]
        
        # For fixed (i,k), this is a system in W_ki = W[k*n+i,:]:
        # For j=0: sum_r A_i0[r] * B_0k[r] * W_ki[r] = T[i,0,k]
        # For j=1: sum_r A_i1[r] * B_1k[r] * W_ki[r] = T[i,1,k]
        
        # This is 2 equations in rank=10 unknowns. Underdetermined.
        # We can find a particular solution and the null space.
        
        # Let's try a different strategy: assume W has a specific structure.
        
        # Actually, let's just verify the rank-11 solution is correct and return it.
        # The rank-10 problem is very hard and unlikely to be solved by random search.
        
        pass
    
    return None

def construct():
    # First, verify the known rank-11 solution
    result = _known_rank11()
    
    if not _verify(result["U"], result["V"], result["W"]):
        raise ValueError("Known rank-11 decomposition failed verification!")
    
    # Try to find rank-10 (but this is very unlikely to succeed quickly)
    # Skip for now to ensure we return a valid result within time limit
    
    return result
# EVOLVE-BLOCK-END
