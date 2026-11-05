"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return rank-7 decomposition of M_{2,2,2} (Strassen)."""
    # Strassen's 7 terms for 2x2 * 2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # Term 2: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # Term 3: A11(B00-B01)
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = -1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return schoolbook decomposition of M_{n,m,p}."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two decompositions.
    M_{n1,m1,p1} rank r1, M_{n2,m2,p2} rank r2
    -> M_{n1*n2, m1*m2, p1*p2} rank r1*r2
    """
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*n2+i2)*m + (j1*m2+j2) = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
            #      = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                            val1 = U1[i1*m1+j1][c1]
                            val2 = U2[i2*m2+j2][c2]
                            if val1 != 0 and val2 != 0:
                                U[idx_u][c] += val1 * val2
            
            # V: index (j1*m2+j2)*p + (k1*p2+k2) = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
            #      = j1*m2*p1*p2 + j2*p1*p2 + k1*p2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx_v = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
                            val1 = V1[j1*p1+k1][c1]
                            val2 = V2[j2*p2+k2][c2]
                            if val1 != 0 and val2 != 0:
                                V[idx_v][c] += val1 * val2
            
            # W: index (k1*p2+k2)*n + (i1*n2+i2) = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
            #      = k1*p2*n1*n2 + k2*n1*n2 + i1*n2 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx_w = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                            val1 = W1[k1*n1+i1][c1]
                            val2 = W2[k2*n2+i2][c2]
                            if val1 != 0 and val2 != 0:
                                W[idx_w][c] += val1 * val2
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(W[0]) if W else 0
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) pattern
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal 1 if this is the multiplication tensor, 0 otherwise
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the standard basis)
    # Wait, no. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    
    # Let me verify by checking that the decomposition reproduces the tensor
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False, (i, j, k, s)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook_nm_p(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 84
    
    # Try Kronecker product approach
    # M_{3,4,7} = M_{3,2,7} ⊗ M_{1,2,1}
    # M_{1,2,1} has rank 2
    # So we need M_{3,2,7}
    
    # M_{3,2,7}: 3*2*7 = 42 schoolbook
    # Can we do better?
    # M_{3,2,7} = M_{2,2,4} ⊗ M_{1,2,3/2}... no
    # M_{3,2,7} = M_{3,2,4} ⊕ M_{3,2,3} (block diagonal in output)
    # M_{3,2,4} rank 24, M_{3,2,3} rank 18, total 42 (same as schoolbook)
    
    # Alternative: M_{3,4,7} using M_{2,4,4} (rank 10)
    # Let's construct M_{2,4,4} first
    # M_{2,4,4}: multiply 2x4 by 4x4
    # Use Kronecker: M_{2,2,2} ⊗ M_{1,2,2}
    # M_{1,2,2} has rank 4 (schoolbook)
    # M_{2,2,2} has rank 7
    # M_{2,2,2} ⊗ M_{1,2,2} gives M_{2,4,4} with rank 28
    
    # That's not 10. Let me try differently.
    
    # For M_{2,4,4}, we can use a direct rank-10 decomposition
    # This is a known result in tensor decomposition literature
    
    # Let me try: M_{2,4,4} = M_{2,2,2} ⊗ M_{1,2,2} gives rank 28
    # But we want rank 10, so we need a different approach
    
    # Actually, let me just use schoolbook for the blocks and see if we can do better
    
    # Block decomposition of M_{3,4,7}:
    # Split p=7 into 4+3
    # Block A: M_{3,4,4} for cols 0-3
    # Block B: M_{3,4,3} for cols 4-6
    # M_{3,4,4} rank 48, M_{3,4,3} rank 36, total 84 (same as schoolbook)
    
    # Need to find overlapping structure
    
    # Let me try: M_{3,4,7} where 4=2*2
    # Use M_{3,2,7} ⊗ M_{1,2,1}
    # M_{1,2,1} rank 2
    # M_{3,2,7} rank ?
    
    # M_{3,2,7}: 
    # Split n=3 as 2+1
    # M_{2,2,7} ⊕ M_{1,2,7}
    # M_{2,2,7} rank 28, M_{1,2,7} rank 14, total 42
    
    # M_{2,2,7} = M_{2,2,4} ⊕ M_{2,2,3}
    # M_{2,2,4} rank 16, M_{2,2,3} rank 12, total 28
    
    # So M_{3,2,7} rank 42, M_{1,2,1} rank 2
    # Kronecker: M_{3,4,7} rank 84 (same as schoolbook)
    
    # This approach isn't helping. Let me try a different factorization.
    
    # M_{3,4,7} = M_{2,2,2} ⊗ M_{?}
    # 3 = 2*?, 4 = 2*?, 7 = 2*?
    # 3/2, 4/2=2, 7/2 - not integers
    
    # Let me try: use M_{2,2,2} (rank 7) for a sub-block and schoolbook for the rest
    
    # Block decomposition:
    # Rows 0-1, middle 0-1, cols 0-1: M_{2,2,2} rank 7
    # Remaining: schoolbook for the rest
    
    # This is getting complex. Let me just return the schoolbook for now
    # and note that a better decomposition requires more sophisticated construction
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
