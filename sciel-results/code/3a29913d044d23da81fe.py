"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Start with the known rank-53 solution as fallback
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0 (a)
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1 (b)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (c)
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1 (d)
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0 (e)
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (f)
        [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0 (g)
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1 (h)
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1 (C11)
    ]
    
    # Schoolbook coefficients for M_{2,2,1}
    U_school = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    
    V_school = [
        [1, 0, 1, 0], # j=0 (e)
        [0, 1, 0, 1]  # j=1 (g)
    ]
    
    W_school = [
        [1, 1, 0, 0], # i=0 (C0)
        [0, 0, 1, 1]  # i=1 (C1)
    ]
    
    # Fill global matrices for the rank-53 fallback
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    # Attempt to find a lower-rank solution via global linear algebra search
    # We look for rank r < 53 by solving the nonlinear system
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 if i==k else 0 (for the multiplication tensor)
    
    # Strategy: Use 6 Strassen blocks (rank 42) + search for M_{2,2,3} with rank < 11
    # The last 3 columns (k=12,13,14) need to be handled
    
    # Try to find a rank-10 decomposition for M_{2,2,3}
    # If found, total rank = 42 + 10 = 52
    
    def verify_decomposition(U_test, V_test, W_test, tol=1e-9):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U_test[0])):
                        s += U_test[i*m+j][r] * V_test[j*p+k][r] * W_test[k*n+i][r]
                    expected = 1.0 if i == k else 0.0
                    if abs(s - expected) > tol:
                        return False
        return True
    
    # Try rank 52: 6 Strassen blocks + rank-10 for M_{2,2,3}
    # First, let's see if we can find a rank-10 decomposition for M_{2,2,3}
    # by using the known rank-7 for M_{2,2,2} and trying to extend it
    
    # For M_{2,2,3}, we need to handle k=0,1,2
    # Use Strassen for k=0,1 (rank 7) and try to find rank-3 for the remaining k=2
    # But that's not quite right - we need a joint decomposition
    
    # Alternative: Try to find a rank-10 decomposition for M_{2,2,3} directly
    # by solving the system numerically
    
    # Build the tensor for M_{2,2,3}
    # T[i,j,k] = 1 if i==k else 0 for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Note: for matrix multiplication A(2x2) * B(2x3) = C(2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{i,k} for the multiplication
    
    # Let's try a different approach: use the known rank-7 for M_{2,2,2}
    # and see if we can add 3 more terms to handle the third column
    
    # Actually, let's try to find a rank-10 decomposition for M_{2,2,3}
    # by using a structured ansatz
    
    # For now, let's just try to verify if rank 52 is achievable
    # by checking if there's a rank-10 decomposition for M_{2,2,3}
    
    # We'll use a simple search: try random initializations and solve
    # the nonlinear system using least squares
    
    if time.time() < deadline - 10:
        # Try to find rank-10 for M_{2,2,3}
        # U: 4x10, V: 6x10, W: 6x10
        # T[i,j,k] = delta_{i,k} for i in {0,1}, j in {0,1}, k in {0,1,2}
        
        # Build the constraint matrix
        # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,k}
        
        # This is a system of 12 equations (2*2*3) with 30 unknowns (10*3)
        # But it's nonlinear
        
        # Let's try to fix U and V, then solve for W
        # Or use a different strategy
        
        # For now, let's just return the rank-53 solution
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
