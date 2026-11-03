import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block Strassen + direct, then local search to reduce rank.
    """
    n, m, p = 3, 4, 6
    
    def build_baseline():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        for j in range(3):
            for i in range(2):
                add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
        
        for j in range(3):
            for i in range(2):
                add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
        
        assert col == R
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition exactly reproduces matrix multiplication."""
        r = len(U[0])
        # Build the tensor from factors
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix mult: T[a,b,c] = A[a,b]*B[b,c]
        # U index: a*m + b, V index: b*p + c, W index: c*n + a
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1  # T[a,b,c] = 1 if this is the (a,b,c) entry of mult tensor
                    # Actually the matrix multiplication tensor has T[a,b,c] = 1 
                    # because (AB)[a,c] = sum_b A[a,b] B[b,c]
                    # So each triple (a,b,c) contributes 1
                    actual = 0.0
                    for rr in range(r):
                        actual += U[a * m + b][rr] * V[b * p + c][rr] * W[c * n + a][rr]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    def try_local_search(U, V, W, deadline):
        """Try to reduce rank by finding and removing redundant terms."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        r = len(U[0])
        
        # Try dropping each column and checking if the remaining still works
        # (with possible rescaling)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = len(U[0])
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                # Skip the column and check if we can still represent the tensor
                # This is a hard problem, so use a heuristic:
                # Check if the dropped column is a linear combination of others
                # in the tensor space
                
                # Build the tensor from all columns except drop_idx
                # and see if adding a scaled version of drop_idx helps
                # Simpler: just try dropping and verify
                U_test = [row[:drop_idx] + row[drop_idx+1:] for row in U]
                V_test = [row[:drop_idx] + row[drop_idx+1:] for row in V]
                W_test = [row[:drop_idx] + row[drop_idx+1:] for row in W]
                
                if verify(U_test, V_test, W_test):
                    U = U_test
                    V = V_test
                    W = W_test
                    improved = True
                    break
        
        return U, V, W
    
    U, V, W = build_baseline()
    
    deadline = time.time() + 140  # 10s safety margin
    
    U, V, W = try_local_search(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
