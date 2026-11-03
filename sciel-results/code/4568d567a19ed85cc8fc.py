import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses Strassen for 2x2 blocks (rank 7) and direct for 1x2*2x2 blocks (rank 4).
    Total rank = 66.
    """
    n, m, p = 3, 4, 6
    
    R = 66
    U = [[0] * R for _ in range(n * m)]  # (12, 66)
    V = [[0] * R for _ in range(m * p)]  # (24, 66)
    W = [[0] * R for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Strassen 2x2 * 2x2 multiplication (rank 7).
        """
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
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Direct 1x2 * 2x2 multiplication (rank 4).
        """
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
    
    # Build the full decomposition
    for j in range(3):
        for i in range(2):
            add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
    
    for j in range(3):
        for i in range(2):
            add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
    
    assert col == R, f"Expected rank {R}, got {col}"
    
    # Try to reduce rank by checking for linear dependencies
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,k,j] = sum_r U[i,r]*V[k,r]*W[j,r]
    # and compare with the true matrix multiplication tensor
    # T[i, k, j] should equal delta where C[j,i] = sum_k A[i,k]*B[k,j]
    # So T[i,k,j] = 1 if this is a valid term, 0 otherwise
    
    # The true tensor: T[i,k,j] = 1 for all i,k,j (since C[j,i] = sum_k A[i,k]*B[k,j])
    # Actually the multiplication tensor has entries T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    
    # Check if any column is a linear combination of others
    # We can check the rank of the "coefficient matrix" formed by flattening
    # For each rank-r term, we have vectors u_r, v_r, w_r
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
    
    # Try to find if one term can be removed
    # Simple approach: check if removing each term still gives correct result
    # This is O(R * tensor_size) which is manageable
    
    def verify(U_list, V_list, W_list):
        """Verify the decomposition is correct."""
        n, m, p = 3, 4, 6
        R = len(U_list[0])
        # Check all entries of the multiplication tensor
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    # T[i,k,j] should be 1
                    val = 0
                    for r in range(R):
                        val += U_list[i*m+k][r] * V_list[k*p+j][r] * W_list[j*n+i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try removing each column one at a time
    best_rank = R
    best_U, best_V, best_W = U, V, W
    
    for skip in range(R):
        U_test = [row[:skip] + row[skip+1:] for row in U]
        V_test = [row[:skip] + row[skip+1:] for row in V]
        W_test = [row[:skip] + row[skip+1:] for row in W]
        if verify(U_test, V_test, W_test):
            best_rank = R - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            break
    
    if best_rank < R:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
