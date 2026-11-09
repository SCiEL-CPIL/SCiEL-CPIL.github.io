"""Matrix multiplication 2x4x5 with Strassen-based rank reduction."""
import time
import sys

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x4 @ 4x5 matrix multiplication.
    
    Strategy: Split the 5 columns of the second matrix into groups:
    - First 4 columns: use Strassen (rank 7) for each 2x2 @ 2x2 block
    - Last 1 column: schoolbook (rank 8) for 2x4 @ 4x1
    
    Total rank: 2*7 + 8 = 22
    
    The matrix A is 2x4, B is 4x5.
    A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    B has 4x5 entries.
    
    We split B's columns into:
    - B1: first 2 columns (for Strassen block 1)
    - B2: next 2 columns (for Strassen block 2)
    - B3: last 1 column (schoolbook)
    
    For Strassen on 2x2 @ 2x2:
    A_block = [[a00, a01], [a10, a11]] (2x2)
    B_block = [[b00, b01], [b10, b11]] (2x2)
    
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)(b00)
    M3 = (a00)(b01-b11)
    M4 = (a11)(b10-b00)
    M5 = (a00+a01)(b11)
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mt is a rank-1 term in the tensor decomposition.
    """
    n, m, p = 2, 4, 5
    rank = 2 * 7 + 8  # 22
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 22
    V = [[0] * rank for _ in range(m * p)]  # 20 x 22
    W = [[0] * rank for _ in range(p * n)]  # 10 x 22
    
    # Helper function to add a Strassen block
    # A_block indices: rows [r0, r0+1], cols [c0, c0+1] in A (2x4)
    # B_block indices: rows [r0, r0+1], cols [c0, c0+1] in B (4x5)
    # C_block indices: rows [r0, r0+1], cols [c0, c0+1] in C (2x5)
    def add_strassen_block(r0, c0, start_col):
        """
        Add 7 rank-1 terms for Strassen decomposition of A[r0:r0+2, c0:c0+2] @ B[r0:r0+2, c0:c0+2]
        
        A indices: (0,c0), (0,c0+1), (1,c0), (1,c0+1) -> U row indices: 0*m+c0, 0*m+c0+1, 1*m+c0, 1*m+c0+1
        B indices: (r0,c0), (r0,c0+1), (r0+1,c0), (r0+1,c0+1) -> V row indices: r0*p+c0, r0*p+c0+1, (r0+1)*p+c0, (r0+1)*p+c0+1
        C indices: (0,c0), (0,c0+1), (1,c0), (1,c0+1) -> W row indices: c0*n+0, c0*n+1, (c0+1)*n+0, (c0+1)*n+1
        """
        # A element indices in U (row = i*m + j)
        a00_idx = 0 * m + c0      # a00
        a01_idx = 0 * m + (c0+1)  # a01
        a10_idx = 1 * m + c0      # a10
        a11_idx = 1 * m + (c0+1)  # a11
        
        # B element indices in V (row = j*p + k)
        b00_idx = r0 * p + c0      # b00
        b01_idx = r0 * p + (c0+1)  # b01
        b10_idx = (r0+1) * p + c0      # b10
        b11_idx = (r0+1) * p + (c0+1)  # b11
        
        # C element indices in W (row = k*n + i)
        c00_idx = c0 * n + 0      # c00
        c01_idx = c0 * n + 1      # c01
        c10_idx = (c0+1) * n + 0  # c10
        c11_idx = (c0+1) * n + 1  # c11
        
        # M1 = (a00+a11)(b00+b11)
        col = start_col
        U[a00_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (a10+a11)(b00)
        U[a10_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = (a00)(b01-b11)
        U[a00_idx][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = (a11)(b10-b00)
        U[a11_idx][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a00+a01)(b11)
        U[a00_idx][col] += 1
        U[a01_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_idx][col] += 1
        U[a00_idx][col] -= 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        return start_col + 7
    
    # Add Strassen block for columns 0,1 of A and rows 0,1 of B
    # A cols 0,1; B rows 0,1; C cols 0,1
    next_col = add_strassen_block(0, 0, 0)
    
    # Add Strassen block for columns 2,3 of A and rows 0,1 of B
    # A cols 2,3; B rows 0,1; C cols 2,3
    next_col = add_strassen_block(0, 2, next_col)
    
    # Now handle the remaining part: A @ B where B has rows 2,3 and all 5 columns
    # This is 2x4 @ 4x5 with only 2 rows in B (rows 2,3)
    # We can use schoolbook for this: 2*4*2 = 16 terms, but we can do better
    # Actually, let's think about this differently.
    
    # Alternative approach: Split B into two parts:
    # B_top: rows 0,1 (2x5) -> handled by Strassen blocks above
    # B_bottom: rows 2,3 (2x5) -> need to handle this
    
    # For B_bottom (rows 2,3), we have A (2x4) @ B_bottom (2x5)
    # This is equivalent to 4x5 @ 5x2 after reshaping, or we can just do schoolbook
    # Schoolbook: 2*4*2 = 16 terms
    
    # But wait, we can be smarter. Let's use the fact that B_bottom is 2x5.
    # A (2x4) @ B_bottom (2x5) = sum over j=0..3 of A[:,j] * B_bottom[j,:]
    # Each term A[:,j] * B_bottom[j,:] is an outer product of (2x1) and (1x5) = 2x5 matrix
    # In tensor form, for each j, we have 5 rank-1 terms (one for each column of C)
    # Total: 4 * 5 = 20 terms... that's worse.
    
    # Actually, for schoolbook: for each i in {0,1}, j in {2,3}, k in {0..4}
    # C[i,k] += A[i,j] * B[j,k]
    # That's 2*2*5 = 20 terms. Still not great.
    
    # Let me reconsider the whole approach.
    
    # Better strategy:
    # Split the 4 rows of B into two groups of 2.
    # For each group, use a decomposition for 2x4 @ 2x5.
    
    # 2x4 @ 2x5: 
    # A (2x4) @ B (2x5)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # For each j, A[i,j] * B[j,k] is a rank-1 term in the (i,k) plane for fixed j.
    # But we can group them.
    
    # Actually, 2x4 @ 2x5 has rank at most 2*4*2 = 16 (schoolbook).
    # Can we do better?
    
    # Let's just use schoolbook for the bottom part: 2*2*5 = 20 terms.
    # Total: 14 + 20 = 34. That's better than 40.
    
    # Actually, let me reconsider. The top part (rows 0,1 of B) with Strassen gives 14.
    # For the bottom part (rows 2,3 of B), schoolbook gives 2*2*5 = 20.
    # Total: 34.
    
    # But we can do better for the bottom part.
    # A (2x4) @ B_bottom (2x5) where B_bottom has rows indexed by j in {0,1} (local)
    # C[i,k] = sum_{j=0}^{1} A[i,2+j] * B_bottom[j,k]
    
    # This is 2*2*5 = 20 schoolbook terms.
    
    # Alternatively, we can view it as:
    # For each j in {0,1} (local), A[:,2+j] is a 2-vector, B_bottom[j,:] is a 5-vector
    # Their outer product is 2x5.
    # So we have 2 outer products, each contributing to all 10 entries of C.
    # In tensor form, each outer product gives 5 rank-1 terms (one per column of C).
    # Total: 2 * 5 = 10 terms.
    
    # Wait, that's not right. Let me think more carefully.
    
    # The tensor T[i,j,k] = A[i,j] * B[j,k] for the full problem.
    # For the bottom part, j is in {2,3} (global), which is {0,1} (local).
    # T[i, j_local, k] = A[i, 2+j_local] * B_bottom[j_local, k]
    
    # This tensor has dimensions 2 x 2 x 5.
    # Its rank is at most 2*2 = 4? No, that's not right either.
    
    # The rank of the 2x2x5 tensor T[i,j,k] = A[i,2+j] * B_bottom[j,k]:
    # For fixed i, T[i,j,k] is a 2x5 matrix with entries A[i,2+j]*B_bottom[j,k].
    # This matrix has rank at most 2 (since it's a sum of 2 rank-1 matrices).
    # Similarly for other unfoldings.
    
    # The tensor rank of T[i,j,k] = a_i * b_j where a_i depends on i and j, b_j depends on j and k...
    # Actually, T[i,j,k] = A[i,2+j] * B_bottom[j,k].
    # For fixed j, T[i,j,k] as a function of i,k is A[i,2+j] * B_bottom[j,k], which is rank 1 in (i,k).
    # So T = sum_{j=0}^{1} T[:,:,j] where each T[:,:,j] is rank 1 in (i,k).
    # But tensor rank considers all three modes.
    
    # The tensor T[i,j,k] = A[i,2+j] * B_bottom[j,k] can be written as:
    # T = sum_{j=0}^{1} (column j of A_bottom) outer (row j of B_bottom)
    # where A_bottom = A[:,2:4] (2x2) and B_bottom is 2x5.
    
    # Each term (column j of A_bottom) outer (row j of B_bottom) is a 2x2x5 tensor
    # that is rank 1 in the (i,k) modes but varies in j.
    
    # Actually, let's just use schoolbook: 20 terms.
    # Total: 14 + 20 = 34.
    
    # Hmm, but the known result from the observation was 36. Let me check if I can do 34.
    
    # Let me add the schoolbook terms for the bottom part.
    
    # For j in {2,3} (global), k in {0..4}:
    # C[i,k] += A[i,j] * B[j,k]
    # In tensor form: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    for j in range(2, 4):
        for k in range(5):
            for i in range(2):
                col = next_col
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                next_col += 1
    
    assert next_col == rank, f"Expected rank {rank}, got {next_col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


def _verify_decomposition(U, V, W, n=2, m=4, p=5, test_cases=100):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    import random
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(test_cases):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the tensor decomposition
        # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # But we need to sum over j as well? No, the tensor is T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # And C[i,k] = sum_j T[i,j,k]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_computed[i, k] += val
        
        if not np.allclose(C_expected, C_computed, atol=1e-9):
            print(f"Verification failed!")
            print(f"A = {A}")
            print(f"B = {B}")
            print(f"C_expected = {C_expected}")
            print(f"C_computed = {C_computed}")
            return False
    
    return True


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W']):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        sys.exit(1)
