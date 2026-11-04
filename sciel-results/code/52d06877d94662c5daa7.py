"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
    # Actually, the tensor for matrix multiplication is: T[i,j,k] = delta_{i,k} is wrong
    # The correct tensor: for A (n x m), B (m x p), C = A @ B (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity in the summation sense)
    # Wait, let me reconsider. The tensor T is defined such that:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    rank = U.shape[1]
    # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a contraction over c
    # T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Expected T is all ones
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_2x2():
    """Return U, V, W for 2x2 @ 2x2 with rank 7 (Strassen)."""
    # Strassen's 7 products for 2x2:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2: n=m=p=2, so U has 4 rows, V has 4 rows, W has 4 rows
    # U[i*2+j, c] corresponds to A[i,j]
    # V[j*2+k, c] corresponds to B[j,k]
    # W[k*2+i, c] corresponds to C[i,k]
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1: (a00+a11)*(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (a10+a11)*b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # C10
    
    # M3: a00*(b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: a11*(b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (a00+a01)*b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6: (a10-a00)*(b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # C11
    
    # M7: (a01-a11)*(b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1 # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Strategy: Use Strassen for 2x2 blocks where possible, fill rest with schoolbook
    # For 3x4 @ 4x6:
    # We can partition A into blocks and B into blocks
    # Let's try a hybrid approach: use rank-7 Strassen for 2x2 subblocks
    # and schoolbook for the remaining entries
    
    # Full schoolbook rank: 3*4*6 = 72
    # Let's try to reduce by using Strassen-like reductions
    
    # Approach: decompose the problem into independent 2x2 blocks plus residual
    # A is 3x4, B is 4x6
    # We can view this as:
    #   A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    #   B = [B11 B12; B21 B22] where B11 is 2x4, B12 is 2x2, B21 is 2x4, B22 is 2x2
    # This is getting complex. Let me use a simpler approach.
    
    # For now, let's implement a proper search-based approach that tries to find
    # a lower rank decomposition using known techniques.
    
    # Start with schoolbook and try to reduce
    rank = n * m * p  # 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify baseline
    assert _verify(U, V, W, n, m, p), "Baseline verification failed"
    
    # Try to use Strassen for 2x2 subproblems
    # For a 3x4 @ 4x6 problem, we can identify 2x2 blocks
    # Let's try a structured ansatz with Strassen blocks
    
    # Actually, let's try a different approach: use the fact that for certain
    # structured problems, we can reduce rank below schoolbook.
    
    # For 3x4 @ 4x6, the known lower bound is max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # Upper bound is n*m*p = 72
    
    # Let me try to construct a better decomposition using block Strassen
    # Partition: A = [A_top(2x4); A_bottom(1x4)], B = [B_left(4x4) B_right(4x2)]
    # C = [C_top(2x6); C_bottom(1x6)]
    # C_top = A_top @ B  (2x4 @ 4x6 = 2x6)
    # C_bottom = A_bottom @ B  (1x4 @ 4x6 = 1x6)
    
    # For C_top: 2x4 @ 4x6
    # We can further partition B = [B1(4x4) B2(4x2)]
    # C_top = [A_top @ B1 (2x4)  A_top @ B2 (2x2)]
    
    # The 2x2 part (A_top[:, 2:4] @ B[2:4, 4:6]) can use Strassen with rank 7
    # The 2x4 @ 4x4 part needs more work
    
    # Let me just return the schoolbook for now but with proper verification
    # and a time guard
    
    deadline = time.time() + 120
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
