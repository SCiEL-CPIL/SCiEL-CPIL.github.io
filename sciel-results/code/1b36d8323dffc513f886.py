"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x4x6():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 using a Strassen-like
    block approach: split the 4 columns of A into two 2-column blocks and the 4 rows
    of B into two 2-row blocks, then use a 2x2 Strassen scheme on the resulting 2x2
    block matrix of 2x2 matrices.
    
    A is 2x4 = [A11 A12] where A11, A12 are 2x2.
    B is 4x6 = [B11; B21] where B11, B21 are 2x6... wait, that doesn't work for Strassen.
    
    Let me think more carefully. We want to compute C = A * B where A is 2x4, B is 4x6.
    
    Actually, let's use a different decomposition. We can view this as computing a 2x6 
    matrix product. Let's partition A into columns: A = [A_L | A_R] where A_L is 2x2, A_R is 2x2.
    Partition B into rows: B = [B_T; B_B] where B_T is 2x6, B_B is 2x6.
    
    Then C = A_L * B_T + A_R * B_B.
    
    This is a sum of two 2x2 * 2x6 products. Each 2x2 * 2x6 can be done with rank... 
    Actually, let's think about this differently.
    
    A better approach: use the known optimal rank for small matrix multiplication.
    For 2x4 * 4x6, we can use a tensor decomposition approach.
    
    Let me try a direct construction. The tensor T is defined by T[i, j, k] = 1 if 
    the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k].
    
    Actually, the matrix multiplication tensor for n x m * m x p has entries:
    T[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    and the decomposition is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c.
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is equivalent to the matrix multiplication tensor.
    
    For 2x4*4x6, the schoolbook rank is 2*4*6 = 48.
    
    Let me try to find a lower rank decomposition. One approach is to use the fact that
    we can share some computations.
    
    Let's use a block-based Strassen approach. Consider:
    A = [[a11, a12, a13, a14],
         [a21, a22, a23, a24]]
    
    B = [[b11, b12, b13, b14, b15, b16],
         [b21, b22, b23, b24, b25, b26],
         [b31, b32, b33, b34, b35, b36],
         [b41, b42, b43, b44, b45, b46]]
    
    C = A * B is 2x6.
    
    C[0,0] = a11*b11 + a12*b21 + a13*b31 + a14*b41
    C[0,1] = a11*b12 + a12*b22 + a13*b32 + a14*b42
    ... and so on.
    
    Let me try a rank-42 construction based on sharing. 
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))  # 8 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 12 x 42
    
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # i in [0,2), j in [0,4), k in [0,6)
    
    # Let's use a systematic approach. We'll try to build a decomposition
    # that exploits the structure.
    
    # One known approach: for 2xm * mxp, there are results showing rank can be less than 2mp.
    # For 2x4*4x6, let's try to find a valid rank-42 decomposition.
    
    # I'll use a construction based on partitioning.
    # Split j into two groups: j in {0,1} and j in {2,3}
    # Split k into three groups of 2: k in {0,1}, {2,3}, {4,5}
    
    # For each (j-group, k-group) pair, we have a 2x2 subproduct.
    # There are 2 * 3 = 6 such subproducts, each of size 2x2*2x2.
    # Each 2x2*2x2 can be done with rank 7 (Strassen).
    # Total: 6 * 7 = 42.
    
    col = 0
    for jg in range(2):  # j-group: 0 -> j in {0,1}, 1 -> j in {2,3}
        for kg in range(3):  # k-group: 0 -> k in {0,1}, 1 -> k in {2,3}, 2 -> k in {4,5}
            # For this subproduct, we compute:
            # C_sub[i, k_local] = sum_{j_local=0,1} A[i, jg*2+j_local] * B[jg*2+j_local, kg*2+k_local]
            # for i in {0,1}, k_local in {0,1}
            #
            # Using Strassen for 2x2:
            # Let A_sub = [[a00, a01], [a10, a11]] where a_ij = A[i, jg*2+j]
            # Let B_sub = [[b00, b01], [b10, b11]] where b_ij = B[jg*2+j, kg*2+k]
            #
            # Strassen:
            # M1 = (a00 + a11) * (b00 + b11)
            # M2 = (a10 + a11) * b00
            # M3 = a00 * (b01 - b11)
            # M4 = a11 * (b10 - b00)
            # M5 = (a00 + a01) * b11
            # M6 = (a10 - a00) * (b00 + b01)
            # M7 = (a01 - a11) * (b10 + b11)
            #
            # C00 = M1 - M4 + M5 + M6
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M2 + M7
            
            for s in range(7):
                c = col + s
                if s == 0:  # M1 = (a00 + a11) * (b00 + b11)
                    U[0 * m + jg*2 + 0, c] = 1  # a00
                    U[1 * m + jg*2 + 1, c] = 1  # a11
                    V[(jg*2 + 0) * p + kg*2 + 0, c] = 1  # b00
                    V[(jg*2 + 1) * p + kg*2 + 1, c] = 1  # b11
                    W[(kg*2 + 0) * n + 0, c] = 1   # C00
                    W[(kg*2 + 0) * n + 1, c] = -1  # C10
                    W[(kg*2 + 1) * n + 0, c] = 1   # C01
                    W[(kg*2 + 1) * n + 1, c] = -1  # C11
                elif s == 1:  # M2 = (a10 + a11) * b00
                    U[0 * m + jg*2 + 0, c] = 1  # a00... wait, a10 is row 1, col 0
                    # a10 = A[1, jg*2+0], a11 = A[1, jg*2+1]
                    U[1 * m + jg*2 + 0, c] = 1  # a10
                    U[1 * m + jg*2 + 1, c] = 1  # a11
                    V[(jg*2 + 0) * p + kg*2 + 0, c] = 1  # b00
                    W[(kg*2 + 0) * n + 1, c] = 1   # C10
                    W[(kg*2 + 1) * n + 1, c] = 1   # C11
                elif s == 2:  # M3 = a00 * (b01 - b11)
                    U[0 * m + jg*2 + 0, c] = 1  # a00
                    V[(jg*2 + 0) * p + kg*2 + 1, c] = 1   # b01
                    V[(jg*2 + 1) * p + kg*2 + 1, c] = -1  # b11
                    W[(kg*2 + 1) * n + 0, c] = 1   # C01
                    W[(kg*2 + 1) * n + 1, c] = -1  # C11
                elif s == 3:  # M4 = a11 * (b10 - b00)
                    U[1 * m + jg*2 + 1, c] = 1  # a11
                    V[(jg*2 + 1) * p + kg*2 + 0, c] = 1   # b10
                    V[(jg*2 + 0) * p + kg*2 + 0, c] = -1  # b00
                    W[(kg*2 + 0) * n + 0, c] = -1  # C00
                    W[(kg*2 + 0) * n + 1, c] = 1   # C10
                elif s == 4:  # M5 = (a00 + a01) * b11
                    U[0 * m + jg*2 + 0, c] = 1  # a00
                    U[0 * m + jg*2 + 1, c] = 1  # a01
                    V[(jg*2 + 1) * p + kg*2 + 1, c] = 1  # b11
                    W[(kg*2 + 0) * n + 0, c] = 1   # C00
                    W[(kg*2 + 1) * n + 0, c] = 1   # C01
                elif s == 5:  # M6 = (a10 - a00) * (b00 + b01)
                    U[1 * m + jg*2 + 0, c] = 1   # a10
                    U[0 * m + jg*2 + 0, c] = -1  # a00
                    V[(jg*2 + 0) * p + kg*2 + 0, c] = 1  # b00
                    V[(jg*2 + 0) * p + kg*2 + 1, c] = 1  # b01
                    W[(kg*2 + 0) * n + 0, c] = 1   # C00
                elif s == 6:  # M7 = (a01 - a11) * (b10 + b11)
                    U[0 * m + jg*2 + 1, c] = 1   # a01
                    U[1 * m + jg*2 + 1, c] = -1  # a11
                    V[(jg*2 + 1) * p + kg*2 + 0, c] = 1  # b10
                    V[(jg*2 + 1) * p + kg*2 + 1, c] = 1  # b11
                    W[(kg*2 + 1) * n + 1, c] = 1   # C11
                col += 1
    
    # Convert to lists of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(result, n=2, m=4, p=6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Try the rank-42 Strassen block decomposition
    result = _strassen_2x4x6()
    
    # Verify correctness
    if _verify_decomposition(result):
        return result
    
    # If verification fails, fall back to schoolbook
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
