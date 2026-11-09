"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import sys

def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 matrix multiplication.
    For 2x2 * 2x2 -> 2x2, using Strassen's 7 multiplications.
    
    The tensor T[i,j,k] for i,j in {0,1} (input indices), k in {0,1} (output col), 
    but actually the standard MM tensor for 2x2:
    T[i,j,k,l] where A is 2x2 (i,j), B is 2x2 (j,k), C is 2x2 (i,k).
    Wait, let me use the convention from the task:
    U has shape (n*m, rank) where n=2, m=2 -> 4 rows
    V has shape (m*p, rank) where m=2, p=2 -> 4 rows  
    W has shape (p*n, rank) where p=2, n=2 -> 4 rows
    
    The trace convention: C[i,k] = sum_j A[i,j] * B[j,k]
    In the tensor decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    For standard MM, T[i,j,k] = 1 for all valid i,j,k (it's the multiplication tensor).
    
    Actually for 2x2 MM, the tensor has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}.
    This is just the 2x2x2 all-ones tensor? No, that's not right.
    
    Let me reconsider. For matrix multiplication A * B where A is n x m, B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The multiplication tensor T is defined by T[i,j,k] = 1 for all i,j,k, and the decomposition
    satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So for 2x2 MM (n=2, m=2, p=2):
    U has shape (4, rank), V has shape (4, rank), W has shape (4, rank)
    And we need sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k in {0,1}
    
    Strassen's algorithm gives rank 7 for 2x2 MM. Let me encode it.
    
    Strassen's 7 products for C = A*B where A, B are 2x2:
    P1 = (A00 + A11)(B00 + B11)
    P2 = (A10 + A11) B00
    P3 = A00 (B10 - B11)
    P4 = A11 (B01 - B00)
    P5 = (A00 + A01) B11
    P6 = (A10 - A00)(B00 + B01)
    P7 = (A01 - A11)(B10 + B11)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    Now I need to express this as a tensor decomposition.
    For each product P_s, we need U, V, W vectors such that
    U[i*2+j] * V[j*2+k] * W[k*2+i] gives the coefficient.
    
    This is getting complex. Let me just use the known rank-7 Strassen decomposition.
    """
    # For 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Using a known explicit decomposition
    # U shape (4,7), V shape (4,7), W shape (4,7)
    
    # Let me construct it from the Strassen formulas
    # Row indices for U: i*2+j for i in {0,1}, j in {0,1} -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Row indices for V: j*2+k for j in {0,1}, k in {0,1} -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Row indices for W: k*2+i for k in {0,1}, i in {0,1} -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # P1 = (A00+A11)(B00+B11)
    # Contribution to U: A00 (row 0) gets +1, A11 (row 3) gets +1
    # Contribution to V: B00 (row 0) gets +1, B11 (row 3) gets +1
    # For W: this product contributes to all C entries
    # C00: +1, C01: 0, C10: 0, C11: +1
    # W[k*2+i]: C00 is W[0*2+0]=W[0], C01 is W[0*2+1]=W[1], C10 is W[1*2+0]=W[2], C11 is W[1*2+1]=W[3]
    # So W[0]=+1, W[1]=0, W[2]=0, W[3]=+1
    
    # P2 = (A10+A11) B00
    # U: A10 (row 2) +1, A11 (row 3) +1
    # V: B00 (row 0) +1
    # C10: +1, C11: -1
    # W[2]=+1, W[3]=-1
    
    # P3 = A00 (B10-B11)
    # U: A00 (row 0) +1
    # V: B10 (row 2) +1, B11 (row 3) -1
    # C01: +1, C11: +1
    # W[1]=+1, W[3]=+1
    
    # P4 = A11 (B01-B00)
    # U: A11 (row 3) +1
    # V: B01 (row 1) +1, B00 (row 0) -1
    # C00: +1, C10: +1
    # W[0]=+1, W[2]=+1
    
    # P5 = (A00+A01) B11
    # U: A00 (row 0) +1, A01 (row 1) +1
    # V: B11 (row 3) +1
    # C00: -1, C01: +1
    # W[0]=-1, W[1]=+1
    
    # P6 = (A10-A00)(B00+B01)
    # U: A10 (row 2) +1, A00 (row 0) -1
    # V: B00 (row 0) +1, B01 (row 1) +1
    # C11: +1
    # W[3]=+1
    
    # P7 = (A01-A11)(B10+B11)
    # U: A01 (row 1) +1, A11 (row 3) -1
    # V: B10 (row 2) +1, B11 (row 3) +1
    # C00: +1, C01: 0
    # W[0]=+1, W[1]=0
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # row 0: (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # row 1: (0,1)
        [0, 1, 0, 0, 0, 1, 0],   # row 2: (1,0)
        [1, 1, 0, 1, 0, 0, -1]   # row 3: (1,1)
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # row 0: (0,0)
        [0, 0, 0, 1, 0, 1, 0],   # row 1: (0,1)
        [0, 0, 1, 0, 0, 0, 1],   # row 2: (1,0)
        [1, 0, -1, 0, 1, 0, 1]   # row 3: (1,1)
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # row 0: (0,0) -> C00
        [0, 0, 1, 0, 1, 0, 0],   # row 1: (0,1) -> C01
        [0, 1, 0, 1, 0, 0, 0],   # row 2: (1,0) -> C10
        [1, -1, 1, 0, 0, 0, 1]   # row 3: (1,1) -> C11
    ]
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Strategy: Decompose 3x4x4 MM using Kronecker/block structure.
    # The 4x4 matrix multiplication can be done with rank 21 (using Strassen-based approaches)
    # or we can use a block decomposition.
    #
    # Actually, let's think about this differently.
    # For 3x4 * 4x4 -> 3x4 result.
    # We can view this as 3 separate 1x4 * 4x4 multiplications.
    # Each 1x4 * 4x4 gives a 1x4 result.
    #
    # For 1x4 * 4x4: A is 1x4, B is 4x4, C is 1x4.
    # C[k] = sum_{j=0}^{3} A[j] * B[j,k]
    # This is a rank 16 tensor (schoolbook) for 1x4x4.
    #
    # Alternatively, we can use the fact that 4x4 MM has known suboptimal ranks.
    # But for 1x4x4, the optimal rank might still be 16 since it's just a dot product structure.
    #
    # Let me try a different approach: use Strassen on 2x2 blocks.
    # Decompose the 4x4 matrix into 2x2 blocks.
    #
    # Actually, let me try to use a known better decomposition.
    # For 3x4x4, the schoolbook rank is 48.
    # Using Strassen on the 4x4 part: 4x4 MM can be done with rank 21 (using 2-level Strassen: 7^2=49, no that's worse)
    # Actually 4x4 via Strassen: split into 2x2 blocks, each 2x2*2x2 uses rank 7.
    # 4x4 = 4 blocks of 2x2 * 2x2, so rank = 4*7 = 28.
    # But we need 3x4 * 4x4, not 4x4 * 4x4.
    #
    # Let me try: split the 4x4 B matrix into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # A is 3x4, split as A = [A_left, A_right] where A_left is 3x2, A_right is 3x2.
    # C = A * B = [A_left * B_left + A_right * B_right, ...]
    # This is getting complicated. Let me just try the simple approach first.
    #
    # Approach: Use rank 28 for the 4x4 multiplication part.
    # For 3x4 * 4x4, think of it as: for each row of A (3 rows), multiply 1x4 * 4x4.
    # Each 1x4 * 4x4 has rank 16 (schoolbook).
    # Total: 3 * 16 = 48 (same as schoolbook).
    #
    # Better approach: Use the structure that C[i,k] = sum_j A[i,j] B[j,k]
    # Group the 4 j-values into two pairs: {0,1} and {2,3}.
    # For each i and k: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # Each sum is a 1x2 * 2x4 partial product.
    #
    # Let me just try to use a known rank-28 decomposition for 4x4 MM and adapt it.
    # Actually, for this problem, let me try a different strategy.
    #
    # Use the fact that we can decompose the 3x4x4 tensor more efficiently.
    # The known best upper bound for 3x4x4 might be around 40-44.
    #
    # Let me try: split A into 3 rows, and for each row use a rank-16 decomposition
    # but share some of the work.
    #
    # Actually, let me just implement the Strassen-based 2x2 block approach for the 4x4 part.
    # 
    # For 1x4 * 4x4 -> 1x4:
    # Split A = [a0, a1, a2, a3] (1x4)
    # Split B into 2x2 blocks: B = [[b00,b01],[b02,b03],[b10,b11],[b12,b13],[b20,b21],[b22,b23],[b30,b31],[b32,b33]]
    # Actually B is 4x4, so split into 2x2 blocks:
    # B = [[B00, B01], [B10, B11]] where each Bi,j is 2x2
    # B00 = [[b00,b01],[b10,b11]], B01 = [[b02,b03],[b12,b13]]
    # B10 = [[b20,b21],[b30,b31]], B11 = [[b22,b23],[b32,b33]]
    #
    # A = [A0, A1] where A0 = [a0,a1] (1x2), A1 = [a2,a3] (1x2)
    #
    # C = A*B = [A0*B00 + A1*B10, A0*B01 + A1*B11] (1x4)
    #
    # Each A_i * B_ij is a 1x2 * 2x2 -> 1x2 multiplication.
    # For 1x2 * 2x2, the rank is 4 (schoolbook) and I don't think it can be improved.
    # So total rank = 4 * 4 = 16 for each row of A.
    # With 3 rows: 48 total. No improvement.
    #
    # Let me try yet another approach. What if I use a rank-15 decomposition for 1x4x4?
    # I'm not sure such exists. Let me just go with a clean implementation of the schoolbook
    # but check if there's a simple improvement.
    #
    # Actually, let me try to use the fact that 4x4 matrix multiplication has a rank of 21
    # (using Laderman's algorithm or similar). If I can use that...
    #
    # For now, let me just return the schoolbook decomposition which gives rank 48.
    # This matches the seed program's score of 1.0.
    
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
