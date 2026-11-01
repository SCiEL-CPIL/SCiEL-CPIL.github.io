"""Matrix multiplication 2x2 x 2x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a (2,2) matrix by a (2,8) matrix.
    
    We use the known result that M(2,2,8) has rank 14 (or use a known construction).
    Actually, for small matrices, we can use the "lattice" method or known decompositions.
    
    A simpler approach: use the schoolbook rank 32, but try to find a lower rank.
    Since we know Strassen-like methods exist, let's construct a rank 28 solution
    by combining 2x2x2 Strassen blocks appropriately.
    
    Actually, for 2x2 * 2x8, we can think of the 2x8 as four 2x2 blocks.
    Using Strassen for each 2x2*2x2 block would give 7*4 = 28 multiplications,
    but we can share some.
    
    Let's use a known rank 14 construction or fall back to a verified lower rank.
    
    For safety and correctness, let's implement a rank 28 solution using
    Strassen's algorithm on the block structure, ensuring the W index convention
    W[k*n + i, r] is correct.
    
    Actually, the best known rank for M(2,2,8) is 14 (Bürgisser 1998, etc.).
    Let's try to construct or verify a rank 14 solution.
    
    Given the complexity, let's use a rank 28 solution which is easier to verify
    and is a significant improvement over 32.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Rank 28 construction using Strassen on 2x2 blocks
    # We view B as a 2x4 matrix of 2x2 blocks: B = [B00 B01 B02 B03; B10 B11 B12 B13]
    # A = [A00 A01; A10 A11]
    # C = A * B, where C is 2x8
    # C[:, 0:2] = A * B[:, 0:2], etc.
    
    # Using Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4 blocks, we need 4*7 = 28 multiplications.
    # But we can potentially share some additions.
    
    # Let's just do 28 separate Strassen blocks for now.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Block indices: A block (i,j) is at row i*m+j in U
    # B block (j,k) where k is block index 0..3, each block is 2x2
    # So B row index = j*8 + k*2 + l where l=0,1 for the 2x2 block
    # C block (i,k) is at row k*2 + l in W (since W[k*n+i, r] with n=2)
    # Wait, W index is W[k*n+i, r] where k is the column block index (0..3) and l is within block
    # So W row = (k*2 + l)*n + i = (k*2+l)*2 + i
    
    # Let's define the Strassen formulas for each block pair
    # For block k (k=0,1,2,3):
    # A is 2x2: A00, A01, A10, A11 (indices 0,1,2,3 in U)
    # B_k is 2x2: B00, B01, B10, B11 (indices k*2, k*2+1, k*2+2, k*2+3 in V rows)
    # C_k is 2x2: C00, C01, C10, C11 (indices in W rows)
    
    strassen_mults = [
        # (name, A_indices, B_indices, C_indices, signs)
        # M1 = (A00+A11)(B00+B11)
        ([(0,1),(2,1)], [(0,1),(2,1)], [(0,1),(3,1)], [1,-1,1,1]),  # C00+=M1, C11-=M1, C01+=M1, C10+=M1? No.
    ]
    
    # Let me be more careful.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each block k (0..3), we have 7 multiplications.
    # Total rank = 28.
    
    def add_to_U(row, col, val):
        U[row][col] += val
    
    def add_to_V(row, col, val):
        V[row][col] += val
    
    def add_to_W(row, col, val):
        W[row][col] += val
    
    for k in range(4):  # block index
        base_v = k * 2  # V row offset for this block
        base_w = k * 4  # W row offset for this block (4 rows per 2x2 block)
        
        # M1 = (A00+A11)(B00+B11)
        col = k * 7 + 0
        add_to_U(0, col, 1)  # A00
        add_to_U(3, col, 1)  # A11
        add_to_V(base_v + 0, col, 1)  # B00
        add_to_V(base_v + 3, col, 1)  # B11
        add_to_W(base_w + 0, col, 1)  # C00 += M1
        add_to_W(base_w + 3, col, 1)  # C11 += M1
        # C01 and C10 don't get M1 directly in standard Strassen... 
        # Wait: C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
        # So M1 goes to C00 and C11.
        # I made an error above. Let me fix.
        
        # Let me redo with correct Strassen output contributions.
        # I'll clear and redo.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(4):
        base_v = k * 2
        base_w = k * 4
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        col = k * 7 + 0
        U[0][col] += 1; U[3][col] += 1
        V[base_v+0][col] += 1; V[base_v+3][col] += 1
        W[base_w+0][col] += 1; W[base_w+3][col] += 1
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        col = k * 7 + 1
        U[2][col] += 1; U[3][col] += 1
        V[base_v+0][col] += 1
        W[base_w+2][col] += 1; W[base_w+3][col] -= 1
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        col = k * 7 + 2
        U[0][col] += 1
        V[base_v+1][col] += 1; V[base_v+3][col] -= 1
        W[base_w+1][col] += 1; W[base_w+3][col] += 1
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        col = k * 7 + 3
        U[3][col] += 1
        V[base_v+2][col] += 1; V[base_v+0][col] -= 1
        W[base_w+0][col] += 1; W[base_w+2][col] += 1
        
        # M5 = (A00+A01)B11 -> C01 += M5, C00 -= M5
        col = k * 7 + 4
        U[0][col] += 1; U[1][col] += 1
        V[base_v+3][col] += 1
        W[base_w+1][col] += 1; W[base_w+0][col] -= 1
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        col = k * 7 + 5
        U[2][col] += 1; U[0][col] -= 1
        V[base_v+0][col] += 1; V[base_v+1][col] += 1
        W[base_w+3][col] += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        col = k * 7 + 6
        U[1][col] += 1; U[3][col] -= 1
        V[base_v+2][col] += 1; V[base_v+3][col] += 1
        W[base_w+0][col] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
