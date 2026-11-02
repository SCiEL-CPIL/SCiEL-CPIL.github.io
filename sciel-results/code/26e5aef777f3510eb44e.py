"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline for safety
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Try to find a lower rank decomposition by exploiting linear dependencies
    # Strategy: Build the tensor as a 3D array and find if we can reduce rank
    # by identifying linearly dependent terms in the standard basis
    
    # The matrix multiplication tensor T has entries T[i,j,k] = A[i,j]*B[j,k]
    # In index notation: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # The standard decomposition uses rank n*m*p terms
    
    # For 2x3x9, let's try to find a rank < 54 decomposition
    # We'll use the approach of finding the tensor rank via null space analysis
    
    # Build the matricization of the tensor
    # T is a tensor of size n x m x p = 2 x 3 x 9
    # Flatten to a matrix of size (n*m) x p = 6 x 9 (mode-1 matricization)
    # But this doesn't directly give tensor rank
    
    # Alternative: Use the fact that for matrix multiplication, we can use
    # known results. The rank of 2x3x9 matrix multiplication is at most 54.
    # Known lower bounds suggest it might be reducible.
    
    # Let's try a constructive approach: 
    # Split the 3x9 matrix B into two parts: B1 (3x4) and B2 (3x5)
    # Then C = A * B = A * B1 + A * B2
    # Rank of 2x3x4 <= 24, Rank of 2x3x5 <= 30, total <= 54 (no improvement)
    
    # Better: Use the known result that rank(2x3x9) can be less than 54
    # by using efficient algorithms. Let's try rank 50 first with Strassen-like approach.
    
    # Actually, let's try to find the minimum rank by checking if rank 48 is possible
    # using a more systematic search.
    
    # For now, let's implement a rank-50 decomposition using block methods
    # Split inner dimension m=3 into m1=2, m2=1
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:3] * B[2:3, :]
    # First term: 2x2x9 matrix multiplication
    # Second term: 2x1x9 matrix multiplication (rank 18)
    
    # For 2x2x9, split p=9 into p1=4, p2=5
    # 2x2x4: can use rank 8 (schoolbook) or potentially less
    # 2x2x5: can use rank 10 (schoolbook) or potentially less
    
    # Let's use Strassen for 2x2x2 blocks within the 2x2x9 multiplication
    # Split 2x2x9 into four 2x2x2 blocks and one 2x2x1 block
    # Each 2x2x2 has rank 7 (Strassen), 2x2x1 has rank 4
    # Total for 2x2x9: 4*7 + 4 = 32
    # Plus 2x1x9: rank 18
    # Total: 32 + 18 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :] which is 2x2x9
    # Split B[0:2, :] into B[0:2, 0:4] and B[0:2, 4:9]
    # B[0:2, 0:4] is 2x2x4, split into two 2x2x2 blocks
    # B[0:2, 4:9] is 2x2x5, split into two 2x2x2 blocks and one 2x2x1 block
    
    # Strassen for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each 2x2x2 block, we use 7 terms
    # For 2x2x4, we have two 2x2x2 blocks, so 14 terms
    # For 2x2x5, we have two 2x2x2 blocks and one 2x2x1 block
    # 2x2x1 has rank 4 (schoolbook)
    # So 2x2x5 has 14 + 4 = 18 terms
    # Total for 2x2x9: 14 + 18 = 32 terms
    
    # Part 2: A[:, 2:3] * B[2:3, :] which is 2x1x9
    # This is just 2*9 = 18 terms (schoolbook)
    
    # Total: 32 + 18 = 50 terms
    
    # Let's build this explicitly
    # Indexing: A has indices (i, j) where i in [0,2), j in [0,3)
    # B has indices (j, k) where j in [0,3), k in [0,9)
    # C has indices (i, k) where i in [0,2), k in [0,9)
    
    # U row index: i*m + j = i*3 + j
    # V row index: j*p + k = j*9 + k
    # W row index: k*n + i = k*2 + i
    
    # Part 1: j in [0,2), k in [0,9)
    # Sub-part 1a: k in [0,4), split into k in [0,2) and k in [2,4)
    # Sub-part 1b: k in [4,9), split into k in [4,6), k in [6,8), k in [8,9)
    
    # For each 2x2x2 block with j in [j0, j0+2), k in [k0, k0+2):
    # We use Strassen's 7 terms
    
    def add_strassen_2x2x2(j0, k0):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[j0 * 3 + 0][col] = 1
        U[(j0+1) * 3 + 1][col] = 1
        V[j0 * 9 + k0][col] = 1
        V[(j0+1) * 9 + (k0+1)][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[(k0+1) * 2 + 0][col] = 1
        W[k0 * 2 + 1][col] = 1
        W[(k0+1) * 2 + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[j0 * 3 + 0][col] = 1
        U[(j0+1) * 3 + 1][col] = 1
        V[j0 * 9 + k0][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[(k0+1) * 2 + 0][col] = 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[j0 * 3 + 0][col] = 1
        V[(j0+1) * 9 + k0][col] = 1
        V[(j0+1) * 9 + (k0+1)][col] = -1
        W[k0 * 2 + 1][col] = 1
        W[(k0+1) * 2 + 1][col] = -1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[(j0+1) * 3 + 1][col] = 1
        V[j0 * 9 + (k0+1)][col] = 1
        V[j0 * 9 + k0][col] = -1
        W[k0 * 2 + 0][col] = 1
        W[(k0+1) * 2 + 0][col] = -1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[j0 * 3 + 0][col] = 1
        U[(j0+1) * 3 + 0][col] = 1
        V[(j0+1) * 9 + (k0+1)][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[(k0+1) * 2 + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[j0 * 3 + 0][col] = -1
        U[(j0+1) * 3 + 0][col] = 1
        V[j0 * 9 + k0][col] = 1
        V[j0 * 9 + (k0+1)][col] = 1
        W[k0 * 2 + 1][col] = 1
        W[(k0+1) * 2 + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[(j0+1) * 3 + 0][col] = 1
        U[(j0+1) * 3 + 1][col] = -1
        V[(j0+1) * 9 + k0][col] = 1
        V[(j0+1) * 9 + (k0+1)][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[(k0+1) * 2 + 0][col] = 1
        col += 1
    
    # Part 1a: 2x2x4 with j in [0,2), k in [0,4)
    # Two 2x2x2 blocks: (j0=0, k0=0) and (j0=0, k0=2)
    add_strassen_2x2x2(0, 0)
    add_strassen_2x2x2(0, 2)
    
    # Part 1b: 2x2x5 with j in [0,2), k in [4,9)
    # Two 2x2x2 blocks: (j0=0, k0=4) and (j0=0, k0=6)
    # One 2x2x1 block: (j0=0, k0=8)
    add_strassen_2x2x2(0, 4)
    add_strassen_2x2x2(0, 6)
    
    # 2x2x1 block at j in [0,2), k=8
    # Schoolbook: 4 terms
    for j in range(2):
        for i in range(2):
            U[i * 3 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 2 + i][col] = 1
            col += 1
    
    # Part 2: 2x1x9 with j=2, k in [0,9)
    # Schoolbook: 18 terms
    for i in range(2):
        for k in range(9):
            U[i * 3 + 2][col] = 1
            V[2 * 9 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check if we're within deadline before returning
    if time.time() < deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: return schoolbook decomposition
        rank_schoolbook = n * m * p
        U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
