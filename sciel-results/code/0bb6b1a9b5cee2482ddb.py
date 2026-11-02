import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_factors():
    """
    Return U, V, W factor matrices for 2x2 * 2x2 matrix multiplication
    using Strassen's algorithm (rank 7).
    
    U: (4, 7) - for A matrix elements
    V: (4, 7) - for B matrix elements  
    W: (4, 7) - for C matrix elements
    
    Indexing: A[i,j] at index i*2+j, B[i,j] at i*2+j, C[i,j] at i*2+j
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A10) * (B10 + B11)
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1  # A00 + A11
    V[0, 0] = 1; V[3, 0] = 1  # B00 + B11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1  # A10 + A11
    V[0, 1] = 1               # B00
    
    # M3
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01 - B11
    
    # M4
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10 - B00
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1  # A00 + A01
    V[3, 4] = 1               # B11
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1  # B00 + B01
    
    # M7
    U[3, 6] = 1; U[2, 6] = -1 # A11 - A10
    V[2, 6] = 1; V[3, 6] = 1  # B10 + B11
    
    # Output C = A*B:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1  # C00
    W[1, 2] = 1; W[1, 4] = 1                              # C01
    W[2, 1] = 1; W[2, 3] = 1                              # C10
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1   # C11
    
    return U, V, W

def _winograd_4x4_4x5_factors():
    """
    Construct rank-minimal decomposition for 4x4 * 4x5 matrix multiplication
    using a Winograd-style approach with block decomposition.
    
    Strategy: Split 4x4 * 4x5 into:
    - 4x4 * 4x4 part (use Strassen recursively: rank 49)
    - 4x4 * 4x1 part (schoolbook: rank 16)
    Total rank: 65
    
    Actually, let's use a more sophisticated approach:
    Split into 2x2 blocks. The 4x4 matrix has 4 blocks of 2x2.
    The 4x5 matrix has 4 blocks of 2x(2.5)... this is messy.
    
    Better: Use the fact that we can compute C = A*B where A is 4x4, B is 4x5.
    We can split B into B1 (4x4) and B2 (4x1).
    C = A*B1 + A*B2
    
    For A*B1 (4x4 * 4x4): Use recursive Strassen with rank 49.
    For A*B2 (4x4 * 4x1): This is a matrix-vector product, rank 16 (schoolbook).
    
    Total rank: 49 + 16 = 65.
    
    Let's implement this.
    """
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4 * 4x4 using recursive Strassen (rank 49)
    # We need to construct the factors for this part.
    
    # Part 2: 4x4 * 4x1 using schoolbook (rank 16)
    
    # Let's build the full decomposition.
    rank1 = 49  # for 4x4 * 4x4
    rank2 = 16  # for 4x4 * 4x1
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # --- Part 1: 4x4 * 4x4 (columns 0..48) ---
    # Recursive Strassen: split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B1 = [[B00, B01], [B10, B11]] where each is 2x2
    # C1 = A * B1 = [[C00, C01], [C10, C11]] where each is 2x2
    
    # For each of the 7 Strassen products at the top level,
    # we compute 2x2 * 2x2 using Strassen (7 products each).
    # Total: 7 * 7 = 49 products.
    
    # Top-level Strassen definitions for 4x4:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A10) * (B10 + B11)
    
    # Each Mi is a 2x2 * 2x2 product, computed with Strassen (7 products).
    
    # Let's define the top-level coefficients.
    # For each top-level product t in 0..6, we have:
    # A-block coefficients: which 2x2 blocks of A are involved and their signs
    # B-block coefficients: which 2x2 blocks of B are involved and their signs
    
    top_a_blocks = [
        [(0, 1), (3, 1)],    # M1: A00 + A11
        [(2, 1), (3, 1)],    # M2: A10 + A11
        [(0, 1)],            # M3: A00
        [(3, 1)],            # M4: A11
        [(0, 1), (1, 1)],    # M5: A00 + A01
        [(2, 1), (0, -1)],   # M6: A10 - A00
        [(3, 1), (2, -1)],   # M7: A11 - A10
    ]
    top_b_blocks = [
        [(0, 1), (3, 1)],    # M1: B00 + B11
        [(0, 1)],            # M2: B00
        [(1, 1), (3, -1)],   # M3: B01 - B11
        [(2, 1), (0, -1)],   # M4: B10 - B00
        [(3, 1)],            # M5: B11
        [(0, 1), (1, 1)],    # M6: B00 + B01
        [(2, 1), (3, 1)],    # M7: B10 + B11
    ]
    
    # Output C1 blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    top_c_blocks = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
        [(2, 1), (4, 1)],                    # C01
        [(1, 1), (3, 1)],                    # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    ]
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # For each top-level product t and each internal Strassen product u:
    # The column index in the full decomposition is t*7 + u
    
    for t in range(7):
        for u in range(7):
            col = t * 7 + u
            
            # A-part: sum of contributions from each A-block
            for (block_idx, coeff) in top_a_blocks[t]:
                # block_idx is 0,1,2,3 for A00, A01, A10, A11
                # A-block block_idx occupies rows:
                # A00: 0,1,4,5
                # A01: 2,3,6,7
                # A10: 8,9,12,13
                # A11: 10,11,14,15
                base = block_idx * 4
                for r in range(4):
                    # The r-th element of the 2x2 block corresponds to U2[r, u]
                    U[base + r, col] += coeff * U2[r, u]
            
            # B-part: sum of contributions from each B-block
            for (block_idx, coeff) in top_b_blocks[t]:
                # B-block block_idx occupies rows:
                # B00: 0,1,4,5
                # B01: 2,3,6,7
                # B10: 8,9,12,13
                # B11: 10,11,14,15
                base = block_idx * 4
                for r in range(4):
                    V[base + r, col] += coeff * V2[r, u]
            
            # C-part: sum of contributions from each C-block
            for (block_idx, coeff) in top_c_blocks[t]:
                # C-block block_idx occupies rows:
                # C00: 0,1,4,5
                # C01: 2,3,6,7
                # C10: 8,9,12,13
                # C11: 10,11,14,15
                base = block_idx * 4
                for r in range(4):
                    W[base + r, col] += coeff * W2[r, u]
    
    # --- Part 2: 4x4 * 4x1 (columns 49..64) ---
    # Schoolbook: C2[i,0] = sum_j A[i,j] * B[j,0]
    # This is a matrix-vector product.
    # A is 4x4, B2 is 4x1, C2 is 4x1.
    # Rank 16: for each (i,j) pair, one term.
    
    for i in range(4):
        for j in range(4):
            col = rank1 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1  # B[j,4] (0-indexed, so column 4 of B)
            W[4 * 4 + i, col] = 1  # C[i,4] (0-indexed, so row 4 of C, which is column 0 of the 4x1 part)
    
    return U, V, W

def construct():
    """
    Find a rank-minimal (or low-rank) exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses a hybrid approach:
    - 4x4 * 4x4 part via recursive Strassen (rank 49)
    - 4x4 * 4x1 part via schoolbook (rank 16)
    Total rank: 65
    
    Returns dict with U, V, W as lists of lists.
    """
    n, m, p = DIMENSIONS
    
    U, V, W = _winograd_4x4_4x5_factors()
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
