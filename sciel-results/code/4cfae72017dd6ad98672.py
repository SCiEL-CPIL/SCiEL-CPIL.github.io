"""Matrix multiplication 4x4x8 rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120  # seconds, safety margin before 150s kill

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 = 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i,j] indexes A[i,j] (i in 0..1, j in 0..1)
      V[j,k] indexes B[j,k] (j in 0..1, k in 0..1)
      W[k,i] indexes C[k,i] (k in 0..1, i in 0..1)
    """
    # Strassen products:
    # M1 = A11*(B12+B22)
    # M2 = (A11+A22)*B22
    # M3 = (A11+A12)*B11
    # M4 = A22*(B21-B11)
    # M5 = (A11+A22)*(B11+B22)
    # M6 = (A12-A22)*(B21+B22)
    # M7 = (A11-A12)*(B11+B21)
    #
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 + M1 - M3 - M7
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Index convention: U[i*2+j, col], V[j*2+k, col], W[k*2+i, col]
    # A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    # B11=V[0], B12=V[1], B21=V[2], B22=V[3]
    # C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    # M1 = A11 * (B12 + B22)
    U[0, 0] = 1  # A11
    V[1, 0] = 1  # B12
    V[3, 0] = 1  # B22
    W[1, 0] = 1  # C12 += M1
    W[3, 0] = 1  # C22 += M1
    
    # M2 = (A11 + A22) * B22
    U[0, 1] = 1  # A11
    U[3, 1] = 1  # A22
    V[3, 1] = 1  # B22
    W[0, 1] = -1  # C11 -= M2
    W[1, 1] = 1   # C12 += M2
    
    # M3 = (A11 + A12) * B11
    U[0, 2] = 1  # A11
    U[1, 2] = 1  # A12
    V[0, 2] = 1  # B11
    W[2, 2] = 1  # C21 += M3
    W[3, 2] = -1 # C22 -= M3
    
    # M4 = A22 * (B21 - B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11 += M4
    W[2, 3] = 1  # C21 += M4
    
    # M5 = (A11 + A22) * (B11 + B22)
    U[0, 4] = 1  # A11
    U[3, 4] = 1  # A22
    V[0, 4] = 1  # B11
    V[3, 4] = 1  # B22
    W[0, 4] = 1  # C11 += M5
    W[3, 4] = 1  # C22 += M5
    
    # M6 = (A12 - A22) * (B21 + B22)
    U[1, 5] = 1  # A12
    U[3, 5] = -1 # A22
    V[2, 5] = 1  # B21
    V[3, 5] = 1  # B22
    W[0, 5] = 1  # C11 += M6
    
    # M7 = (A11 - A12) * (B11 + B21)
    U[0, 6] = 1  # A11
    U[1, 6] = -1 # A12
    V[0, 6] = 1  # B11
    V[2, 6] = 1  # B21
    W[3, 6] = -1 # C22 -= M7
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank decomposition for 4x4 * 4x4 = 4x4 using Strassen on 2x2 blocks.
    A (4x4) split into 2x2 blocks: A11, A12, A21, A22
    B (4x4) split into 2x2 blocks: B11, B12, B21, B22
    C (4x4) = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    For each Cij, we have a sum of two 2x2*2x2 products.
    Using Strassen for each product would give 7+7=14 per Cij, total 56.
    But we can do better by sharing some Strassen intermediates.
    
    Actually, let's just use Strassen independently for each of the 4 block products.
    C11 = A11*B11 + A12*B21: use Strassen on (A11,B11) and (A12,B21) separately?
    No, Strassen works on a single 2x2*2x2. For a sum of two such products,
    we can view it as a (2x4)*(4x2) problem, but that's harder.
    
    Simpler: Just use Strassen for each of the 4 output blocks, treating each
    Cij computation as requiring 2 Strassen multiplications (one for each term).
    That gives 4 blocks * 2 terms * 7 rank = 56 rank for 4x4*4x4.
    
    Wait, that's not right either. Each Cij = Ai1*B1j + Ai2*B2j.
    This is the product of a (2x4) matrix [Ai1 Ai2] with a (4x2) matrix [B1j; B2j].
    The schoolbook rank for (2x4x2) is 2*4*2=16.
    Using Strassen-style: we can do it in fewer multiplications.
    
    Let me just use the simple approach: for each of the 4 output 2x2 blocks,
    compute the two 2x2*2x2 products using Strassen (7 each), giving 14 per block,
    total 56 for 4x4*4x4.
    """
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # For 4x4*4x4:
    # U indices: A[i,j] -> i*4+j, i,j in 0..3
    # V indices: B[j,k] -> j*4+k, j,k in 0..3
    # W indices: C[k,i] -> k*4+i, k,i in 0..3
    # 
    # Block structure:
    # A11: rows 0-1, cols 0-1 -> U indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> U indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> U indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> U indices 10,11,14,15
    #
    # B11: rows 0-1, cols 0-1 -> V indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> V indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> V indices 10,11,14,15
    #
    # C11: rows 0-1, cols 0-1 -> W indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> W indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> W indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> W indices 10,11,14,15
    
    total_rank = 56
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((16, total_rank), dtype=float)
    W = np.zeros((16, total_rank), dtype=float)
    
    col = 0
    # For each output block (ci, cj) in {(0,0),(0,1),(1,0),(1,1)}:
    #   Cij = Ai1*B1j + Ai2*B2j
    #   For each term, apply Strassen
    for ci in range(2):
        for cj in range(2):
            # Cij = A_{ci,0} * B_{0,cj} + A_{ci,1} * B_{1,cj}
            for term in range(2):
                # A block: row ci, col term -> rows ci*2..ci*2+1, cols term*2..term*2+1
                # B block: row term, col cj -> rows term*2..term*2+1, cols cj*2..cj*2+1
                # C block: row cj, col ci -> rows cj*2..cj*2+1, cols ci*2..ci*2+1
                
                for s in range(7):
                    # U for A block
                    for r in range(2):
                        for c in range(2):
                            a_idx = (ci*2 + r) * 4 + (term*2 + c)
                            u_val = S_U[r*2+c, s]
                            if u_val != 0:
                                U[a_idx, col] += u_val
                    # V for B block
                    for r in range(2):
                        for c in range(2):
                            b_idx = (term*2 + r) * 4 + (cj*2 + c)
                            v_val = S_V[r*2+c, s]
                            if v_val != 0:
                                V[b_idx, col] += v_val
                    # W for C block
                    for r in range(2):
                        for c in range(2):
                            # C index: k*n + i where k is row of C, i is col of C
                            # C block: rows cj*2..cj*2+1 (these are k), cols ci*2..ci*2+1 (these are i)
                            c_idx = (cj*2 + r) * 4 + (ci*2 + c)
                            w_val = S_W[r*2+c, s]
                            if w_val != 0:
                                W[c_idx, col] += w_val
                    col += 1
    
    return U, V, W

def construct():
    """
    Construct a rank decomposition for 4x4 * 4x8 = 4x8 matrix multiplication.
    C[k,i] = sum_j A[i,j] * B[j,k]
    U shape: (16, rank) for A[i,j] indexed as i*4+j
    V shape: (32, rank) for B[j,k] indexed as j*8+k
    W shape: (32, rank) for C[k,i] indexed as k*4+i (trace convention)
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the 4x4x4 Strassen decomposition (rank 56)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Now extend to 4x4x8 by splitting B into two 4x4 blocks: B = [B1 | B2]
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # B1: j in 0..3, k in 0..3 -> V indices j*8+k for k in 0..3
    # B2: j in 0..3, k in 4..7 -> V indices j*8+k for k in 4..7
    # C1: k in 0..3, i in 0..3 -> W indices k*4+i
    # C2: k in 4..7, i in 0..3 -> W indices k*4+i
    
    # For B1 (first 4 columns of each row of B):
    #   V4[j*4+k] corresponds to B[j,k] for k in 0..3
    #   In the 4x4x8 problem, B[j,k] for k in 0..3 is at V index j*8+k
    # For B2 (last 4 columns):
    #   B[j,k] for k in 4..7 is at V index j*8+k
    #   We can reuse the same U and W structure, just shift V indices
    
    rank = 2 * 56  # 112, using Strassen independently for each half
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First half: C1 = A * B1
    for r in range(56):
        U[:, r] = U4[:, r]
        W[:, r] = W4[:, r]  # C1 indices: k in 0..3, i in 0..3 -> k*4+i
        # B1: j in 0..3, k in 0..3 -> V index j*8+k
        for j in range(4):
            for k in range(4):
                V[j*8+k, r] = V4[j*4+k, r]
    
    # Second half: C2 = A * B2
    for r in range(56):
        U[:, 56+r] = U4[:, r]
        W[:, 56+r] = W4[:, r]  # C2 indices: k in 4..7, i in 0..3 -> k*4+i
        # B2: j in 0..3, k in 4..7 -> V index j*8+k
        # V4[j*4+k] corresponds to B[j,k] where k is local 0..3
        # In 4x4x8, B[j, 4+k] is at V index j*8+(4+k)
        for j in range(4):
            for k in range(4):
                V[j*8+4+k, 56+r] = V4[j*4+k, r]
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
