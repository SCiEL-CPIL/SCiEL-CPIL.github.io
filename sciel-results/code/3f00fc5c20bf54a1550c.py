import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # u vectors (A11, A12, A21, A22) indices 0,1,2,3
    u = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1], # M7: A12-A22
    ]
    
    # v vectors (B11, B12, B21, B22) indices 0,1,2,3
    v = [
        [1, 0, 0, 1],  # M1: B11+B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12-B22
        [0, 0, 1, -1], # M4: B21-B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11+B12
        [0, 0, 1, 1],  # M7: B21+B22
    ]
    
    # w vectors (C11, C12, C21, C22) indices 0,1,2,3
    # W[k*2+i, r] where C index = k*2+i
    w = [
        [1, 0, 0, 1],  # M1: C11+C22
        [0, 0, 1, -1], # M2: C21-C22
        [0, 1, 0, 1],  # M3: C12+C22
        [1, 0, 1, 0],  # M4: C11+C21
        [-1, 1, 0, 0], # M5: -C11+C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 0],  # M7: C11
    ]
    
    for r in range(7):
        U[:, r] = u[r]
        V[:, r] = v[r]
        W[:, r] = w[r]
    
    return U, V, W


def _strassen_4x4x4():
    """
    Return U, V, W for 4x4x4 matrix multiplication with rank 49.
    Uses 7 Strassen 2x2x2 subproblems.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # The 4x4 matrix is split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen at 4x4 level uses 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_r is a 2x2 * 2x2 multiplication, which itself uses 7 Strassen terms.
    # So total 7*7 = 49 terms.
    
    # For each top-level Strassen term r (0..6), we need to map:
    # - The A-block combination to A indices
    # - The B-block combination to B indices
    # - The C-block combination to C indices
    
    # Block index mapping:
    # A blocks: A11->(0,0), A12->(0,1), A21->(1,0), A22->(1,1)
    # In flat A index (i*4+j): 
    #   A11: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    #   A12: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    #   A21: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    #   A22: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    # B blocks: B11->(0,0), B12->(0,1), B21->(1,0), B22->(1,1)
    # In flat B index (j*4+k):
    #   B11: j in {0,1}, k in {0,1} -> indices 0,1,4,5
    #   B12: j in {0,1}, k in {2,3} -> indices 2,3,6,7
    #   B21: j in {2,3}, k in {0,1} -> indices 8,9,12,13
    #   B22: j in {2,3}, k in {2,3} -> indices 10,11,14,15
    
    # C blocks: C11->(0,0), C12->(0,1), C21->(1,0), C22->(1,1)
    # In flat C index (k*4+i):
    #   C11: k in {0,1}, i in {0,1} -> indices 0,1,4,5
    #   C12: k in {0,1}, i in {2,3} -> indices 2,3,6,7
    #   C21: k in {2,3}, i in {0,1} -> indices 8,9,12,13
    #   C22: k in {2,3}, i in {2,3} -> indices 10,11,14,15
    
    # For each top-level term r, the sub-problem is:
    # (sum of certain A blocks) * (sum of certain B blocks) -> (sum of certain C blocks)
    
    # Define which blocks are involved in each top-level term:
    # a_blocks[r]: list of (block_idx, coefficient) for A
    # b_blocks[r]: list of (block_idx, coefficient) for B
    # c_blocks[r]: list of (block_idx, coefficient) for C
    
    # Block indices: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
    
    a_blocks = [
        [(0, 1), (3, 1)],   # M1: A11+A22
        [(2, 1), (3, 1)],   # M2: A21+A22
        [(0, 1)],           # M3: A11
        [(3, 1)],           # M4: A22
        [(0, 1), (1, 1)],   # M5: A11+A12
        [(2, 1), (0, -1)],  # M6: A21-A11
        [(1, 1), (3, -1)],  # M7: A12-A22
    ]
    
    b_blocks = [
        [(0, 1), (3, 1)],   # M1: B11+B22
        [(0, 1)],           # M2: B11
        [(1, 1), (3, -1)],  # M3: B12-B22
        [(2, 1), (0, -1)],  # M4: B21-B11
        [(3, 1)],           # M5: B22
        [(0, 1), (1, 1)],   # M6: B11+B12
        [(2, 1), (3, 1)],   # M7: B21+B22
    ]
    
    c_blocks = [
        [(0, 1), (3, 1)],   # M1: C11+C22
        [(2, 1), (3, -1)],  # M2: C21-C22
        [(1, 1), (3, 1)],   # M3: C12+C22
        [(0, 1), (2, 1)],   # M4: C11+C21
        [(0, -1), (1, 1)],  # M5: -C11+C12
        [(3, 1)],           # M6: C22
        [(0, 1)],           # M7: C11
    ]
    
    # For each top-level term r, and each sub-term s (0..6 from 2x2 Strassen):
    # The sub-term s has u2[s], v2[s], w2[s] for the 2x2 sub-problem.
    # We need to expand these into the 4x4 space.
    
    # For a given block combination, the flat indices are:
    def block_flat_indices(block_idx, dim):
        """Return flat indices for a 2x2 block in a 4x4 matrix."""
        # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        row_start = (block_idx // 2) * 2
        col_start = (block_idx % 2) * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                indices.append((row_start + dr) * dim + (col_start + dc))
        return indices
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # U: expand u2[s] over the A blocks in a_blocks[r]
            for block_idx, coeff_a in a_blocks[r]:
                flat_idxs = block_flat_indices(block_idx, 4)
                for local_idx, flat_idx in enumerate(flat_idxs):
                    U4[flat_idx, col] += coeff_a * U2[local_idx, s]
            
            # V: expand v2[s] over the B blocks in b_blocks[r]
            for block_idx, coeff_b in b_blocks[r]:
                flat_idxs = block_flat_indices(block_idx, 4)
                for local_idx, flat_idx in enumerate(flat_idxs):
                    V4[flat_idx, col] += coeff_b * V2[local_idx, s]
            
            # W: expand w2[s] over the C blocks in c_blocks[r]
            for block_idx, coeff_c in c_blocks[r]:
                flat_idxs = block_flat_indices(block_idx, 4)
                for local_idx, flat_idx in enumerate(flat_idxs):
                    W4[flat_idx, col] += coeff_c * W2[local_idx, s]
    
    return U4, V4, W4


def _construct_rank_85():
    """
    Construct a rank-85 decomposition for 4x5*5x5.
    
    Strategy: Split A = [A1(4x4) | A2(4x1)], B = [B1(4x5); B2(1x5)]
    C = A1*B1 + A2*B2
    
    A1*B1: 4x4*4x5. Split B1 = [B1a(4x4) | B1b(4x1)]
      A1*B1a: 4x4*4x4, rank 49 (Strassen)
      A1*B1b: 4x4*4x1, rank 16 (schoolbook)
    A2*B2: 4x1*1x5, rank 20 (schoolbook)
    
    Total: 49 + 16 + 20 = 85
    """
    n, m, p = N, M, P  # 4, 5, 5
    R = 85
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    c = 0
    
    # Part 1: A1*B1a: 4x4*4x4, rank 49
    # A1 is A[0:4, 0:4], B1a is B[0:4, 0:4]
    # U indices: i*5+j for i in 0..3, j in 0..3
    # V indices: j*5+k for j in 0..3, k in 0..3
    # W indices: k*4+i for k in 0..3, i in 0..3
    
    U4, V4, W4 = _strassen_4x4x4()
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, c+r] = U4[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V[j*5+k, c+r] = V4[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[k*4+i, c+r] = W4[k*4+i, r]
    c += 49
    
    # Part 2: A1*B1b: 4x4*4x1, rank 16
    # A1 is A[0:4, 0:4], B1b is B[0:4, 4] (column 4 of B)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U[i*5+j, c] = 1.0
            V[j*5+4, c] = 1.0
            W[4*4+i, c] = 1.0  # k=4, so W[4*4+i]
            c += 1
    
    # Part 3: A2*B2: 4x1*1x5, rank 20
    # A2 is A[0:4, 4] (column 4 of A), B2 is B[4, 0:5] (row 4 of B)
    # Schoolbook: 20 terms
    for i in range(4):
        for k in range(5):
            U[i*5+4, c] = 1.0
            V[4*5+k, c] = 1.0
            W[k*4+i, c] = 1.0
            c += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Try rank-85 decomposition
    try:
        U, V, W = _construct_rank_85()
        if _validate(U, V, W):
            elapsed = time.time() - t0
            if elapsed < DEADLINE_S:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback to schoolbook rank 100
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    
    if not _validate(U, V, W):
        raise ValueError("Schoolbook decomposition failed validation")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
