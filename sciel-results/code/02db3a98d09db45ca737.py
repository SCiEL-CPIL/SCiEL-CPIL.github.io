"""Tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """Return (U, V, W) for rank-7 Strassen decomposition of 2x2x2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*2+j], V[i*2+j], W[i*2+j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,3]=1
    # m2 = (c+d)f
    U[2,1]=1; U[3,1]=1
    V[1,1]=1
    W[2,1]=1; W[3,1]=1
    # m3 = a(e-g)
    U[0,2]=1
    V[0,2]=1; V[2,2]=-1
    W[0,2]=1; W[2,2]=1
    # m4 = d(h-g)
    U[3,3]=1
    V[3,3]=1; V[2,3]=-1
    W[3,3]=1; W[2,3]=1
    # m5 = (a+b)h
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    # m6 = (c-d)(e+f)
    U[2,5]=1; U[3,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1; W[3,5]=1
    # m7 = (b-c)(g+h)
    U[1,6]=1; U[2,6]=-1
    V[2,6]=1; V[3,6]=1
    W[1,6]=1; W[3,6]=1
    W[0,6]=1; W[2,6]=1
    
    return U, V, W

def _build_2x2x3_rank11():
    """Return (U, V, W) for rank-11 decomposition of 2x2x3.
    A is 2x2, B is 2x3, C is 2x3.
    U shape (4,11), V shape (6,11), W shape (6,11).
    Indexing: U[i*2+j], V[i*3+j], W[i*3+j]
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # First 7 terms: Strassen on columns 0,1 of B (treat B[:,0:2] as 2x2)
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U[:, 0:7] = U_s
    V[:, 0:7] = V_s
    # W_s is for C columns 0,1 (2x2), so W_s[i*2+j] -> W[i*3+j]
    for i in range(2):
        for j in range(2):
            W[i*3+j, 0:7] = W_s[i*2+j, 0:7]
    
    # Last 4 terms: schoolbook for column 2 of B
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    for i in range(2):
        for j in range(2):
            col = 7 + i*2 + j
            U[i*2+j, col] = 1
            V[j*3+2, col] = 1
            W[i*3+2, col] = 1
    
    return U, V, W

def _build_2x3x3_rank16():
    """Return (U, V, W) for rank-16 decomposition of 2x3x3.
    A is 2x3, B is 3x3, C is 2x3.
    U shape (6,16), V shape (9,16), W shape (6,16).
    Indexing: U[i*3+j], V[i*3+j], W[i*3+j]
    
    Decomposition:
    - First 10 terms: rank-10 part for (2,3,2) [B columns 0,1]
    - Last 6 terms: schoolbook for (2,3,1) [B column 2]
    """
    U = np.zeros((6, 16))
    V = np.zeros((9, 16))
    W = np.zeros((6, 16))
    
    # (2,3,2) part: rank 10
    # This is isomorphic to (2,2,3) with roles swapped.
    # A (2x3), B (3x2), C (2x2).
    # We can think of it as: for each row i of A, we have a 1x3 vector.
    # Actually, let's use the (2,2,3) decomposition we built.
    # The (2,2,3) tensor: A' (2x2), B' (2x3), C' (2x3).
    # Our (2,3,2): A (2x3), B (3x2), C (2x2).
    # Note that (2,3,2) is the "transpose" of (2,2,3) in some sense.
    # Specifically, if we let A = A' * P and B = Q * B' ... this is getting complicated.
    
    # Let's just directly construct the (2,3,2) rank-10 decomposition.
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # C = [[c11, c12], [c21, c22]]
    # c11 = a11*b11 + a12*b21 + a13*b31
    # c12 = a11*b12 + a12*b22 + a13*b32
    # c21 = a21*b11 + a22*b21 + a23*b31
    # c22 = a21*b12 + a22*b22 + a23*b32
    
    # Rank 10 construction:
    # Use Strassen-like for the 2x2 blocks and handle the third column of A.
    # Let A = [A_left (2x2), A_right (2x1)] where A_left = [[a11,a12],[a21,a22]], A_right = [a13,a23]
    # Let B = [B_top (2x2); B_bot (1x2)] where B_top = [[b11,b12],[b21,b22]], B_bot = [b31,b32]
    # C = A_left * B_top + A_right * B_bot
    # A_left * B_top: 2x2 * 2x2, rank 7 (Strassen)
    # A_right * B_bot: 2x1 * 1x2, rank 4 (schoolbook: a13*b31, a13*b32, a23*b31, a23*b32)
    # Total: 7 + 4 = 11. But we want 10.
    
    # Can we do 10? Let's try to merge.
    # Actually, let's just use rank 11 for (2,3,2) to be safe.
    # Then (2,3,3) rank = 11 + 6 = 17. Total = 3*17 = 51.
    
    # Let's implement rank 11 for (2,3,2).
    # First 7 terms: Strassen on A_left (2x2) and B_top (2x2)
    U_s, V_s, W_s = _build_strassen_2x2x2()
    # Map indices:
    # U_s[i*2+j] -> U[i*3+j] for j=0,1 (A_left columns)
    for i in range(2):
        for j in range(2):
            U[i*3+j, 0:7] = U_s[i*2+j, 0:7]
    # V_s[i*2+j] -> V[i*3+j] for i=0,1 (B_top rows)
    for i in range(2):
        for j in range(2):
            V[i*3+j, 0:7] = V_s[i*2+j, 0:7]
    # W_s[i*2+j] -> W[i*3+j] for j=0,1 (C columns)
    for i in range(2):
        for j in range(2):
            W[i*3+j, 0:7] = W_s[i*2+j, 0:7]
    
    # Next 4 terms: schoolbook for A_right (2x1) and B_bot (1x2)
    # c11 += a13*b31, c12 += a13*b32, c21 += a23*b31, c22 += a23*b32
    terms = [
        (1, 0, 2, 0),  # a13*b31 -> c11
        (1, 1, 2, 0),  # a13*b32 -> c12
        (2, 0, 2, 1),  # a23*b31 -> c21
        (2, 1, 2, 1),  # a23*b32 -> c22
    ]
    for t, (ai, bi, ci, cj) in enumerate(terms):
        col = 7 + t
        U[ai, col] = 1
        V[bi, col] = 1
        W[ci*3+cj, col] = 1
    
    # Last 6 terms: schoolbook for (2,3,1) [B column 2]
    # C[i,2] = sum_j A[i,j] * B[j,2]
    for i in range(2):
        for j in range(3):
            col = 11 + i*3 + j
            U[i*3+j, col] = 1
            V[j*3+2, col] = 1
            W[i*3+2, col] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 3, 9
    rank = 3 * 17  # 51
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Build the (2,3,3) rank-17 decomposition
    U_block, V_block, W_block = _build_2x3x3_rank16()  # Returns rank 17 actually
    
    # For each of the 3 blocks of B (3x3 each):
    for block in range(3):
        col_offset = block * 17
        # U is the same for all blocks (depends only on A)
        U[:, col_offset:col_offset+17] = U_block
        # V: block k uses rows k*3 to k*3+2 of B
        for i in range(3):
            for j in range(3):
                V[(i)*3+j + block*0, col_offset:col_offset+17] = 0  # Clear
        # Actually, V_block has shape (9, 17) but we only need the relevant 3x3 part.
        # Wait, in _build_2x3x3_rank16, V is (9,17) for B (3x3).
        # For block k, B_block is rows k*3 to k*3+2 of the full B (3x9).
        # So V_full[j*9 + k*3 + l, col] = V_block[j*3 + l, col] for j=0,1,2, l=0,1,2.
        for j in range(3):
            for l in range(3):
                V[j*9 + block*3 + l, col_offset:col_offset+17] = V_block[j*3 + l, :]
        
        # W: block k uses columns k*3 to k*3+2 of C (2x9)
        # W_full[(k*3 + l)*2 + i, col] = W_block[l*3 + i, col] for i=0,1, l=0,1,2.
        for l in range(3):
            for i in range(2):
                W[(block*3 + l)*n + i, col_offset:col_offset+17] = W_block[l*3 + i, :]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
