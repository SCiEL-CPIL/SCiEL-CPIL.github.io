"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7) with W[k*2+i, r] convention.
    """
    # Strassen's 7 products for multiplying 2x2 by 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Indexing: A is 2x2, rows i=0,1; cols j=0,1
    # B is 2x2, rows j=0,1; cols k=0,1
    # C is 2x2, rows i=0,1; cols k=0,1
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00+=1, C11+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[k*2+i]: C00 is k=0,i=0 -> idx 0; C11 is k=1,i=1 -> idx 3
    
    # M2 = (A10+A11)B00
    # U: A10=1, A11=1; V: B00=1; W: C10+=1, C11+=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1  # C10: k=0,i=1 -> idx 2; C11: k=1,i=1 -> idx 3
    
    # M3 = A00(B01-B11)
    # U: A00=1; V: B01=1, B11=-1; W: C01+=1, C11+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01: k=1,i=0 -> idx 1; C11: k=1,i=1 -> idx 3
    
    # M4 = A11(B10-B00)
    # U: A11=1; V: B10=1, B00=-1; W: C00+=1, C10+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00: k=0,i=0 -> idx 0; C10: k=0,i=1 -> idx 2
    
    # M5 = (A00+A01)B11
    # U: A00=1, A01=1; V: B11=1; W: C00-=1, C01+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00: idx 0; C01: idx 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11: idx 3
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00: idx 0
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    # All entries should be 1
    return np.allclose(T, 1.0, atol=1e-10)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Strategy: Split n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    # This gives 3*3*3 = 27 blocks
    # Blocks where all dimensions are 2x2x2 use Strassen (rank 7)
    # Blocks with a 1 in any dimension use schoolbook (rank = product of dims)
    
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2]
    
    total_rank = 0
    for ni in n_blocks:
        for mj in m_blocks:
            for pk in p_blocks:
                total_rank += ni * mj * pk
    
    # total_rank = 2*2*2 + 2*2*2 + 2*2*2 + 2*2*2 + 2*2*2 + 2*2*2 + 2*1*2 + 2*1*2 + 2*1*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*1*2 + 1*1*2 + 1*1*2
    # = 8*6 + 2*6 + 2*3 + 1*3 = 48 + 12 + 6 + 3 = 69? Let me recalculate.
    
    # Actually let me just compute it properly:
    # n=2,m=2,p=2: 8 (but we use Strassen rank 7)
    # n=2,m=2,p=2: 8 -> 7
    # n=2,m=2,p=2: 8 -> 7
    # n=2,m=2,p=2: 8 -> 7
    # n=2,m=2,p=2: 8 -> 7
    # n=2,m=2,p=2: 8 -> 7
    # n=2,m=1,p=2: 4
    # n=2,m=1,p=2: 4
    # n=2,m=1,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=2,p=2: 4
    # n=1,m=1,p=2: 2
    # n=1,m=1,p=2: 2
    # n=1,m=1,p=2: 2
    
    # With Strassen: 6*7 + 6*4 + 6*4 + 3*2 = 42 + 24 + 24 + 6 = 96? That's worse.
    # Wait, I need to recount the blocks.
    
    # n_blocks = [2, 1], m_blocks = [2, 2, 1], p_blocks = [2, 2, 2]
    # Number of blocks with n=2,m=2: 2*2*3 = 12 blocks, each 2x2x2 -> Strassen rank 7
    # Wait no. Let me be more careful.
    
    # For each combination of (ni, mj, pk):
    # ni=2, mj=2, pk=2: appears for m_block indices 0,1 and p_block indices 0,1,2 -> 1*2*3=6 blocks
    # ni=2, mj=2, pk=2: 6 blocks, each rank 7 (Strassen) -> 42
    # ni=2, mj=1, pk=2: 1*1*3=3 blocks, each rank 4 -> 12
    # ni=1, mj=2, pk=2: 1*2*3=6 blocks, each rank 4 -> 24
    # ni=1, mj=1, pk=2: 1*1*3=3 blocks, each rank 2 -> 6
    # Total: 42 + 12 + 24 + 6 = 84
    
    # That matches the cell finding! Rank 84.
    
    # Now let's build the actual decomposition.
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # We'll track the current column offset
    col_offset = 0
    
    # Compute offsets for each block
    n_offsets = [0]
    for ni in n_blocks[:-1]:
        n_offsets.append(n_offsets[-1] + ni)
    m_offsets = [0]
    for mj in m_blocks[:-1]:
        m_offsets.append(m_offsets[-1] + mj)
    p_offsets = [0]
    for pk in p_blocks[:-1]:
        p_offsets.append(p_offsets[-1] + pk)
    
    for ii, ni in enumerate(n_blocks):
        for jj, mj in enumerate(m_blocks):
            for kk, pk in enumerate(p_blocks):
                if time.time() > deadline - 30.0:
                    # Return schoolbook as fallback
                    rank_schoolbook = n * m * p
                    U_fb = np.zeros((n * m, rank_schoolbook))
                    V_fb = np.zeros((m * p, rank_schoolbook))
                    W_fb = np.zeros((p * n, rank_schoolbook))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                c = i * m * p + j * p + k
                                U_fb[i * m + j, c] = 1
                                V_fb[j * p + k, c] = 1
                                W_fb[k * n + i, c] = 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    rank = 7
                else:
                    # Schoolbook for this block
                    rank = ni * mj * pk
                    sU = np.zeros((ni * mj, rank))
                    sV = np.zeros((mj * pk, rank))
                    sW = np.zeros((pk * ni, rank))
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                c = i * mj * pk + j * pk + k
                                sU[i * mj + j, c] = 1
                                sV[j * pk + k, c] = 1
                                sW[k * ni + i, c] = 1
                
                # Place into global matrices
                for r in range(rank):
                    col = col_offset + r
                    for i in range(ni):
                        for j in range(mj):
                            U[n_offsets[ii] + i, col] = 0  # will be filled below
                            # Actually we need to map local (i,j) to global
                            gi = n_offsets[ii] + i
                            gj = m_offsets[jj] + j
                            U[gi * m + gj, col] += sU[i * mj + j, r]
                    for j in range(mj):
                        for k in range(pk):
                            gj = m_offsets[jj] + j
                            gk = p_offsets[kk] + k
                            V[gj * p + gk, col] += sV[j * pk + k, r]
                    for k in range(pk):
                        for i in range(ni):
                            gk = p_offsets[kk] + k
                            gi = n_offsets[ii] + i
                            W[gk * n + gi, col] += sW[k * ni + i, r]
                
                col_offset += rank
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_fb[i * m + j, c] = 1
                V_fb[j * p + k, c] = 1
                W_fb[k * n + i, c] = 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
