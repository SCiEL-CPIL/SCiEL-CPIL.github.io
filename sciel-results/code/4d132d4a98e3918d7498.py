"""Matrix multiplication 2,4,5 rank-minimal decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T should equal 1 for all (i,j,k) since (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2_block():
    """Return U, V, W for 2x2x2 Strassen multiplication, rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # U[i*2+j, c] for A[i,j]
    # V[j*2+k, c] for B[j,k]
    # W[k*2+i, c] for C[k,i]... wait, W[k*n+i, c] with n=2 so W[k*2+i, c]
    
    # c=0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = -1; W[3, 0] = 1
    # C00 += M1, C01 += M1, C10 -= M1, C11 += M1
    # W[k*2+i]: k=0,i=0 -> W[0]; k=0,i=1 -> W[1]; k=1,i=0 -> W[2]; k=1,i=1 -> W[3]
    # C[i,k] = sum_j A[i,j]B[j,k]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me redo with proper W indexing: W[k*n+i, c] where result C[i,k]
    # So W[k*2+i, c] contributes to C[i,k]
    
    # c=0: M1 = (A00+A11)*(B00+B11)
    # C00 += M1, C01 += 0, C10 += 0, C11 += M1
    # Wait, let me recheck Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1   # C00: k=0,i=0
    W[3, 0] = 1   # C11: k=1,i=1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1   # C10: k=1,i=0
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1   # C01: k=0,i=1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1   # C00: k=0,i=0
    W[2, 3] = 1   # C10: k=1,i=0
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00: k=0,i=0
    W[1, 4] = 1   # C01: k=0,i=1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C11: k=1,i=1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00: k=0,i=0
    
    return U, V, W

def _schoolbook_2x2x1():
    """2x2 matrix times 2x1 vector, rank 4.
    U shape (4, 4), V shape (2, 4), W shape (2, 4).
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    # C[i] = sum_j A[i,j] * B[j]
    for i in range(2):
        for j in range(2):
            c = i * 2 + j
            U[i * 2 + j, c] = 1
            V[j, c] = 1
            W[i, c] = 1
    return U, V, W

def _combine_blocks(blocks_U, blocks_V, blocks_W, total_rank):
    """Combine multiple blocks into a single decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for bU, bV, bW in zip(blocks_U, blocks_V, blocks_W):
        r = bU.shape[1]
        U[:, col:col+r] = bU
        V[:, col:col+r] = bV
        W[:, col:col+r] = bW
        col += r
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: decompose 2x4x5 as two 2x2x5 blocks (splitting the 4 columns of A / rows of B)
    # Each 2x2x5 can be decomposed using Strassen on a 2x2x2 part + schoolbook on 2x2x1 part
    # 2x2x2 Strassen: rank 7
    # 2x2x1 schoolbook: rank 4
    # Total per 2x2x5 block: 11
    # Two blocks: 22... wait that's not right.
    
    # Actually, let me think more carefully.
    # A is 2x4, B is 4x5. C = A*B is 2x5.
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x2.
    # Split B = [B1; B2] where B1 is 2x5, B2 is 2x5.
    # C = A1*B1 + A2*B2.
    # Each term is 2x2 times 2x5.
    # 2x2 times 2x5: we can split B into [B1a | B1b] where B1a is 2x2, B1b is 2x3.
    # Or better: use Strassen for the 2x2x2 part and schoolbook for the rest.
    # 2x2 * 2x5 = 2x2*2x2 + 2x2*2x3
    # 2x2*2x2: Strassen rank 7
    # 2x2*2x3: schoolbook rank 2*2*3 = 12... that's worse.
    
    # Alternative: 2x2 * 2x5 directly.
    # We can view this as 5 separate 2x2*2x1 multiplications.
    # Each 2x2*2x1 has rank 4 (schoolbook). Total: 5*4 = 20.
    # Or: group them. 2x2*2x2 (Strassen, rank 7) + 2x2*2x3 (schoolbook, rank 12) = 19.
    # Hmm.
    
    # Actually the known result from the knowledge: rank 36.
    # "Decomposing the 2x4x5 multiplication into two 2x2x5 blocks, each further decomposed 
    # into two 2x2x2 Strassen blocks (rank 7) and one 2x2x1 schoolbook block (rank 4), 
    # reduces the total rank from 40 to 36."
    # So: 2 blocks * (2*7 + 1*4) = 2 * 18 = 36.
    # Wait, that means each 2x2x5 block uses two 2x2x2 Strassen blocks and one 2x2x1 schoolbook?
    # That would be 2*7 + 4 = 18 per block, total 36.
    # But 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1? That's 2+2+1 = 5 columns. Yes!
    
    # So the decomposition is:
    # Split the 4 intermediate dimension into:
    #   A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x5)
    # C = A1*B1 + A2*B2
    # For each 2x2*2x5 product:
    #   Split the 5 columns of B into [B_left | B_mid | B_right] where B_left is 2x2, B_mid is 2x2, B_right is 2x1
    #   2x2*2x2: Strassen (rank 7)
    #   2x2*2x2: Strassen (rank 7)  
    #   2x2*2x1: schoolbook (rank 4)
    #   Total per 2x2x5: 18
    # Grand total: 36
    
    # Let's build this explicitly.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x5: rows j=0,1,2,3; cols k=0,1,2,3,4
    # C is 2x5: rows i=0,1; cols k=0,1,2,3,4
    
    # Block 1: A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x5)
    # Block 2: A2 = A[:, 2:4] (2x2), B2 = B[2:4, :] (2x5)
    
    # For each block, split B's 5 columns into [0:2], [2:4], [4:5]
    
    U_all = []
    V_all = []
    W_all = []
    
    for block_idx in range(2):
        # A_block is 2x2, indexed as A[i, block_idx*2 + j] for i,j in 0,1
        # B_block is 2x5, indexed as B[block_idx*2 + j, k] for j in 0,1, k in 0..4
        
        # Sub-block 1: B columns 0,1 (2x2 part)
        sU, sV, sW = _strassen_2x2x2_block()
        # Remap indices
        # sU[i*2+j, c] -> U[i*4 + (block_idx*2 + j), c]
        # sV[j*2+k, c] -> V[(block_idx*2 + j)*5 + k, c]
        # sW[k*2+i, c] -> W[k*2 + i, c]
        
        U_b = np.zeros((8, 7))
        V_b = np.zeros((10, 7))
        W_b = np.zeros((10, 7))
        
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U_b[i * 4 + block_idx * 2 + j, c] = sU[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V_b[(block_idx * 2 + j) * 5 + k, c] = sV[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W_b[k * 2 + i, c] = sW[k * 2 + i, c]
        
        U_all.append(U_b)
        V_all.append(V_b)
        W_all.append(W_b)
        
        # Sub-block 2: B columns 2,3 (2x2 part)
        sU, sV, sW = _strassen_2x2x2_block()
        U_b = np.zeros((8, 7))
        V_b = np.zeros((10, 7))
        W_b = np.zeros((10, 7))
        
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U_b[i * 4 + block_idx * 2 + j, c] = sU[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V_b[(block_idx * 2 + j) * 5 + (k + 2), c] = sV[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W_b[(k + 2) * 2 + i, c] = sW[k * 2 + i, c]
        
        U_all.append(U_b)
        V_all.append(V_b)
        W_all.append(W_b)
        
        # Sub-block 3: B column 4 (2x1 part)
        sU, sV, sW = _schoolbook_2x2x1()
        U_b = np.zeros((8, 4))
        V_b = np.zeros((10, 4))
        W_b = np.zeros((10, 4))
        
        for i in range(2):
            for j in range(2):
                for c in range(4):
                    U_b[i * 4 + block_idx * 2 + j, c] = sU[i * 2 + j, c]
        for j in range(2):
            for c in range(4):
                V_b[(block_idx * 2 + j) * 5 + 4, c] = sV[j, c]
        for i in range(2):
            for c in range(4):
                W_b[4 * 2 + i, c] = sW[i, c]
        
        U_all.append(U_b)
        V_all.append(V_b)
        W_all.append(W_b)
    
    # Combine all blocks
    total_rank = sum(b.shape[1] for b in U_all)
    U = np.zeros((8, total_rank))
    V = np.zeros((10, total_rank))
    W = np.zeros((10, total_rank))
    col = 0
    for bU, bV, bW in zip(U_all, V_all, W_all):
        r = bU.shape[1]
        U[:, col:col+r] = bU
        V[:, col:col+r] = bV
        W[:, col:col+r] = bW
        col += r
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
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
    
    # Convert to lists of integers
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
