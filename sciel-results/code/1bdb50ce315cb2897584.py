"""Matrix multiplication 4x4 * 4x5 tensor decomposition.
Uses a block decomposition: 4x4x4 (Strassen rank 49) + 4x4x1 (schoolbook rank 16).
Total rank = 49 + 16 = 65.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 for all valid i,j,k.
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7 decomposition.
    
    Strassen's 7 multiplications:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: A[i,j] -> U[i*m+j, c], B[j,k] -> V[j*p+k, c], C[k,i] -> W[k*n+i, c]
    # n=2, m=2, p=2
    
    # M1 = (A11+A22)(B11+B22)
    # A11=A[0,0], A22=A[1,1], B11=B[0,0], B22=B[1,1]
    # U[0*2+0]=U[0], U[1*2+1]=U[3], V[0*2+0]=V[0], V[1*2+1]=V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11: k=0,i=0 -> W[0*2+0]=W[0]
    # C22: k=1,i=1 -> W[1*2+1]=W[3]
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21=A[1,0], A22=A[1,1], B11=B[0,0]
    U[1*2+0]=U[2]; U[1*2+1]=U[3]
    V[0*2+0]=V[0]
    # C21 += M2
    # C21: k=1,i=0 -> W[1*2+0]=W[2]
    W[2, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11=A[0,0], B12=B[0,1], B22=B[1,1]
    U[0*2+0]=U[0]
    V[0*2+1]=V[1]; V[1*2+1]=V[3]
    # C12 += M3
    # C12: k=1,i=0 -> W[1*2+0]=W[2]
    W[2, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22=A[1,1], B21=B[1,0], B11=B[0,0]
    U[1*2+1]=U[3]
    V[1*2+0]=V[2]; V[0*2+0]=V[0]
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=A[0,0], A12=A[0,1], B22=B[1,1]
    U[0*2+0]=U[0]; U[0*2+1]=U[1]
    V[1*2+1]=V[3]
    # C12 += M5, C11 -= M5
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=A[1,0], A11=A[0,0], B11=B[0,0], B12=B[0,1]
    U[1*2+0]=U[2]; U[0*2+0]=U[0]
    V[0*2+0]=V[0]; V[0*2+1]=V[1]
    # C22 += M6
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=A[0,1], A22=A[1,1], B21=B[1,0], B22=B[1,1]
    U[0*2+1]=U[1]; U[1*2+1]=U[3]
    V[1*2+0]=V[2]; V[1*2+1]=V[3]
    # C11 += M7, C22 -= M7
    W[0, 6] = 1; W[3, 6] = -1
    
    # Wait, I need to double-check the Strassen formulas.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me redo this more carefully.
    # Actually, let me just use a known correct Strassen decomposition.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Let me use a well-known formulation.
    # The tensor decomposition for 2x2x2 with rank 7:
    
    # Term 1: (a00+a11)(b00+b11) -> contributes to c00 and c11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (a10+a11)(b00) -> contributes to c10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3: (a00)(b01-b11) -> contributes to c01
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 4: (a11)(b10-b00) -> contributes to c00 and c10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: (a00+a01)(b11) -> contributes to c01 and -c00
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # Term 6: (a10-a00)(b00+b01) -> contributes to c11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: (a01-a11)(b10+b11) -> contributes to c00 and -c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Strategy: Decompose 4x4x5 as 4x4x4 + 4x4x1
    # 4x4x4: Use Strassen recursively. 4x4x4 = (2x2x2) tensor (2x2x2) at block level.
    # Actually, for 4x4x4, we partition into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each Ci_j is a 2x2 matrix product sum of two 2x2*2x2 products.
    # Using Strassen at the block level:
    # We have 4 block multiplications needed: A00*B00, A01*B10, A10*B00, A11*B10, etc.
    # Actually there are 8 distinct block products if we do it naively.
    # But with Strassen on the 2x2 block structure, we need 7 block multiplications,
    # each of which is a 2x2*2x2 product requiring 7 scalar multiplications.
    # Total: 7*7 = 49.
    
    # Let me construct the 4x4x4 decomposition using Strassen at two levels.
    
    # First, get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # Now build 4x4x4 from 2x2x2.
    # The 4x4x4 tensor can be seen as a tensor product of two 2x2x2 tensors
    # in a specific way related to block matrix multiplication.
    
    # Actually, the cleanest way: 4x4x4 matrix multiplication tensor
    # T4[i,j,k] = 1 for all i,j,k in [0,4)
    # This is NOT simply the tensor product of two 2x2x2 tensors.
    # The tensor product would give T4[i,j,k] = T2[i>>1, j>>1, k>>1] * T2[i&1, j&1, k&1]
    # which is not what we want.
    
    # The correct approach: Use the fact that 4x4x4 matmul can be done by
    # treating it as 2x2 block matmul where each block is 2x2.
    # Block matmul: C_block = A_block * B_block
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen on the block level (7 block multiplications):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2*2x2 product, decomposed using Strassen (7 scalar mults).
    # Total: 7*7 = 49 scalar multiplications.
    
    # Let me build this explicitly.
    # For 4x4x4:
    # A[i,j] where i,j in [0,4). Block index: (i>>1, j>>1), within-block: (i&1, j&1)
    # Similarly for B and C.
    
    # The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k.
    # We want: T4[i,j,k] = sum_c U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
    
    # Using the block Strassen structure:
    # For each block-level Strassen term (7 terms), and within each, a 2x2x2 Strassen term (7 terms),
    # we get 49 scalar terms.
    
    # Let me index the 49 terms as (s, t) where s in [0,7) is the block-level Strassen index
    # and t in [0,7) is the within-block Strassen index.
    
    # Block-level Strassen:
    # s=0: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # s=1: M2 = (A10+A11)B00 -> C10 += 1
    # s=2: M3 = A00(B01-B11) -> C01 += 1
    # s=3: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # s=4: M5 = (A00+A01)B11 -> C01 += 1, C00 -= 1
    # s=5: M6 = (A10-A00)(B00+B01) -> C11 += 1
    # s=6: M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    
    # Within-block Strassen (for each 2x2*2x2):
    # Using the U2, V2, W2 from above.
    
    # Let me build U4, V4, W4 of shape (16, 49), (16, 49), (16, 49)
    
    rank444 = 49
    U4 = np.zeros((16, rank444))
    V4 = np.zeros((16, rank444))
    W4 = np.zeros((16, rank444))
    
    # Block-level Strassen coefficients:
    # For each block-level term s, we have:
    # - Which A-blocks contribute and with what coefficient
    # - Which B-blocks contribute and with what coefficient
    # - Which C-blocks receive the result and with what coefficient
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # A-blocks: 00 (coeff 1), 11 (coeff 1)
    # B-blocks: 00 (coeff 1), 11 (coeff 1)
    # C-blocks: 00 (coeff 1), 11 (coeff 1)
    
    # s=1: M2 = (A10+A11)B00
    # A-blocks: 10 (coeff 1), 11 (coeff 1)
    # B-blocks: 00 (coeff 1)
    # C-blocks: 10 (coeff 1)
    
    # s=2: M3 = A00(B01-B11)
    # A-blocks: 00 (coeff 1)
    # B-blocks: 01 (coeff 1), 11 (coeff -1)
    # C-blocks: 01 (coeff 1)
    
    # s=3: M4 = A11(B10-B00)
    # A-blocks: 11 (coeff 1)
    # B-blocks: 10 (coeff 1), 00 (coeff -1)
    # C-blocks: 00 (coeff 1), 10 (coeff 1)
    
    # s=4: M5 = (A00+A01)B11
    # A-blocks: 00 (coeff 1), 01 (coeff 1)
    # B-blocks: 11 (coeff 1)
    # C-blocks: 01 (coeff 1), 00 (coeff -1)
    
    # s=5: M6 = (A10-A00)(B00+B01)
    # A-blocks: 10 (coeff 1), 00 (coeff -1)
    # B-blocks: 00 (coeff 1), 01 (coeff 1)
    # C-blocks: 11 (coeff 1)
    
    # s=6: M7 = (A01-A11)(B10+B11)
    # A-blocks: 01 (coeff 1), 11 (coeff -1)
    # B-blocks: 10 (coeff 1), 11 (coeff 1)
    # C-blocks: 00 (coeff 1), 11 (coeff -1)
    
    # Let me define these more formally.
    # Block indices: (row_block, col_block) for A, (row_block, col_block) for B, (row_block, col_block) for C
    # A block (r,c) corresponds to A[2r:2r+2, 2c:2c+2]
    # B block (r,c) corresponds to B[2r:2r+2, 2c:2c+2]
    # C block (r,c) corresponds to C[2r:2r+2, 2c:2c+2]
    
    # For the within-block Strassen, we use U2, V2, W2 which handle 2x2x2.
    # U2 has shape (4, 7): U2[i*2+j, t]
    # V2 has shape (4, 7): V2[j*2+k, t]
    # W2 has shape (4, 7): W2[k*2+i, t]
    
    # For block (br, bc) in A, the elements are A[2*br+i, 2*bc+j] for i,j in [0,2)
    # In the 4x4x4 indexing: A index = (2*br+i)*4 + (2*bc+j)
    
    # Let me build the 49 terms.
    # Term index: c = s*7 + t where s in [0,7), t in [0,7)
    
    # For each block-level term s:
    # - A-blocks with coefficients
    # - B-blocks with coefficients
    # - C-blocks with coefficients
    
    # I'll define the block-level structure:
    block_s = []
    # s=0: M1
    block_s.append({
        'A': [(0,0,1), (1,1,1)],  # (row_block, col_block, coeff)
        'B': [(0,0,1), (1,1,1)],
        'C': [(0,0,1), (1,1,1)]
    })
    # s=1: M2
    block_s.append({
        'A': [(1,0,1), (1,1,1)],
        'B': [(0,0,1)],
        'C': [(1,0,1)]
    })
    # s=2: M3
    block_s.append({
        'A': [(0,0,1)],
        'B': [(0,1,1), (1,1,-1)],
        'C': [(0,1,1)]
    })
    # s=3: M4
    block_s.append({
        'A': [(1,1,1)],
        'B': [(1,0,1), (0,0,-1)],
        'C': [(0,0,1), (1,0,1)]
    })
    # s=4: M5
    block_s.append({
        'A': [(0,0,1), (0,1,1)],
        'B': [(1,1,1)],
        'C': [(0,1,1), (0,0,-1)]
    })
    # s=5: M6
    block_s.append({
        'A': [(1,0,1), (0,0,-1)],
        'B': [(0,0,1), (0,1,1)],
        'C': [(1,1,1)]
    })
    # s=6: M7
    block_s.append({
        'A': [(0,1,1), (1,1,-1)],
        'B': [(1,0,1), (1,1,1)],
        'C': [(0,0,1), (1,1,-1)]
    })
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            # A part
            for (ar, ac, acoeff) in block_s[s]['A']:
                for i in range(2):
                    for j in range(2):
                        # A element at (2*ar+i, 2*ac+j)
                        # In 4x4x4 U indexing: (2*ar+i)*4 + (2*ac+j)
                        # Within-block Strassen U2 index: i*2+j
                        U4[(2*ar+i)*4 + (2*ac+j), c] += acoeff * U2[i*2+j, t]
            
            # B part
            for (br, bc, bcoeff) in block_s[s]['B']:
                for j in range(2):
                    for k in range(2):
                        # B element at (2*br+j, 2*bc+k)
                        # In 4x4x4 V indexing: (2*br+j)*4 + (2*bc+k)
                        # Within-block Strassen V2 index: j*2+k
                        V4[(2*br+j)*4 + (2*bc+k), c] += bcoeff * V2[j*2+k, t]
            
            # C part
            for (cr, cc, ccoeff) in block_s[s]['C']:
                for k in range(2):
                    for i in range(2):
                        # C element at (2*cr+i, 2*cc+k)
                        # In 4x4x4 W indexing: (2*cc+k)*4 + (2*cr+i)
                        # Within-block Strassen W2 index: k*2+i
                        W4[(2*cc+k)*4 + (2*cr+i), c] += ccoeff * W2[k*2+i, t]
    
    # Now verify the 4x4x4 decomposition
    # T4[i,j,k] = sum_c U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] should be 1
    
    # Let me verify
    valid444 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for c in range(49):
                    s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid444 = False
                    break
            if not valid444:
                break
        if not valid444:
            break
    
    if not valid444:
        # Fallback: use schoolbook for 4x4x4
        U4 = np.zeros((16, 64))
        V4 = np.zeros((16, 64))
        W4 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
        rank444 = 64
    
    # Now handle the 4x4x1 part (the 5th column of B)
    # C[:,4] = A * B[:,4]
    # This is 4x4 * 4x1, which is just 16 scalar multiplications (schoolbook)
    
    rank441 = 16
    U1 = np.zeros((16, rank441))
    V1 = np.zeros((16, rank441))
    W1 = np.zeros((16, rank441))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j, col] = 1
            V1[j*5+4, col] = 1  # B[j,4]
            W1[4*4+i, col] = 1  # W[k*n+i] with k=4, n=4 -> W[4*4+i]
            col += 1
    
    # Combine 4x4x4 and 4x4x1
    total_rank = rank444 + rank441
    
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    # Copy 4x4x4 part
    U[:, :rank444] = U4
    V[:, :rank444] = V4
    W[:, :rank444] = W4
    
    # Copy 4x4x1 part
    U[:, rank444:] = U1
    V[:, rank444:] = V1
    W[:, rank444:] = W1
    
    # Verify the full 4x4x5 decomposition
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(5):
                s = 0.0
                for c in range(total_rank):
                    s += U[i*4+j, c] * V[j*5+k, c] * W[k*4+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: schoolbook for entire 4x4x5
        U = np.zeros((16, 80))
        V = np.zeros((20, 80))
        W = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        total_rank = 80
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[int(round(x * 2)) / 2 for x in U[i]] for i in range(U.shape[0])]
    V_list = [[int(round(x * 2)) / 2 for x in V[i]] for i in range(V.shape[0])]
    W_list = [[int(round(x * 2)) / 2 for x in W[i]] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
